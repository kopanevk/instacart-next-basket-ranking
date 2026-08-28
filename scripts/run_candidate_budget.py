"""Measure score-preserving ranking candidate pools on full development data."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_instacart  # noqa: E402
from src.evaluation import (  # noqa: E402
    assert_no_target_in_history,
    build_evaluation_examples,
    classify_repeat_explore,
    history_items,
    target_items,
)
from src.ranking import build_candidate_pool, candidate_recall_ceiling  # noqa: E402
from src.retrieval import (  # noqa: E402
    build_co_visitation,
    co_visitation_candidates,
    personal_repeat_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 150, 200])
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def checkpoint(event: str, **values: object) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "elapsed_seconds": round(perf_counter() - STARTED, 3),
                "peak_rss_mib": round(peak_rss_mib(), 1),
                **values,
            },
            sort_keys=True,
        ),
        flush=True,
    )


STARTED = perf_counter()
args = parse_args()
budgets = sorted(set(args.budgets))
if not budgets or min(budgets) <= 0:
    raise ValueError("budgets must be positive")
args.output_dir.mkdir(parents=True, exist_ok=True)

checkpoint("load_started")
data = load_instacart(args.data_dir, tables=("orders", "order_products__prior"))
examples = build_evaluation_examples(data["orders"])
history = history_items(
    data["orders"], data["order_products__prior"], examples, "ranker"
)
target_raw = target_items(
    data["order_products__prior"], pd.DataFrame(), examples, "ranker"
)
target = classify_repeat_explore(history, target_raw)
assert_no_target_in_history(history, examples, "ranker")
checkpoint("history_ready", users=len(examples), interactions=len(history))
del data, target_raw
gc.collect()

max_budget = max(budgets)
personal = personal_repeat_candidates(history, max_budget)
checkpoint("personal_ready", rows=len(personal))
index = build_co_visitation(history, max_neighbors=args.max_neighbors)
checkpoint("index_ready", edges=index.similarity.nnz)
covis = co_visitation_candidates(
    history,
    examples["user_id"].to_numpy(),
    index,
    max_budget,
    exclude_seen=True,
    batch_size=args.batch_size,
)
checkpoint("covis_ready", rows=len(covis))

rows = []
diagnostics = []
for budget in budgets:
    started = perf_counter()
    pool = build_candidate_pool(personal, covis, budget)
    counts = pool.groupby("user_id", observed=True).size().reindex(
        examples["user_id"], fill_value=0
    )
    ceiling = candidate_recall_ceiling(pool, target)
    for metric in ceiling.itertuples(index=False):
        rows.append(
            {
                "component_k": budget,
                "segment": metric.segment,
                "recall": metric.recall,
                "users": metric.users,
            }
        )
    diagnostics.append(
        {
            "component_k": budget,
            "candidate_rows": int(len(pool)),
            "mean_candidates": float(counts.mean()),
            "p50_candidates": float(counts.quantile(0.50)),
            "p95_candidates": float(counts.quantile(0.95)),
            "pool_memory_mib": float(pool.memory_usage(deep=True).sum() / 2**20),
            "build_seconds": perf_counter() - started,
        }
    )
    checkpoint("budget_finished", **diagnostics[-1])
    del pool, counts
    gc.collect()

metrics = pd.DataFrame(rows)
diagnostics_frame = pd.DataFrame(diagnostics)
metrics.to_csv(args.output_dir / "candidate_budget_recall.csv", index=False)
diagnostics_frame.to_csv(args.output_dir / "candidate_budget_cost.csv", index=False)
summary = {
    "run": {
        "wall_seconds": perf_counter() - STARTED,
        "peak_rss_mib": peak_rss_mib(),
        "budgets": budgets,
        "max_neighbors": args.max_neighbors,
        "batch_size": args.batch_size,
    },
    "personal_rows_at_max_budget": int(len(personal)),
    "covis_rows_at_max_budget": int(len(covis)),
    "retained_covis_edges": int(index.similarity.nnz),
}
(args.output_dir / "candidate_budget_diagnostics.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True)
)
print(diagnostics_frame.to_string(index=False), flush=True)
print(metrics.to_string(index=False), flush=True)

