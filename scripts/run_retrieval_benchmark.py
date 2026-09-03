"""Run the existing leakage-safe retrieval pipeline with resource diagnostics."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_instacart, resolve_data_dir  # noqa: E402
from src.evaluation import (  # noqa: E402
    assert_no_target_in_history,
    build_evaluation_examples,
    classify_repeat_explore,
    history_items,
    retrieval_recall,
    target_items,
)
from src.retrieval import (  # noqa: E402
    build_co_visitation,
    co_visitation_candidates,
    global_popularity,
    personal_repeat_candidates,
    popularity_candidates,
    union_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--sample-users",
        type=int,
        default=0,
        help="Fixed-seed user sample; 0 means all development users.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-k", type=int, default=100)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return value / divisor


def frame_mib(frame: pd.DataFrame) -> float:
    return frame.memory_usage(deep=True).sum() / 2**20


def checkpoint(event: str, **values: Any) -> None:
    payload = {
        "event": event,
        "elapsed_seconds": round(perf_counter() - STARTED_AT, 3),
        "peak_rss_mib": round(peak_rss_mib(), 1),
        **values,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def timed(name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    checkpoint(f"{name}_started")
    started = perf_counter()
    result = function(*args, **kwargs)
    timings[name] = perf_counter() - started
    checkpoint(f"{name}_finished", step_seconds=round(timings[name], 3))
    return result


def evaluate(name: str, candidates: pd.DataFrame) -> pd.DataFrame:
    metrics = retrieval_recall(candidates, target, METRIC_KS)
    metrics.insert(0, "method", name)
    metric_frames.append(metrics)
    candidate_stats[name] = {
        "rows": int(len(candidates)),
        "memory_mib": frame_mib(candidates),
        "users": int(candidates["user_id"].nunique()),
    }
    checkpoint(
        "method_evaluated",
        method=name,
        candidate_rows=len(candidates),
        candidate_memory_mib=round(frame_mib(candidates), 1),
    )
    return metrics


STARTED_AT = perf_counter()
args = parse_args()
if args.sample_users < 0:
    raise ValueError("--sample-users must be non-negative")
if min(args.max_k, args.max_neighbors, args.batch_size) <= 0:
    raise ValueError("K, neighbors, and batch size must be positive")

METRIC_KS = tuple(k for k in (10, 20, 50, 100) if k <= args.max_k)
if args.max_k not in METRIC_KS:
    METRIC_KS = (*METRIC_KS, args.max_k)

timings: dict[str, float] = {}
candidate_stats: dict[str, dict[str, float | int]] = {}
metric_frames: list[pd.DataFrame] = []
args.output_dir.mkdir(parents=True, exist_ok=True)
data_dir = resolve_data_dir(args.data_dir)
checkpoint(
    "run_started",
    mode="sample" if args.sample_users else "full",
    requested_sample_users=args.sample_users,
    seed=args.seed,
    data_dir=str(data_dir),
)

loaded = timed(
    "load_data",
    load_instacart,
    data_dir,
    tables=("orders", "order_products__prior"),
)
orders = loaded["orders"]
prior_items = loaded["order_products__prior"]
examples = build_evaluation_examples(orders)
all_evaluation_users = len(examples)
if args.sample_users:
    if args.sample_users > all_evaluation_users:
        raise ValueError("sample size exceeds the evaluation population")
    rng = np.random.default_rng(args.seed)
    selected_users = np.sort(
        rng.choice(examples["user_id"].to_numpy(), args.sample_users, replace=False)
    )
    examples = examples.loc[examples["user_id"].isin(selected_users)].reset_index(
        drop=True
    )

history = timed("materialize_history", history_items, orders, prior_items, examples, "ranker")
target_raw = target_items(prior_items, pd.DataFrame(), examples, "ranker")
target = classify_repeat_explore(history, target_raw)
assert_no_target_in_history(history, examples, "ranker")
cutoff = history[["user_id", "order_number"]].merge(
    examples[["user_id", "ranker_history_max_order_number"]],
    on="user_id",
    validate="many_to_one",
)
assert (cutoff["order_number"] <= cutoff["ranker_history_max_order_number"]).all()
assert set(examples["ranker_target_order_id"]).isdisjoint(history["order_id"])

history_stats = {
    "evaluation_users": int(len(examples)),
    "all_evaluation_users": int(all_evaluation_users),
    "history_orders": int(history["order_id"].nunique()),
    "history_items": int(history["product_id"].nunique()),
    "history_interactions": int(len(history)),
    "history_unique_order_items": int(
        history[["order_id", "product_id"]].drop_duplicates().shape[0]
    ),
    "history_memory_mib": frame_mib(history),
    "target_rows": int(len(target)),
    "target_repeat_share": float(target["item_type"].astype("string").eq("repeat").mean()),
}
checkpoint("history_ready", **history_stats)
del loaded, orders, prior_items, target_raw, cutoff
gc.collect()

user_ids = examples["user_id"].to_numpy()
popularity = timed("fit_global_popularity", global_popularity, history)
global_candidates = timed(
    "retrieve_global_popularity",
    popularity_candidates,
    user_ids,
    popularity,
    args.max_k,
    source="global_popularity",
)
evaluate("Global Popularity", global_candidates)
del global_candidates
gc.collect()

personal_candidates = timed(
    "retrieve_personal_repeat",
    personal_repeat_candidates,
    history,
    args.max_k,
    source="personal_repeat",
)
evaluate("Personal Repeat", personal_candidates)

explore_pop_candidates = timed(
    "retrieve_explore_popularity",
    popularity_candidates,
    user_ids,
    popularity,
    args.max_k,
    history=history,
    exclude_seen=True,
    source="explore_popularity",
)
evaluate("Explore Popularity", explore_pop_candidates)
repeat_pop_union = timed(
    "union_repeat_popularity",
    union_candidates,
    [personal_candidates, explore_pop_candidates],
    args.max_k,
    source_priority=["personal_repeat", "explore_popularity"],
)
evaluate("Repeat + Explore Popularity", repeat_pop_union)
del explore_pop_candidates, repeat_pop_union, popularity
gc.collect()

co_visitation = timed(
    "fit_co_visitation",
    build_co_visitation,
    history,
    max_neighbors=args.max_neighbors,
)
co_visitation_stats = {
    "shape": list(co_visitation.similarity.shape),
    "nnz": int(co_visitation.similarity.nnz),
    "memory_mib": (
        co_visitation.similarity.data.nbytes
        + co_visitation.similarity.indices.nbytes
        + co_visitation.similarity.indptr.nbytes
        + co_visitation.item_ids.nbytes
    )
    / 2**20,
}
checkpoint("co_visitation_ready", **co_visitation_stats)

co_vis_candidates = timed(
    "retrieve_co_visitation",
    co_visitation_candidates,
    history,
    user_ids,
    co_visitation,
    args.max_k,
    exclude_seen=True,
    batch_size=args.batch_size,
    source="co_visitation_explore",
)
evaluate("Co-visitation Explore", co_vis_candidates)
repeat_co_vis_union = timed(
    "union_repeat_co_visitation",
    union_candidates,
    [personal_candidates, co_vis_candidates],
    args.max_k,
    source_priority=["personal_repeat", "co_visitation_explore"],
)
evaluate("Repeat + Co-visitation", repeat_co_vis_union)

metrics = pd.concat(metric_frames, ignore_index=True)
run_name = f"sample_{len(examples)}" if args.sample_users else "full"
metrics_path = args.output_dir / f"retrieval_metrics_{run_name}.csv"
diagnostics_path = args.output_dir / f"retrieval_diagnostics_{run_name}.json"
metrics.to_csv(metrics_path, index=False)

diagnostics = {
    "run": {
        "mode": "sample" if args.sample_users else "full",
        "sample_users": int(args.sample_users),
        "seed": int(args.seed),
        "max_k": int(args.max_k),
        "metric_ks": list(METRIC_KS),
        "max_neighbors": int(args.max_neighbors),
        "batch_size": int(args.batch_size),
        "wall_seconds": perf_counter() - STARTED_AT,
        "peak_rss_mib": peak_rss_mib(),
    },
    "hardware": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    },
    "history": history_stats,
    "co_visitation": co_visitation_stats,
    "timings_seconds": timings,
    "candidates": candidate_stats,
}
diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
checkpoint(
    "run_finished",
    metrics_path=str(metrics_path),
    diagnostics_path=str(diagnostics_path),
)
print(metrics.to_string(index=False), flush=True)

