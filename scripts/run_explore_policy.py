"""Evaluate deterministic explore quotas using the frozen development classifier."""

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
from catboost import CatBoostClassifier, Pool

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
from src.features import ALL_FEATURES, assemble_ranking_features, build_feature_tables  # noqa: E402
from src.ranking import (  # noqa: E402
    apply_explore_quota,
    assert_groups_contiguous,
    attach_groups_and_labels,
    build_candidate_pool,
    candidate_recall_ceiling,
    classify_candidate_novelty,
    ranking_metrics,
    split_group_users,
)
from src.retrieval import (  # noqa: E402
    build_co_visitation,
    co_visitation_candidates,
    personal_repeat_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--component-k", type=int, default=150)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thread-count", type=int, default=10)
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=Path("reports/ranking/full/models/classifier.cbm"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if sys.platform == "darwin" else 1024)


def checkpoint(event: str, **values: Any) -> None:
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


def timed(name: str, function: Callable[..., Any], *values: Any, **kwargs: Any) -> Any:
    checkpoint(f"{name}_started")
    started = perf_counter()
    result = function(*values, **kwargs)
    timings[name] = perf_counter() - started
    checkpoint(f"{name}_finished", step_seconds=round(timings[name], 3))
    return result


def metric_value(
    metrics: pd.DataFrame,
    metric: str,
    segment: str = "overall",
    k: int = 10,
) -> float:
    return float(
        metrics.loc[
            metrics["k"].eq(k)
            & metrics["metric"].eq(metric)
            & metrics["segment"].eq(segment),
            "value",
        ].item()
    )


def summarize_policy_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, selected in metrics.groupby("policy", sort=False):
        rows.append(
            {
                "policy": policy,
                "ndcg_at_10": metric_value(selected, "ndcg"),
                "recall_at_10": metric_value(selected, "recall"),
                "repeat_recall_at_10": metric_value(selected, "recall", "repeat"),
                "explore_recall_at_10": metric_value(selected, "recall", "explore"),
            }
        )
    result = pd.DataFrame(rows)
    baseline = result.loc[result["policy"].eq("unconstrained")].iloc[0]
    for column in (
        "ndcg_at_10",
        "recall_at_10",
        "repeat_recall_at_10",
        "explore_recall_at_10",
    ):
        result[f"delta_{column}"] = result[column] - baseline[column]
    return result


STARTED = perf_counter()
args = parse_args()
if args.component_k != 150:
    raise ValueError("development candidate protocol is frozen at component-k=150")
if args.seed != 42:
    raise ValueError("development split is frozen at seed=42")
if not args.classifier_model.is_absolute():
    args.classifier_model = ROOT / args.classifier_model
if not args.classifier_model.exists():
    raise FileNotFoundError(f"Frozen classifier is missing: {args.classifier_model}")
args.output_dir.mkdir(parents=True, exist_ok=True)
timings: dict[str, float] = {}

checkpoint("run_started", model=str(args.classifier_model))
data = timed(
    "load_data",
    load_instacart,
    args.data_dir,
    tables=("orders", "order_products__prior", "products"),
)
orders = data["orders"]
prior_items = data["order_products__prior"]
products = data["products"]
examples = build_evaluation_examples(orders)
history = timed(
    "materialize_history", history_items, orders, prior_items, examples, "ranker"
)
target_raw = target_items(prior_items, pd.DataFrame(), examples, "ranker")
target = classify_repeat_explore(history, target_raw)
assert_no_target_in_history(history, examples, "ranker")
train_users, validation_users = split_group_users(
    examples["user_id"], validation_fraction=args.validation_fraction, seed=args.seed
)
validation_examples = examples.loc[examples["user_id"].isin(validation_users)].copy()
validation_target = target.loc[target["user_id"].isin(validation_users)].copy()
validation_history = history.loc[history["user_id"].isin(validation_users)]
checkpoint(
    "validation_scope_ready",
    train_users=len(train_users),
    validation_users=len(validation_users),
    visible_interactions=len(history),
)

personal = timed(
    "retrieve_personal",
    personal_repeat_candidates,
    validation_history,
    args.component_k,
)
index = timed(
    "fit_co_visitation", build_co_visitation, history, max_neighbors=args.max_neighbors
)
covis = timed(
    "retrieve_covis",
    co_visitation_candidates,
    history,
    validation_users,
    index,
    args.component_k,
    exclude_seen=True,
    batch_size=args.batch_size,
)
candidate_pool = timed(
    "build_candidate_pool", build_candidate_pool, personal, covis, args.component_k
)
candidate_ceiling = candidate_recall_ceiling(candidate_pool, validation_target)
feature_tables = timed(
    "build_feature_tables", build_feature_tables, history, orders, products, examples
)
labeled_candidates = timed(
    "attach_groups_and_labels",
    attach_groups_and_labels,
    candidate_pool,
    validation_examples,
    validation_target,
)
ranking_data = timed(
    "assemble_ranking_features",
    assemble_ranking_features,
    labeled_candidates,
    feature_tables,
)
assert_groups_contiguous(ranking_data)
checkpoint(
    "ranking_data_ready",
    groups=ranking_data["query_id"].nunique(),
    rows=len(ranking_data),
    positives=int(ranking_data["label"].sum()),
    positive_rate=round(float(ranking_data["label"].mean()), 6),
)

novelty = timed(
    "classify_candidates",
    classify_candidate_novelty,
    ranking_data[["user_id", "target_order_id", "product_id"]],
    history,
)
user_ratios = (
    ranking_data[["user_id", "u_reorder_ratio"]]
    .drop_duplicates("user_id")
    .sort_values("user_id", ignore_index=True)
)
ratio_low, ratio_high = user_ratios["u_reorder_ratio"].quantile([1 / 3, 2 / 3])
user_ratios["repeat_segment"] = pd.Categorical(
    np.select(
        [
            user_ratios["u_reorder_ratio"].le(ratio_low),
            user_ratios["u_reorder_ratio"].le(ratio_high),
        ],
        ["low-repeat", "medium-repeat"],
        default="high-repeat",
    ),
    categories=["low-repeat", "medium-repeat", "high-repeat"],
    ordered=True,
)

categorical = ["aisle_id", "department_id"]
validation_pool = timed(
    "classifier_pool",
    Pool,
    ranking_data[ALL_FEATURES],
    label=ranking_data["label"],
    group_id=ranking_data["query_id"],
    cat_features=categorical,
    feature_names=ALL_FEATURES,
)
classifier = CatBoostClassifier()
classifier.load_model(args.classifier_model)
if classifier.feature_names_ != ALL_FEATURES:
    raise AssertionError("frozen classifier feature contract changed")
checkpoint("classifier_predict_started")
started = perf_counter()
classifier_scores = classifier.predict_proba(validation_pool)[:, 1]
timings["classifier_predict"] = perf_counter() - started
predictions = novelty.copy()
predictions["model_score"] = np.asarray(classifier_scores, dtype=np.float32)
baseline_metrics = ranking_metrics(
    predictions,
    validation_target,
    score_column="model_score",
    ks=(10,),
)
baseline_ndcg = metric_value(baseline_metrics, "ndcg")
if not np.isclose(baseline_ndcg, 0.427566, atol=5e-6):
    raise AssertionError(f"classifier baseline changed: {baseline_ndcg:.8f}")
checkpoint(
    "classifier_predict_finished",
    step_seconds=round(timings["classifier_predict"], 3),
    ndcg_at_10=round(baseline_ndcg, 6),
)

del data, orders, prior_items, products, target_raw, target, validation_history
del personal, index, covis, candidate_pool, feature_tables, labeled_candidates
del history, ranking_data, validation_pool, classifier, classifier_scores, novelty
gc.collect()

policy_metric_frames = []
policy_outputs: dict[str, pd.DataFrame] = {}
diagnostic_rows = []
available_explore = (
    predictions["candidate_type"]
    .astype("string")
    .eq("explore")
    .groupby([predictions["user_id"], predictions["target_order_id"]])
    .sum()
)
for quota in range(4):
    policy = "unconstrained" if quota == 0 else f">={quota} explore"
    selected = timed(
        f"policy_{quota}",
        apply_explore_quota,
        predictions,
        score_column="model_score",
        top_k=10,
        min_explore=quota,
    )
    metrics = ranking_metrics(
        selected,
        validation_target,
        score_column="policy_score",
        ks=(10,),
    )
    metrics.insert(0, "policy", policy)
    policy_metric_frames.append(metrics)
    selected_explore = (
        selected["candidate_type"]
        .astype("string")
        .eq("explore")
        .groupby([selected["user_id"], selected["target_order_id"]])
        .sum()
        .reindex(available_explore.index, fill_value=0)
    )
    expected = available_explore.clip(upper=quota)
    diagnostic_rows.append(
        {
            "policy": policy,
            "selected_rows": int(len(selected)),
            "mean_explore_items_at_10": float(selected_explore.mean()),
            "groups_meeting_available_quota": int(selected_explore.ge(expected).sum()),
            "groups": int(len(expected)),
        }
    )
    policy_outputs[policy] = selected
    checkpoint(
        "policy_evaluated",
        policy=policy,
        ndcg_at_10=round(metric_value(metrics, "ndcg"), 6),
        repeat_recall_at_10=round(metric_value(metrics, "recall", "repeat"), 6),
        explore_recall_at_10=round(metric_value(metrics, "recall", "explore"), 6),
    )

policy_metrics = pd.concat(policy_metric_frames, ignore_index=True)
policy_summary = summarize_policy_metrics(policy_metrics)

segment_rows = []
for segment in user_ratios["repeat_segment"].cat.categories:
    segment_users = user_ratios.loc[
        user_ratios["repeat_segment"].eq(segment), "user_id"
    ]
    segment_target = validation_target.loc[
        validation_target["user_id"].isin(segment_users)
    ]
    for policy, selected in policy_outputs.items():
        segment_metrics = ranking_metrics(
            selected.loc[selected["user_id"].isin(segment_users)],
            segment_target,
            score_column="policy_score",
            ks=(10,),
        )
        segment_rows.append(
            {
                "repeat_segment": str(segment),
                "policy": policy,
                "ndcg_at_10": metric_value(segment_metrics, "ndcg"),
                "recall_at_10": metric_value(segment_metrics, "recall"),
                "repeat_recall_at_10": metric_value(
                    segment_metrics, "recall", "repeat"
                ),
                "explore_recall_at_10": metric_value(
                    segment_metrics, "recall", "explore"
                ),
            }
        )
segment_summary = pd.DataFrame(segment_rows)
segment_baseline = segment_summary.loc[
    segment_summary["policy"].eq("unconstrained")
].set_index("repeat_segment")
for column in (
    "ndcg_at_10",
    "recall_at_10",
    "repeat_recall_at_10",
    "explore_recall_at_10",
):
    segment_summary[f"delta_{column}"] = segment_summary[column] - segment_summary[
        "repeat_segment"
    ].map(segment_baseline[column])

policy_metrics.to_csv(args.output_dir / "explore_policy_metrics.csv", index=False)
policy_summary.to_csv(args.output_dir / "explore_policy_summary.csv", index=False)
segment_summary.to_csv(
    args.output_dir / "explore_policy_segment_summary.csv", index=False
)
pd.DataFrame(diagnostic_rows).to_csv(
    args.output_dir / "explore_policy_diagnostics.csv", index=False
)
candidate_ceiling.to_csv(
    args.output_dir / "candidate_ceiling_validation.csv", index=False
)
user_ratios.to_csv(args.output_dir / "validation_user_segments.csv", index=False)

diagnostics = {
    "run": {
        "wall_seconds": perf_counter() - STARTED,
        "peak_rss_mib": peak_rss_mib(),
        "seed": args.seed,
        "component_k": args.component_k,
        "model": str(args.classifier_model),
        "training_performed": False,
    },
    "hardware": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    },
    "segmentation": {
        "u_reorder_ratio_low_upper": float(ratio_low),
        "u_reorder_ratio_medium_upper": float(ratio_high),
        "users_by_segment": {
            str(key): int(value)
            for key, value in user_ratios["repeat_segment"].value_counts(
                sort=False
            ).items()
        },
    },
    "timings_seconds": timings,
}
(args.output_dir / "explore_policy_diagnostics.json").write_text(
    json.dumps(diagnostics, indent=2, sort_keys=True)
)
checkpoint("run_finished", policies=len(policy_summary), training_performed=False)
print(policy_summary.to_string(index=False), flush=True)
print(segment_summary.to_string(index=False), flush=True)
