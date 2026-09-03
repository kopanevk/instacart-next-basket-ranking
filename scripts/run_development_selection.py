"""Confirm YetiRank convergence and measure a constrained explore policy."""

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
from catboost import CatBoostClassifier, CatBoostRanker, Pool

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
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--thread-count", type=int, default=10)
    parser.add_argument("--external-curve-period", type=int, default=10)
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


def frame_mib(frame: pd.DataFrame) -> float:
    return frame.memory_usage(deep=True).sum() / 2**20


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


def make_pool(feature_names: list[str], row_mask: np.ndarray) -> Pool:
    selected = ranking_data.loc[row_mask, ["query_id", "label", *feature_names]]
    assert_groups_contiguous(selected)
    categorical = [
        feature for feature in ("aisle_id", "department_id") if feature in feature_names
    ]
    return Pool(
        selected[feature_names],
        label=selected["label"],
        group_id=selected["query_id"],
        cat_features=categorical,
        feature_names=feature_names,
    )


def prediction_frame(scores: np.ndarray) -> pd.DataFrame:
    result = validation_keys.copy()
    result["model_score"] = np.asarray(scores, dtype=np.float32)
    return result


def evaluate_scores(scores: np.ndarray, *, ks: tuple[int, ...] = (10, 20)) -> pd.DataFrame:
    return ranking_metrics(
        prediction_frame(scores),
        validation_target,
        score_column="model_score",
        ks=ks,
    )


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


def curve_metric(evals_result: dict[str, dict[str, list[float]]]) -> tuple[str, list[float]]:
    validation_name = next(name for name in evals_result if name != "learn")
    metrics = evals_result[validation_name]
    metric_name = next(name for name in metrics if name.startswith("NDCG"))
    return metric_name, metrics[metric_name]


def policy_summary(metrics: pd.DataFrame) -> pd.DataFrame:
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
if args.iterations <= 30:
    raise ValueError("convergence run must extend the previous 30-tree cap")
if args.external_curve_period <= 0:
    raise ValueError("external-curve-period must be positive")
if not args.classifier_model.is_absolute():
    args.classifier_model = ROOT / args.classifier_model
if not args.classifier_model.exists():
    raise FileNotFoundError(
        f"Frozen development classifier is missing: {args.classifier_model}"
    )
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "models").mkdir(exist_ok=True)
(args.output_dir / "catboost_info").mkdir(exist_ok=True)
timings: dict[str, float] = {}

checkpoint(
    "run_started",
    iterations=args.iterations,
    early_stopping_rounds=args.early_stopping_rounds,
    learning_rate=args.learning_rate,
)
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
    "materialize_history",
    history_items,
    orders,
    prior_items,
    examples,
    "ranker",
)
target_raw = target_items(prior_items, pd.DataFrame(), examples, "ranker")
target = classify_repeat_explore(history, target_raw)
assert_no_target_in_history(history, examples, "ranker")
checkpoint("history_ready", users=len(examples), interactions=len(history))

personal = timed(
    "retrieve_personal", personal_repeat_candidates, history, args.component_k
)
index = timed(
    "fit_co_visitation", build_co_visitation, history, max_neighbors=args.max_neighbors
)
covis = timed(
    "retrieve_covis",
    co_visitation_candidates,
    history,
    examples["user_id"].to_numpy(),
    index,
    args.component_k,
    exclude_seen=True,
    batch_size=args.batch_size,
)
candidate_pool = timed(
    "build_candidate_pool", build_candidate_pool, personal, covis, args.component_k
)
feature_tables = timed(
    "build_feature_tables", build_feature_tables, history, orders, products, examples
)
labeled_candidates = timed(
    "attach_groups_and_labels",
    attach_groups_and_labels,
    candidate_pool,
    examples,
    target,
)
ranking_data = timed(
    "assemble_ranking_features",
    assemble_ranking_features,
    labeled_candidates,
    feature_tables,
)
assert_groups_contiguous(ranking_data)

train_users, validation_users = split_group_users(
    examples["user_id"], validation_fraction=args.validation_fraction, seed=args.seed
)
train_mask = ranking_data["user_id"].isin(train_users).to_numpy()
validation_mask = ranking_data["user_id"].isin(validation_users).to_numpy()
validation_target = target.loc[target["user_id"].isin(validation_users)].copy()
validation_keys = ranking_data.loc[
    validation_mask, ["user_id", "target_order_id", "product_id"]
].reset_index(drop=True)
validation_ceiling = candidate_recall_ceiling(validation_keys, validation_target)
user_ratios = (
    ranking_data.loc[validation_mask, ["user_id", "u_reorder_ratio"]]
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
dataset_stats = {
    "groups": int(ranking_data["query_id"].nunique()),
    "candidate_rows": int(len(ranking_data)),
    "positive_rows": int(ranking_data["label"].sum()),
    "positive_rate": float(ranking_data["label"].mean()),
    "train_users": int(len(train_users)),
    "validation_users": int(len(validation_users)),
    "train_rows": int(train_mask.sum()),
    "validation_rows": int(validation_mask.sum()),
    "memory_mib": frame_mib(ranking_data),
}
checkpoint("ranking_data_ready", **dataset_stats)

validation_novelty = timed(
    "classify_validation_candidates",
    classify_candidate_novelty,
    validation_keys,
    history,
)
del data, orders, prior_items, products, target_raw
del personal, covis, index, candidate_pool, labeled_candidates, feature_tables, history
gc.collect()

train_pool = timed("ranker_train_pool", make_pool, ALL_FEATURES, train_mask)
validation_pool = timed(
    "ranker_validation_pool", make_pool, ALL_FEATURES, validation_mask
)
ranker = CatBoostRanker(
    iterations=args.iterations,
    depth=args.depth,
    learning_rate=args.learning_rate,
    loss_function="YetiRank",
    eval_metric="NDCG:top=10",
    random_seed=args.seed,
    thread_count=args.thread_count,
    od_type="Iter",
    od_wait=args.early_stopping_rounds,
    allow_writing_files=True,
    train_dir=str(args.output_dir / "catboost_info"),
    save_snapshot=True,
    snapshot_file="yetirank_snapshot.cbsnapshot",
    snapshot_interval=60,
    verbose=10,
)
checkpoint("yetirank_fit_started", features=len(ALL_FEATURES))
started = perf_counter()
ranker.fit(train_pool, eval_set=validation_pool, use_best_model=False)
timings["yetirank_fit"] = perf_counter() - started
evals_result = ranker.get_evals_result()
internal_metric_name, internal_values = curve_metric(evals_result)
internal_best_zero_based = int(ranker.get_best_iteration())
internal_best_trees = internal_best_zero_based + 1
checkpoint(
    "yetirank_fit_finished",
    step_seconds=round(timings["yetirank_fit"], 3),
    trained_trees=ranker.tree_count_,
    internal_best_iteration=internal_best_zero_based,
    internal_best_ndcg=round(float(internal_values[internal_best_zero_based]), 7),
)

internal_curve = pd.DataFrame(
    {
        "iteration": np.arange(1, len(internal_values) + 1),
        "candidate_relative_validation_ndcg_at_10": internal_values,
    }
)
internal_curve.to_csv(args.output_dir / "yetirank_learning_curve.csv", index=False)

curve_tree_counts = sorted(
    set(
        [1, internal_best_trees, ranker.tree_count_]
        + list(
            range(
                args.external_curve_period,
                ranker.tree_count_ + 1,
                args.external_curve_period,
            )
        )
    )
)
external_rows = []
checkpoint("external_curve_started", checkpoints=len(curve_tree_counts))
started = perf_counter()
for tree_count in curve_tree_counts:
    scores = ranker.predict(validation_pool, ntree_end=tree_count)
    metrics_at_iteration = evaluate_scores(scores, ks=(10,))
    external_rows.append(
        {
            "trees": tree_count,
            "full_target_ndcg_at_10": metric_value(metrics_at_iteration, "ndcg"),
            "full_target_recall_at_10": metric_value(metrics_at_iteration, "recall"),
            "repeat_recall_at_10": metric_value(
                metrics_at_iteration, "recall", "repeat"
            ),
            "explore_recall_at_10": metric_value(
                metrics_at_iteration, "recall", "explore"
            ),
        }
    )
    checkpoint(
        "external_curve_checkpoint",
        trees=tree_count,
        ndcg_at_10=round(external_rows[-1]["full_target_ndcg_at_10"], 6),
    )
    del scores, metrics_at_iteration
    gc.collect()
timings["external_curve"] = perf_counter() - started
external_curve = pd.DataFrame(external_rows)
external_curve.to_csv(args.output_dir / "yetirank_external_curve.csv", index=False)
selected_curve_row = external_curve.loc[
    external_curve["full_target_ndcg_at_10"].idxmax()
]
selected_trees = int(selected_curve_row["trees"])

ranker_scores = ranker.predict(validation_pool, ntree_end=selected_trees)
ranker_metrics = evaluate_scores(ranker_scores)
ranker_metrics.insert(0, "method", "CatBoostRanker converged")
selected_ranker = ranker.copy()
selected_ranker.shrink(selected_trees)
selected_ranker.save_model(args.output_dir / "models" / "ranker_converged.cbm")
del ranker_scores, selected_ranker, ranker, train_pool
gc.collect()

classifier = CatBoostClassifier()
classifier.load_model(args.classifier_model)
if classifier.feature_names_ != ALL_FEATURES:
    raise AssertionError("frozen classifier feature contract does not match ALL_FEATURES")
checkpoint("classifier_prediction_started")
started = perf_counter()
classifier_scores = classifier.predict_proba(validation_pool)[:, 1]
timings["classifier_prediction"] = perf_counter() - started
classifier_metrics = evaluate_scores(classifier_scores)
classifier_metrics.insert(0, "method", "CatBoostClassifier")
classifier_ndcg = metric_value(classifier_metrics, "ndcg")
if not np.isclose(classifier_ndcg, 0.427566, atol=5e-6):
    raise AssertionError(
        f"frozen classifier baseline changed unexpectedly: {classifier_ndcg:.8f}"
    )
checkpoint(
    "classifier_prediction_finished",
    step_seconds=round(timings["classifier_prediction"], 3),
    ndcg_at_10=round(classifier_ndcg, 6),
)
del classifier, validation_pool
gc.collect()

policy_candidates = validation_novelty.copy()
policy_candidates["model_score"] = np.asarray(classifier_scores, dtype=np.float32)
del classifier_scores, validation_novelty, ranking_data, train_mask, validation_mask
gc.collect()

policy_metric_frames = []
policy_outputs: dict[str, pd.DataFrame] = {}
policy_diagnostics = []
for quota in range(4):
    policy = "unconstrained" if quota == 0 else f">={quota} explore"
    selected = timed(
        f"policy_quota_{quota}",
        apply_explore_quota,
        policy_candidates,
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
    selected_counts = (
        selected["candidate_type"]
        .astype("string")
        .eq("explore")
        .groupby([selected[column] for column in ("user_id", "target_order_id")])
        .sum()
    )
    available_counts = (
        policy_candidates["candidate_type"]
        .astype("string")
        .eq("explore")
        .groupby(
            [
                policy_candidates[column]
                for column in ("user_id", "target_order_id")
            ]
        )
        .sum()
    )
    expected = available_counts.clip(upper=quota)
    policy_diagnostics.append(
        {
            "policy": policy,
            "rows": int(len(selected)),
            "mean_explore_items": float(selected_counts.mean()),
            "groups_meeting_available_quota": int(
                selected_counts.reindex(expected.index, fill_value=0).ge(expected).sum()
            ),
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
policy_summary_frame = policy_summary(policy_metrics)
policy_metrics.to_csv(args.output_dir / "explore_policy_metrics.csv", index=False)
policy_summary_frame.to_csv(
    args.output_dir / "explore_policy_summary.csv", index=False
)
pd.DataFrame(policy_diagnostics).to_csv(
    args.output_dir / "explore_policy_diagnostics.csv", index=False
)

segment_metric_frames = []
for segment in user_ratios["repeat_segment"].cat.categories:
    segment_users = user_ratios.loc[
        user_ratios["repeat_segment"].eq(segment), "user_id"
    ]
    segment_target = validation_target.loc[
        validation_target["user_id"].isin(segment_users)
    ]
    for policy, selected in policy_outputs.items():
        segment_predictions = selected.loc[selected["user_id"].isin(segment_users)]
        metrics = ranking_metrics(
            segment_predictions,
            segment_target,
            score_column="policy_score",
            ks=(10,),
        )
        metrics.insert(0, "policy", policy)
        metrics.insert(0, "repeat_segment", segment)
        segment_metric_frames.append(metrics)

segment_metrics = pd.concat(segment_metric_frames, ignore_index=True)
segment_summary_rows = []
for (segment, policy), selected in segment_metrics.groupby(
    ["repeat_segment", "policy"], sort=False
):
    segment_summary_rows.append(
        {
            "repeat_segment": segment,
            "policy": policy,
            "ndcg_at_10": metric_value(selected, "ndcg"),
            "recall_at_10": metric_value(selected, "recall"),
            "repeat_recall_at_10": metric_value(selected, "recall", "repeat"),
            "explore_recall_at_10": metric_value(selected, "recall", "explore"),
        }
    )
segment_summary = pd.DataFrame(segment_summary_rows)
baseline_by_segment = segment_summary.loc[
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
    ].map(baseline_by_segment[column])
segment_metrics.to_csv(
    args.output_dir / "explore_policy_segment_metrics.csv", index=False
)
segment_summary.to_csv(
    args.output_dir / "explore_policy_segment_summary.csv", index=False
)
user_ratios.to_csv(args.output_dir / "validation_user_segments.csv", index=False)

model_comparison = pd.concat([classifier_metrics, ranker_metrics], ignore_index=True)
model_comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
validation_ceiling.to_csv(args.output_dir / "candidate_ceiling_validation.csv", index=False)

diagnostics = {
    "run": {
        "wall_seconds": perf_counter() - STARTED,
        "peak_rss_mib": peak_rss_mib(),
        "iterations_cap": args.iterations,
        "early_stopping_rounds": args.early_stopping_rounds,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "component_k": args.component_k,
    },
    "hardware": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    },
    "dataset": dataset_stats,
    "convergence": {
        "trained_trees": int(len(internal_values)),
        "catboost_internal_metric": internal_metric_name,
        "catboost_best_iteration_zero_based": internal_best_zero_based,
        "catboost_best_trees": internal_best_trees,
        "catboost_best_candidate_relative_ndcg_at_10": float(
            internal_values[internal_best_zero_based]
        ),
        "selected_external_trees": selected_trees,
        "selected_full_target_ndcg_at_10": float(
            selected_curve_row["full_target_ndcg_at_10"]
        ),
        "classifier_full_target_ndcg_at_10": classifier_ndcg,
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
(args.output_dir / "development_selection_diagnostics.json").write_text(
    json.dumps(diagnostics, indent=2, sort_keys=True)
)
checkpoint(
    "run_finished",
    selected_trees=selected_trees,
    ranker_ndcg_at_10=round(
        metric_value(ranker_metrics, "ndcg"), 6
    ),
    classifier_ndcg_at_10=round(classifier_ndcg, 6),
)
print(model_comparison.to_string(index=False), flush=True)
print(policy_summary_frame.to_string(index=False), flush=True)
print(segment_summary.to_string(index=False), flush=True)
