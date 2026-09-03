"""Run the single frozen-MVP evaluation on the sealed final Instacart order."""

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
    assert_protocol_integrity,
    build_evaluation_examples,
    classify_repeat_explore,
    history_items,
    target_items,
)
from src.features import ALL_FEATURES, assemble_ranking_features, build_feature_tables  # noqa: E402
from src.final import (  # noqa: E402
    FROZEN_BATCH_SIZE,
    FROZEN_CLASSIFIER_PARAMS,
    FROZEN_COMPONENT_K,
    FROZEN_FEATURES,
    FROZEN_MAX_NEIGHBORS,
    FROZEN_TOP_K,
    assert_frozen_feature_contract,
    reorder_ratio_segment,
    target_basket_segment,
)
from src.ranking import (  # noqa: E402
    apply_explore_quota,
    assert_groups_contiguous,
    attach_groups_and_labels,
    build_candidate_pool,
    candidate_recall_ceiling,
    classify_candidate_novelty,
    ranking_metrics,
)
from src.retrieval import (  # noqa: E402
    build_co_visitation,
    co_visitation_candidates,
    personal_repeat_candidates,
)


DEVELOPMENT_METRICS = {
    "candidate_recall": 0.684273,
    "ndcg_at_10": 0.427566,
    "recall_at_10": 0.357336,
    "repeat_recall_at_10": 0.599810,
    "explore_recall_at_10": 0.002909,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/final"))
    parser.add_argument("--thread-count", type=int, default=10)
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


def retrieve_frozen_pool(
    history: pd.DataFrame,
    user_ids: np.ndarray,
    prefix: str,
) -> tuple[pd.DataFrame, int]:
    personal = timed(
        f"{prefix}_retrieve_personal",
        personal_repeat_candidates,
        history,
        FROZEN_COMPONENT_K,
    )
    index = timed(
        f"{prefix}_fit_co_visitation",
        build_co_visitation,
        history,
        max_neighbors=FROZEN_MAX_NEIGHBORS,
    )
    covis = timed(
        f"{prefix}_retrieve_covis",
        co_visitation_candidates,
        history,
        user_ids,
        index,
        FROZEN_COMPONENT_K,
        exclude_seen=True,
        batch_size=FROZEN_BATCH_SIZE,
    )
    result = timed(
        f"{prefix}_build_candidate_pool",
        build_candidate_pool,
        personal,
        covis,
        FROZEN_COMPONENT_K,
    )
    retained_edges = int(index.similarity.nnz)
    if result["repeat_rank"].dropna().gt(FROZEN_COMPONENT_K).any():
        raise AssertionError("repeat candidates exceed frozen component K")
    if result["covis_rank"].dropna().gt(FROZEN_COMPONENT_K).any():
        raise AssertionError("co-visitation candidates exceed frozen component K")
    del personal, covis, index
    gc.collect()
    return result, retained_edges


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


def combined_final_metrics(
    ceiling: pd.DataFrame,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    ceiling_rows = ceiling.rename(columns={"recall": "value", "users": "groups"})
    ceiling_rows = ceiling_rows.assign(
        evaluation="candidate_ceiling",
        method="Frozen candidate union",
        k=pd.array([pd.NA] * len(ceiling_rows), dtype="Int64"),
        metric="recall",
    )
    ranking_rows = ranking.assign(
        evaluation="ranking",
        method="CatBoostClassifier",
        k=ranking["k"].astype("Int64"),
    )
    columns = ["evaluation", "method", "k", "metric", "segment", "value", "groups"]
    return pd.concat(
        [ceiling_rows[columns], ranking_rows[columns]], ignore_index=True
    )


def segment_result(
    dimension: str,
    segment: str,
    user_ids: pd.Series,
    pool: pd.DataFrame,
    top_ten: pd.DataFrame,
    target: pd.DataFrame,
) -> dict[str, Any]:
    selected_target = target.loc[target["user_id"].isin(user_ids)]
    selected_predictions = top_ten.loc[top_ten["user_id"].isin(user_ids)]
    selected_pool = pool.loc[pool["user_id"].isin(user_ids)]
    metrics = ranking_metrics(
        selected_predictions,
        selected_target,
        score_column="policy_score",
        ks=(10,),
    )
    ceiling = candidate_recall_ceiling(selected_pool, selected_target).set_index(
        "segment"
    )
    return {
        "dimension": dimension,
        "segment": segment,
        "users": int(selected_target["user_id"].nunique()),
        "candidate_recall": float(ceiling.loc["overall", "recall"]),
        "candidate_repeat_recall": float(ceiling.loc["repeat", "recall"]),
        "candidate_explore_recall": float(ceiling.loc["explore", "recall"]),
        "ndcg_at_10": metric_value(metrics, "ndcg"),
        "recall_at_10": metric_value(metrics, "recall"),
        "repeat_recall_at_10": metric_value(metrics, "recall", "repeat"),
        "explore_recall_at_10": metric_value(metrics, "recall", "explore"),
    }


STARTED = perf_counter()
args = parse_args()
args.output_dir = args.output_dir.resolve()
if (args.output_dir / "final_metrics.csv").exists():
    raise FileExistsError(
        "final_metrics.csv already exists; the sealed holdout must not be rerun"
    )
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "models").mkdir(exist_ok=True)
timings: dict[str, float] = {}
assert_frozen_feature_contract(ALL_FEATURES)
checkpoint(
    "run_started",
    component_k=FROZEN_COMPONENT_K,
    top_k=FROZEN_TOP_K,
    features=len(FROZEN_FEATURES),
)

# Phase 1: freeze a final model using development labels only. The sealed
# order_products__train table is deliberately not loaded in this phase.
checkpoint("development_training_phase_started", final_labels_loaded=False)
development_data = timed(
    "development_load_data",
    load_instacart,
    args.data_dir,
    tables=("orders", "order_products__prior", "products"),
)
orders = development_data["orders"]
prior_items = development_data["order_products__prior"]
products = development_data["products"]
examples = build_evaluation_examples(orders)
development_history = timed(
    "development_history",
    history_items,
    orders,
    prior_items,
    examples,
    "ranker",
)
development_target = target_items(
    prior_items, pd.DataFrame(), examples, "ranker"
)
assert_no_target_in_history(development_history, examples, "ranker")
development_pool, development_edges = retrieve_frozen_pool(
    development_history,
    examples["user_id"].to_numpy(),
    "development",
)
development_feature_tables = timed(
    "development_build_feature_tables",
    build_feature_tables,
    development_history,
    orders,
    products,
    examples,
    stage="ranker",
)
development_labeled = timed(
    "development_attach_labels",
    attach_groups_and_labels,
    development_pool,
    examples,
    development_target,
    stage="ranker",
)
development_ranking = timed(
    "development_assemble_features",
    assemble_ranking_features,
    development_labeled,
    development_feature_tables,
)
assert_groups_contiguous(development_ranking)
assert_frozen_feature_contract(ALL_FEATURES)
if len(development_ranking) != 27_012_118:
    raise AssertionError("development candidate rows changed after pipeline freeze")
if development_ranking["query_id"].nunique() != 131_209:
    raise AssertionError("development group count changed after pipeline freeze")
development_stats = {
    "groups": int(development_ranking["query_id"].nunique()),
    "candidate_rows": int(len(development_ranking)),
    "positive_rows": int(development_ranking["label"].sum()),
    "positive_rate": float(development_ranking["label"].mean()),
    "memory_mib": frame_mib(development_ranking),
    "co_visitation_edges": development_edges,
}
checkpoint("development_training_data_ready", **development_stats)

categorical_features = ["aisle_id", "department_id"]
development_catboost_pool = timed(
    "development_catboost_pool",
    Pool,
    development_ranking[list(FROZEN_FEATURES)],
    label=development_ranking["label"],
    cat_features=categorical_features,
    feature_names=list(FROZEN_FEATURES),
)
classifier = CatBoostClassifier(
    **FROZEN_CLASSIFIER_PARAMS,
    eval_metric="Logloss",
    thread_count=args.thread_count,
    allow_writing_files=False,
    verbose=10,
)
checkpoint("final_classifier_fit_started", final_labels_loaded=False)
started = perf_counter()
classifier.fit(development_catboost_pool)
timings["final_classifier_fit"] = perf_counter() - started
model_path = args.output_dir / "models" / "classifier_frozen_full_development.cbm"
classifier.save_model(model_path)
if classifier.tree_count_ != FROZEN_CLASSIFIER_PARAMS["iterations"]:
    raise AssertionError("final classifier tree count differs from frozen protocol")
if tuple(classifier.feature_names_) != FROZEN_FEATURES:
    raise AssertionError("trained classifier feature contract drifted")
checkpoint(
    "final_classifier_fit_finished",
    step_seconds=round(timings["final_classifier_fit"], 3),
    trees=classifier.tree_count_,
    final_labels_loaded=False,
)

del development_data, orders, prior_items, products, examples
del development_history, development_target, development_pool
del development_feature_tables, development_labeled, development_ranking
del development_catboost_pool
gc.collect()

# Phase 2: first and only final-label opening. Train rows are used only by
# target_items(..., stage="final") and are deleted immediately afterwards.
checkpoint("final_labels_opening", model_already_frozen=True)
final_labels_opened_at = perf_counter() - STARTED
final_data = timed(
    "final_load_data",
    load_instacart,
    args.data_dir,
    tables=("orders", "order_products__prior", "products", "order_products__train"),
)
orders = final_data["orders"]
prior_items = final_data["order_products__prior"]
products = final_data["products"]
train_items = final_data["order_products__train"]
examples = build_evaluation_examples(orders)
assert_protocol_integrity(orders, prior_items, train_items, examples)
final_history = timed(
    "final_history",
    history_items,
    orders,
    prior_items,
    examples,
    "final",
)
final_target_raw = target_items(prior_items, train_items, examples, "final")
final_target = classify_repeat_explore(final_history, final_target_raw)
assert_no_target_in_history(final_history, examples, "final")
if not set(examples["ranker_target_order_id"]).issubset(
    set(final_history["order_id"].unique())
):
    raise AssertionError("order n-1 was not promoted into ordinary final history")
if set(final_target["order_id"]).intersection(final_history["order_id"]):
    raise AssertionError("final target rows entered final history")
final_label_rows = int(len(train_items))
del train_items, final_target_raw
gc.collect()
checkpoint(
    "final_labels_ready",
    target_rows=len(final_target),
    final_label_source_rows=final_label_rows,
)

final_pool, final_edges = retrieve_frozen_pool(
    final_history,
    examples["user_id"].to_numpy(),
    "final",
)
final_ceiling = candidate_recall_ceiling(final_pool, final_target)
final_feature_tables = timed(
    "final_build_feature_tables",
    build_feature_tables,
    final_history,
    orders,
    products,
    examples,
    stage="final",
)
final_labeled = timed(
    "final_attach_labels",
    attach_groups_and_labels,
    final_pool,
    examples,
    final_target,
    stage="final",
)
final_ranking = timed(
    "final_assemble_features",
    assemble_ranking_features,
    final_labeled,
    final_feature_tables,
)
assert_groups_contiguous(final_ranking)
assert_frozen_feature_contract(ALL_FEATURES)
final_stats = {
    "users": int(len(examples)),
    "history_interactions": int(len(final_history)),
    "target_rows": int(len(final_target)),
    "candidate_rows": int(len(final_ranking)),
    "mean_candidates_per_user": float(
        final_ranking.groupby("user_id", observed=True).size().mean()
    ),
    "positive_rows": int(final_ranking["label"].sum()),
    "positive_rate": float(final_ranking["label"].mean()),
    "ranking_memory_mib": frame_mib(final_ranking),
    "co_visitation_edges": final_edges,
}
checkpoint("final_ranking_data_ready", **final_stats)

final_catboost_pool = timed(
    "final_catboost_pool",
    Pool,
    final_ranking[list(FROZEN_FEATURES)],
    cat_features=categorical_features,
    feature_names=list(FROZEN_FEATURES),
)
checkpoint("final_predict_started")
started = perf_counter()
final_scores = classifier.predict_proba(final_catboost_pool)[:, 1]
timings["final_predict"] = perf_counter() - started
predictions = final_ranking[
    ["user_id", "target_order_id", "product_id"]
].copy()
predictions["model_score"] = np.asarray(final_scores, dtype=np.float32)
final_ranking_metrics = ranking_metrics(
    predictions,
    final_target,
    score_column="model_score",
    ks=(10, 20),
)
checkpoint(
    "final_predict_finished",
    step_seconds=round(timings["final_predict"], 3),
    ndcg_at_10=round(metric_value(final_ranking_metrics, "ndcg"), 6),
    recall_at_10=round(metric_value(final_ranking_metrics, "recall"), 6),
)

candidate_types = timed(
    "final_classify_candidate_novelty",
    classify_candidate_novelty,
    predictions,
    final_history,
)
top_ten = timed(
    "final_top_ten",
    apply_explore_quota,
    candidate_types,
    score_column="model_score",
    top_k=FROZEN_TOP_K,
    min_explore=0,
)
if len(top_ten) != len(examples) * FROZEN_TOP_K:
    raise AssertionError("final output does not contain exactly Top-10 per user")
if top_ten.duplicated(["user_id", "target_order_id", "product_id"]).any():
    raise AssertionError("duplicate products in final Top-10")

hit_labels = final_target[["user_id", "order_id", "product_id", "item_type"]].rename(
    columns={"order_id": "target_order_id", "item_type": "target_item_type"}
)
prediction_artifact = top_ten[
    [
        "user_id",
        "target_order_id",
        "product_id",
        "model_score",
        "policy_rank",
        "candidate_type",
    ]
].merge(
    hit_labels,
    on=["user_id", "target_order_id", "product_id"],
    how="left",
    validate="one_to_one",
)
prediction_artifact["hit"] = prediction_artifact["target_item_type"].notna().astype(
    "int8"
)
prediction_artifact.to_csv(
    args.output_dir / "final_predictions_top10.csv.gz",
    index=False,
    compression="gzip",
)

final_metric_artifact = combined_final_metrics(final_ceiling, final_ranking_metrics)
final_metric_artifact.to_csv(args.output_dir / "final_metrics.csv", index=False)
final_ceiling.to_csv(args.output_dir / "candidate_ceiling.csv", index=False)

final_values = {
    "candidate_recall": float(
        final_ceiling.loc[final_ceiling["segment"].eq("overall"), "recall"].item()
    ),
    "ndcg_at_10": metric_value(final_ranking_metrics, "ndcg"),
    "recall_at_10": metric_value(final_ranking_metrics, "recall"),
    "repeat_recall_at_10": metric_value(
        final_ranking_metrics, "recall", "repeat"
    ),
    "explore_recall_at_10": metric_value(
        final_ranking_metrics, "recall", "explore"
    ),
}
generalization = pd.DataFrame(
    [
        {
            "metric": metric,
            "development": development_value,
            "final": final_values[metric],
            "delta": final_values[metric] - development_value,
        }
        for metric, development_value in DEVELOPMENT_METRICS.items()
    ]
)
generalization.to_csv(args.output_dir / "development_vs_final.csv", index=False)

# Strictly post-hoc analysis of already produced predictions.
user_segments = (
    final_ranking[["user_id", "u_reorder_ratio"]]
    .drop_duplicates("user_id")
    .sort_values("user_id", ignore_index=True)
)
user_segments["segment"] = reorder_ratio_segment(user_segments["u_reorder_ratio"])
basket_segments = (
    final_target.groupby(["user_id", "order_id"], observed=True)
    .size()
    .rename("target_basket_size")
    .reset_index()
)
basket_segments["segment"] = target_basket_segment(
    basket_segments["target_basket_size"]
)
segment_rows = []
for segment in user_segments["segment"].cat.categories:
    segment_rows.append(
        segment_result(
            "u_reorder_ratio",
            str(segment),
            user_segments.loc[user_segments["segment"].eq(segment), "user_id"],
            final_pool,
            top_ten,
            final_target,
        )
    )
for segment in basket_segments["segment"].cat.categories:
    segment_rows.append(
        segment_result(
            "target_basket_size",
            str(segment),
            basket_segments.loc[basket_segments["segment"].eq(segment), "user_id"],
            final_pool,
            top_ten,
            final_target,
        )
    )
segment_summary = pd.DataFrame(segment_rows)
segment_summary.to_csv(args.output_dir / "final_segment_metrics.csv", index=False)

diagnostics = {
    "run": {
        "wall_seconds": perf_counter() - STARTED,
        "peak_rss_mib": peak_rss_mib(),
        "final_labels_opened_elapsed_seconds": final_labels_opened_at,
        "development_model_saved_before_final_labels": True,
        "final_evaluation_count": 1,
    },
    "frozen_configuration": {
        "component_k_per_source": FROZEN_COMPONENT_K,
        "max_neighbors": FROZEN_MAX_NEIGHBORS,
        "top_k": FROZEN_TOP_K,
        "candidate_sources": ["personal_repeat", "co_visitation_explore"],
        "ranking_policy": "unconstrained_classifier_score",
        "features": list(FROZEN_FEATURES),
        "classifier_parameters": {
            **FROZEN_CLASSIFIER_PARAMS,
            "thread_count": args.thread_count,
        },
    },
    "hardware": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    },
    "development_training": development_stats,
    "final_data": final_stats,
    "leakage_checks": {
        "final_label_source": "order_products__train",
        "final_label_source_rows": final_label_rows,
        "final_history_source": "order_products__prior only",
        "final_history_max_order": "n-1",
        "target_order_in_history": False,
        "ranker_target_n_minus_1_in_final_history": True,
        "final_labels_used_for_training": False,
    },
    "timings_seconds": timings,
}
(args.output_dir / "final_diagnostics.json").write_text(
    json.dumps(diagnostics, indent=2, sort_keys=True)
)
checkpoint(
    "run_finished",
    final_metrics_rows=len(final_metric_artifact),
    segment_rows=len(segment_summary),
)
print(final_metric_artifact.to_string(index=False), flush=True)
print(generalization.to_string(index=False), flush=True)
print(segment_summary.to_string(index=False), flush=True)
