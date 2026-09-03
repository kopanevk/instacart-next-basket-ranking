"""Run the full leakage-safe heuristic/classifier/ranker experiment."""

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
from src.features import (  # noqa: E402
    ALL_FEATURES,
    BASE_FEATURES,
    CLOCK_FEATURES,
    INTERVAL_FEATURES,
    assemble_ranking_features,
    build_feature_tables,
)
from src.ranking import (  # noqa: E402
    assert_groups_contiguous,
    attach_groups_and_labels,
    build_candidate_pool,
    candidate_recall_ceiling,
    heuristic_scores,
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
    parser.add_argument("--sample-users", type=int, default=0)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classifier-iterations", type=int, default=120)
    parser.add_argument("--ranker-iterations", type=int, default=30)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--thread-count", type=int, default=10)
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


def timed(name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    checkpoint(f"{name}_started")
    started = perf_counter()
    result = function(*args, **kwargs)
    timings[name] = perf_counter() - started
    checkpoint(f"{name}_finished", step_seconds=round(timings[name], 3))
    return result


def make_pool(feature_names: list[str], row_mask: np.ndarray) -> Pool:
    selected = ranking_data.loc[
        row_mask, ["query_id", "label", *feature_names]
    ]
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
    frame = ranking_data.loc[
        validation_mask,
        ["user_id", "target_order_id", "product_id"],
    ].copy()
    frame["model_score"] = np.asarray(scores, dtype=np.float32)
    return frame


def evaluate_method(name: str, scores: np.ndarray) -> pd.DataFrame:
    predictions = prediction_frame(scores)
    metrics = ranking_metrics(
        predictions, validation_target, score_column="model_score", ks=(10, 20)
    )
    metrics.insert(0, "method", name)
    metric_frames.append(metrics)
    checkpoint(
        "method_evaluated",
        method=name,
        ndcg_at_10=round(metric_value(metrics, 10, "ndcg", "overall"), 6),
        recall_at_10=round(metric_value(metrics, 10, "recall", "overall"), 6),
    )
    del predictions
    gc.collect()
    return metrics


def metric_value(
    metrics: pd.DataFrame, k: int, metric: str, segment: str
) -> float:
    return float(
        metrics.loc[
            metrics["k"].eq(k)
            & metrics["metric"].eq(metric)
            & metrics["segment"].eq(segment),
            "value",
        ].item()
    )


def ranker_parameters() -> dict[str, Any]:
    return {
        "iterations": args.ranker_iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=10",
        "random_seed": args.seed,
        "thread_count": args.thread_count,
        "od_type": "Iter",
        "od_wait": 15,
        "allow_writing_files": False,
        "verbose": 10,
    }


def train_ranker(label: str, feature_names: list[str]) -> tuple[CatBoostRanker, pd.DataFrame]:
    train_pool = timed(f"{label}_train_pool", make_pool, feature_names, train_mask)
    validation_pool = timed(
        f"{label}_validation_pool", make_pool, feature_names, validation_mask
    )
    model = CatBoostRanker(**ranker_parameters())
    checkpoint(f"{label}_fit_started", features=len(feature_names))
    started = perf_counter()
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
    timings[f"{label}_fit"] = perf_counter() - started
    checkpoint(
        f"{label}_fit_finished",
        step_seconds=round(timings[f"{label}_fit"], 3),
        best_iteration=model.get_best_iteration(),
    )
    scores = model.predict(validation_pool)
    metrics = evaluate_method(label, scores)
    del train_pool, validation_pool, scores
    gc.collect()
    return model, metrics


STARTED = perf_counter()
args = parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "models").mkdir(exist_ok=True)
timings: dict[str, float] = {}
metric_frames: list[pd.DataFrame] = []
context_rows: list[dict[str, Any]] = []

checkpoint("run_started", component_k=args.component_k, seed=args.seed)
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
all_evaluation_users = len(examples)
if args.sample_users:
    if not 1 < args.sample_users <= all_evaluation_users:
        raise ValueError("sample-users must be between 2 and the evaluation population")
    sample_rng = np.random.default_rng(args.seed)
    sampled_users = sample_rng.choice(
        examples["user_id"].to_numpy(), args.sample_users, replace=False
    )
    examples = examples.loc[examples["user_id"].isin(sampled_users)].reset_index(
        drop=True
    )
history = timed(
    "materialize_history", history_items, orders, prior_items, examples, "ranker"
)
target_raw = target_items(prior_items, pd.DataFrame(), examples, "ranker")
target = classify_repeat_explore(history, target_raw)
assert_no_target_in_history(history, examples, "ranker")
checkpoint("history_ready", users=len(examples), interactions=len(history))

personal = timed(
    "retrieve_personal", personal_repeat_candidates, history, args.component_k
)
index = timed(
    "fit_co_visitation",
    build_co_visitation,
    history,
    max_neighbors=args.max_neighbors,
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
ceiling = candidate_recall_ceiling(candidate_pool, target)
checkpoint(
    "candidate_pool_ready",
    rows=len(candidate_pool),
    memory_mib=round(frame_mib(candidate_pool), 1),
    overall_recall=round(
        ceiling.loc[ceiling["segment"].eq("overall"), "recall"].item(), 6
    ),
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
checkpoint(
    "ranking_data_ready",
    rows=len(ranking_data),
    groups=ranking_data["query_id"].nunique(),
    positives=int(ranking_data["label"].sum()),
    positive_rate=round(float(ranking_data["label"].mean()), 6),
    memory_mib=round(frame_mib(ranking_data), 1),
)
del data, orders, prior_items, products, history, target_raw
del personal, covis, index, candidate_pool, labeled_candidates, feature_tables
gc.collect()

train_users, validation_users = split_group_users(
    examples["user_id"],
    validation_fraction=args.validation_fraction,
    seed=args.seed,
)
train_mask = ranking_data["user_id"].isin(train_users).to_numpy()
validation_mask = ranking_data["user_id"].isin(validation_users).to_numpy()
assert not (train_mask & validation_mask).any()
validation_target = target.loc[target["user_id"].isin(validation_users)].copy()
validation_candidates = ranking_data.loc[
    validation_mask, ["user_id", "product_id"]
]
validation_ceiling = candidate_recall_ceiling(validation_candidates, validation_target)
group_positive_counts = ranking_data.groupby("query_id", observed=True)["label"].sum()
dataset_stats = {
    "groups": int(ranking_data["query_id"].nunique()),
    "candidate_rows": int(len(ranking_data)),
    "positive_rows": int(ranking_data["label"].sum()),
    "positive_rate": float(ranking_data["label"].mean()),
    "mean_candidates_per_group": float(
        ranking_data.groupby("query_id", observed=True).size().mean()
    ),
    "train_users": int(len(train_users)),
    "validation_users": int(len(validation_users)),
    "train_rows": int(train_mask.sum()),
    "validation_rows": int(validation_mask.sum()),
    "all_negative_groups": int(group_positive_counts.eq(0).sum()),
    "memory_mib": frame_mib(ranking_data),
}
checkpoint("split_ready", **dataset_stats)
del validation_candidates, group_positive_counts
gc.collect()

heuristic = heuristic_scores(
    ranking_data.loc[validation_mask, ["repeat_rank", "covis_rank"]]
)
heuristic_metrics = evaluate_method("Heuristic 2:1", heuristic)
del heuristic

all_train_pool = timed("classifier_train_pool", make_pool, ALL_FEATURES, train_mask)
all_validation_pool = timed(
    "classifier_validation_pool", make_pool, ALL_FEATURES, validation_mask
)
classifier = CatBoostClassifier(
    iterations=args.classifier_iterations,
    depth=args.depth,
    learning_rate=args.learning_rate,
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=args.seed,
    thread_count=args.thread_count,
    od_type="Iter",
    od_wait=15,
    allow_writing_files=False,
    verbose=10,
)
checkpoint("classifier_fit_started", features=len(ALL_FEATURES))
started = perf_counter()
classifier.fit(all_train_pool, eval_set=all_validation_pool, use_best_model=True)
timings["classifier_fit"] = perf_counter() - started
checkpoint(
    "classifier_fit_finished",
    step_seconds=round(timings["classifier_fit"], 3),
    best_iteration=classifier.get_best_iteration(),
)
classifier_scores = classifier.predict_proba(all_validation_pool)[:, 1]
classifier_metrics = evaluate_method("CatBoostClassifier", classifier_scores)
classifier.save_model(args.output_dir / "models" / "classifier.cbm")
del classifier_scores, classifier, all_train_pool, all_validation_pool
gc.collect()

main_ranker, main_ranker_metrics = train_ranker("CatBoostRanker ALL", ALL_FEATURES)
importance = pd.DataFrame(
    {
        "feature": ALL_FEATURES,
        "importance": main_ranker.get_feature_importance(
            type="PredictionValuesChange"
        ),
    }
).sort_values("importance", ascending=False, ignore_index=True)
main_ranker.save_model(args.output_dir / "models" / "ranker_all_context.cbm")
context_rows.append(
    {
        "variant": "BASE + ALL CONTEXT",
        "features": len(ALL_FEATURES),
        "best_iteration": main_ranker.get_best_iteration(),
        "ndcg_at_10": metric_value(main_ranker_metrics, 10, "ndcg", "overall"),
        "repeat_recall_at_10": metric_value(main_ranker_metrics, 10, "recall", "repeat"),
        "explore_recall_at_10": metric_value(main_ranker_metrics, 10, "recall", "explore"),
    }
)
del main_ranker
gc.collect()

context_feature_sets = {
    "BASE": BASE_FEATURES,
    "BASE + INTERVAL": BASE_FEATURES + INTERVAL_FEATURES,
    "BASE + CLOCK": BASE_FEATURES + CLOCK_FEATURES,
}
for variant, feature_names in context_feature_sets.items():
    model, metrics = train_ranker(f"CatBoostRanker {variant}", feature_names)
    context_rows.append(
        {
            "variant": variant,
            "features": len(feature_names),
            "best_iteration": model.get_best_iteration(),
            "ndcg_at_10": metric_value(metrics, 10, "ndcg", "overall"),
            "repeat_recall_at_10": metric_value(metrics, 10, "recall", "repeat"),
            "explore_recall_at_10": metric_value(metrics, 10, "recall", "explore"),
        }
    )
    model.save_model(
        args.output_dir / "models" / f"ranker_{variant.lower().replace(' + ', '_').replace(' ', '_')}.cbm"
    )
    del model
    gc.collect()

metrics = pd.concat(metric_frames, ignore_index=True)
context = pd.DataFrame(context_rows)
metrics.to_csv(args.output_dir / "ranking_metrics.csv", index=False)
context.to_csv(args.output_dir / "context_ablation.csv", index=False)
importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
ceiling.assign(scope="all_users").to_csv(
    args.output_dir / "candidate_ceiling_full.csv", index=False
)
validation_ceiling.assign(scope="validation_users").to_csv(
    args.output_dir / "candidate_ceiling_validation.csv", index=False
)

diagnostics = {
    "run": {
        "wall_seconds": perf_counter() - STARTED,
        "peak_rss_mib": peak_rss_mib(),
        "component_k": args.component_k,
        "max_neighbors": args.max_neighbors,
        "classifier_iterations": args.classifier_iterations,
        "ranker_iterations": args.ranker_iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "sample_users": args.sample_users,
        "all_evaluation_users": all_evaluation_users,
    },
    "hardware": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    },
    "dataset": dataset_stats,
    "timings_seconds": timings,
}
(args.output_dir / "ranking_diagnostics.json").write_text(
    json.dumps(diagnostics, indent=2, sort_keys=True)
)
checkpoint("run_finished", metrics_rows=len(metrics))
print(metrics.to_string(index=False), flush=True)
print(context.to_string(index=False), flush=True)
print(importance.head(16).to_string(index=False), flush=True)
