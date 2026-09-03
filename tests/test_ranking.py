import numpy as np
import pandas as pd
import pytest

from src.ranking import (
    apply_explore_quota,
    assert_groups_contiguous,
    attach_groups_and_labels,
    build_candidate_pool,
    candidate_recall_ceiling,
    classify_candidate_novelty,
    heuristic_scores,
    ranking_metrics,
    split_group_users,
)


def test_candidate_novelty_uses_visible_history_only():
    candidates = pd.DataFrame(
        [(1, 101, 10), (1, 101, 20), (2, 202, 10)],
        columns=["user_id", "target_order_id", "product_id"],
    )
    visible_history = pd.DataFrame(
        [(1, 1, 10), (2, 2, 99)],
        columns=["user_id", "order_id", "product_id"],
    )

    classified = classify_candidate_novelty(candidates, visible_history)

    assert classified["candidate_type"].astype("string").tolist() == [
        "repeat",
        "explore",
        "explore",
    ]


def test_explore_quota_keeps_top_k_unique_and_preserves_model_order():
    candidates = pd.DataFrame(
        [
            (1, 101, 10, 0.9, "repeat"),
            (1, 101, 11, 0.8, "repeat"),
            (1, 101, 12, 0.7, "repeat"),
            (1, 101, 13, 0.6, "repeat"),
            (1, 101, 20, 0.5, "explore"),
            (1, 101, 21, 0.4, "explore"),
        ],
        columns=[
            "user_id",
            "target_order_id",
            "product_id",
            "model_score",
            "candidate_type",
        ],
    )

    selected = apply_explore_quota(
        candidates, score_column="model_score", top_k=4, min_explore=2
    )

    assert selected["product_id"].tolist() == [10, 11, 20, 21]
    assert selected["candidate_type"].eq("explore").sum() == 2
    assert selected["policy_rank"].tolist() == [1, 2, 3, 4]
    assert not selected.duplicated(
        ["user_id", "target_order_id", "product_id"]
    ).any()
    assert selected["model_score"].is_monotonic_decreasing


def test_explore_quota_degrades_when_explore_candidates_are_insufficient():
    candidates = pd.DataFrame(
        [
            (2, 202, 30, 0.9, "repeat"),
            (2, 202, 31, 0.8, "repeat"),
            (2, 202, 40, 0.7, "explore"),
            (2, 202, 32, 0.6, "repeat"),
            (2, 202, 33, 0.5, "repeat"),
        ],
        columns=[
            "user_id",
            "target_order_id",
            "product_id",
            "model_score",
            "candidate_type",
        ],
    )

    selected = apply_explore_quota(
        candidates, score_column="model_score", top_k=4, min_explore=3
    )

    assert selected["product_id"].tolist() == [30, 31, 40, 32]
    assert len(selected) == 4
    assert selected["candidate_type"].eq("explore").sum() == 1


def test_candidate_pool_preserves_scores_sources_and_deduplicates():
    repeat = pd.DataFrame(
        [(1, 10, 3.0, 1), (1, 11, 2.0, 2)],
        columns=["user_id", "product_id", "score", "rank"],
    )
    covis = pd.DataFrame(
        [(1, 10, 0.7, 1), (1, 20, 0.4, 2)],
        columns=["user_id", "product_id", "score", "rank"],
    )
    pool = build_candidate_pool(repeat, covis, component_k=2)

    assert pool["product_id"].tolist() == [10, 11, 20]
    shared = pool.loc[pool["product_id"].eq(10)].iloc[0]
    assert shared["repeat_score"] == 3.0
    assert np.isclose(shared["covis_score"], 0.7)
    assert shared["candidate_source"] == "repeat+covis"
    repeat_only = pool.loc[pool["product_id"].eq(11)].iloc[0]
    covis_only = pool.loc[pool["product_id"].eq(20)].iloc[0]
    assert pd.isna(repeat_only["covis_score"])
    assert pd.isna(covis_only["repeat_score"])


def test_heuristic_uses_two_repeat_then_one_covis_schedule():
    candidates = pd.DataFrame(
        {
            "repeat_rank": [1, 2, 3, float("nan")],
            "covis_rank": [float("nan"), float("nan"), float("nan"), 1],
        }
    )
    assert heuristic_scores(candidates).tolist() == [-1.0, -2.0, -4.0, -3.0]


def test_candidate_labels_and_group_boundary_are_correct():
    pool = pd.DataFrame(
        [(1, 10), (1, 20), (2, 30)], columns=["user_id", "product_id"]
    )
    examples = pd.DataFrame(
        [(1, 101), (2, 202)], columns=["user_id", "ranker_target_order_id"]
    )
    target = pd.DataFrame(
        [(1, 101, 20), (2, 202, 99)],
        columns=["user_id", "order_id", "product_id"],
    )
    labeled = attach_groups_and_labels(pool, examples, target)

    actual = dict(zip(labeled["product_id"], labeled["label"]))
    assert actual == {10: 0, 20: 1, 30: 0}
    assert labeled.loc[labeled["user_id"].eq(1), "query_id"].unique().tolist() == [101]
    assert_groups_contiguous(labeled)


def test_group_user_split_is_deterministic_and_disjoint():
    train_a, validation_a = split_group_users(range(100), seed=42)
    train_b, validation_b = split_group_users(range(100), seed=42)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(validation_a, validation_b)
    assert len(validation_a) == 20
    assert set(train_a).isdisjoint(validation_a)


def test_noncontiguous_catboost_groups_are_rejected():
    good = pd.DataFrame({"query_id": [1, 1, 2, 2]})
    bad = pd.DataFrame({"query_id": [1, 2, 1]})
    assert_groups_contiguous(good)
    with pytest.raises(AssertionError):
        assert_groups_contiguous(bad)


def test_candidate_ceiling_uses_full_target_denominator():
    pool = pd.DataFrame([(1, 10), (2, 20)], columns=["user_id", "product_id"])
    target = pd.DataFrame(
        [(1, 10, "repeat"), (1, 11, "explore"), (2, 20, "explore")],
        columns=["user_id", "product_id", "item_type"],
    )
    metrics = candidate_recall_ceiling(pool, target).set_index("segment")

    assert metrics.loc["overall", "recall"] == 0.75
    assert metrics.loc["repeat", "recall"] == 1.0
    assert metrics.loc["explore", "recall"] == 0.5


def test_ndcg_and_recall_match_manual_example_with_all_negative_group():
    predictions = pd.DataFrame(
        [
            (1, 101, 10, 0.9),
            (1, 101, 99, 0.8),
            (1, 101, 11, 0.7),
            (2, 202, 50, 0.9),
        ],
        columns=["user_id", "target_order_id", "product_id", "score"],
    )
    target = pd.DataFrame(
        [(1, 101, 10, "repeat"), (1, 101, 11, "explore"), (2, 202, 20, "explore")],
        columns=["user_id", "order_id", "product_id", "item_type"],
    )
    metrics = ranking_metrics(predictions, target, score_column="score", ks=(2, 3))

    def value(k, metric, segment="overall"):
        return metrics.loc[
            metrics["k"].eq(k)
            & metrics["metric"].eq(metric)
            & metrics["segment"].eq(segment),
            "value",
        ].item()

    expected_user_1_ndcg_at_2 = 1 / (1 + 1 / np.log2(3))
    assert np.isclose(value(2, "ndcg"), expected_user_1_ndcg_at_2 / 2)
    assert np.isclose(value(2, "recall"), 0.25)
    assert np.isclose(value(3, "recall"), 0.5)
    assert np.isclose(value(3, "recall", "repeat"), 1.0)
    assert np.isclose(value(3, "recall", "explore"), 0.5)
