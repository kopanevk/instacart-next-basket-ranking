import numpy as np
import pandas as pd

from src.evaluation import retrieval_recall
from src.retrieval import (
    build_co_visitation,
    co_visitation_candidates,
    global_popularity,
    personal_repeat_candidates,
    popularity_candidates,
    union_candidates,
)


def _history():
    return pd.DataFrame(
        [
            (1, 10, 1, 1),
            (1, 10, 1, 2),
            (2, 10, 2, 1),
            (2, 10, 2, 2),
            (3, 20, 1, 1),
            (3, 20, 1, 3),
            (4, 30, 1, 4),
            (4, 30, 1, 5),
        ],
        columns=["order_id", "user_id", "order_number", "product_id"],
    )


def test_popularity_uses_only_passed_allowed_history():
    history = _history()
    target_basket = pd.DataFrame(
        [(99, 10, 3, 999)] ,
        columns=history.columns,
    )
    popularity = global_popularity(history)

    assert popularity.iloc[0]["product_id"] == 1
    assert 999 not in set(popularity["product_id"])
    assert 999 in set(global_popularity(pd.concat([history, target_basket]))["product_id"])


def test_personal_repeat_never_invents_unseen_items():
    history = _history()
    candidates = personal_repeat_candidates(history, k=10)
    seen = history[["user_id", "product_id"]].drop_duplicates()

    assert len(candidates.merge(seen, on=["user_id", "product_id"])) == len(candidates)
    user_10 = candidates.loc[candidates["user_id"].eq(10)]
    assert user_10["product_id"].tolist() == [1, 2]
    assert user_10["score"].tolist() == [2, 2]


def test_explore_popularity_excludes_seen_items():
    history = _history()
    popularity = global_popularity(history)
    candidates = popularity_candidates(
        [10, 20], popularity, k=3, history=history, exclude_seen=True
    )
    seen = history[["user_id", "product_id"]].drop_duplicates()

    assert candidates.merge(seen, on=["user_id", "product_id"]).empty
    assert candidates.loc[candidates["user_id"].eq(10), "product_id"].tolist() == [3, 4, 5]


def test_co_visitation_cosine_and_target_exclusion():
    history = _history()
    target_only = pd.DataFrame(
        [(99, 10, 3, 999), (99, 10, 3, 2)], columns=history.columns
    )
    index = build_co_visitation(history, max_neighbors=10)

    assert 999 not in set(index.item_ids)
    assert 999 in set(build_co_visitation(pd.concat([history, target_only])).item_ids)
    positions = {item: pos for pos, item in enumerate(index.item_ids)}
    actual = index.similarity[positions[1], positions[2]]
    assert np.isclose(actual, 2 / np.sqrt(3 * 2))


def test_co_visitation_explore_candidates_exclude_history():
    history = _history()
    index = build_co_visitation(history, max_neighbors=10)
    candidates = co_visitation_candidates(history, [10, 20], index, k=10)
    seen = history[["user_id", "product_id"]].drop_duplicates()

    assert candidates.merge(seen, on=["user_id", "product_id"]).empty
    assert candidates.loc[candidates["user_id"].eq(10), "product_id"].tolist() == [3]
    assert candidates.loc[candidates["user_id"].eq(20), "product_id"].tolist() == [2]


def test_candidate_union_deduplicates_and_preserves_sources():
    first = pd.DataFrame(
        [(1, 10, 1, "repeat"), (1, 20, 2, "repeat")],
        columns=["user_id", "product_id", "rank", "source"],
    )
    second = pd.DataFrame(
        [(1, 10, 1, "explore"), (1, 30, 2, "explore")],
        columns=["user_id", "product_id", "rank", "source"],
    )
    combined = union_candidates(
        [first, second], k=3, source_priority=["repeat", "explore"]
    )

    assert combined["product_id"].tolist() == [10, 20, 30]
    assert combined["product_id"].is_unique
    assert combined.loc[combined["product_id"].eq(10), "source"].item() == "repeat|explore"


def test_recall_at_k_uses_full_segment_denominators():
    target = pd.DataFrame(
        [
            (1, 10, "repeat"),
            (1, 11, "repeat"),
            (1, 12, "explore"),
            (2, 20, "explore"),
            (2, 21, "explore"),
        ],
        columns=["user_id", "product_id", "item_type"],
    )
    candidates = pd.DataFrame(
        [(1, 10, 1), (1, 12, 2), (2, 20, 1), (2, 99, 2)],
        columns=["user_id", "product_id", "rank"],
    )
    metrics = retrieval_recall(candidates, target, ks=(1, 2))

    def value(k, segment, column="recall"):
        return metrics.loc[
            metrics["k"].eq(k) & metrics["segment"].eq(segment), column
        ].item()

    assert np.isclose(value(1, "overall"), (1 / 3 + 1 / 2) / 2)
    assert np.isclose(value(1, "repeat"), 1 / 2)
    assert np.isclose(value(1, "explore"), (0 / 1 + 1 / 2) / 2)
    assert np.isclose(value(2, "overall"), (2 / 3 + 1 / 2) / 2)
    assert np.isclose(value(2, "explore"), (1 / 1 + 1 / 2) / 2)
    assert value(1, "repeat", "users") == 1
    assert value(1, "explore", "users") == 2


def test_empty_segment_denominator_is_excluded_not_zero():
    target = pd.DataFrame(
        [(1, 10, "repeat"), (2, 20, "repeat")],
        columns=["user_id", "product_id", "item_type"],
    )
    candidates = pd.DataFrame(
        [(1, 10, 1)], columns=["user_id", "product_id", "rank"]
    )
    metrics = retrieval_recall(candidates, target, ks=(1,))

    explore = metrics.loc[metrics["segment"].eq("explore")].iloc[0]
    repeat = metrics.loc[metrics["segment"].eq("repeat")].iloc[0]
    assert np.isnan(explore["recall"])
    assert explore["users"] == 0
    assert repeat["recall"] == 0.5

