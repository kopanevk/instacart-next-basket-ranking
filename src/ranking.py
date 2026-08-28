"""Candidate-pool assembly, group splitting, and ranking evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.evaluation import Stage


GROUP_COLUMNS = ["user_id", "target_order_id"]


def classify_candidate_novelty(
    candidates: pd.DataFrame,
    visible_history: pd.DataFrame,
) -> pd.DataFrame:
    """Mark candidates repeat/explore using visible history and nothing later."""
    _require_columns(candidates, {"user_id", "product_id"}, "candidates")
    _require_columns(visible_history, {"user_id", "product_id"}, "visible_history")
    if candidates.duplicated([*GROUP_COLUMNS, "product_id"]).any():
        raise ValueError("candidate products must be unique within a ranking group")

    seen = (
        visible_history[["user_id", "product_id"]]
        .drop_duplicates()
        .assign(_seen=np.int8(1))
    )
    result = candidates.merge(
        seen,
        on=["user_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    result["candidate_type"] = pd.Categorical(
        np.where(result["_seen"].eq(1), "repeat", "explore"),
        categories=["repeat", "explore"],
    )
    return result.drop(columns="_seen")


def apply_explore_quota(
    candidates: pd.DataFrame,
    *,
    score_column: str,
    top_k: int = 10,
    min_explore: int = 0,
) -> pd.DataFrame:
    """Select Top-K by model score while reserving up to N explore positions.

    When the unconstrained Top-K misses the quota, the best-scoring explore
    candidates below the cutoff replace the weakest repeat candidates. The
    surviving candidates retain their relative model-score order.
    """
    required = {*GROUP_COLUMNS, "product_id", "candidate_type", score_column}
    _require_columns(candidates, required, "candidates")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 <= min_explore <= top_k:
        raise ValueError("min_explore must be between zero and top_k")
    if candidates.duplicated([*GROUP_COLUMNS, "product_id"]).any():
        raise ValueError("candidate products must be unique within a ranking group")
    observed_types = set(candidates["candidate_type"].astype("string").dropna())
    if not observed_types.issubset({"repeat", "explore"}):
        raise ValueError("candidate_type must contain only repeat/explore")

    ranked = candidates.sort_values(
        [*GROUP_COLUMNS, score_column, "product_id"],
        ascending=[True, True, False, True],
    ).copy()
    group = ranked.groupby(GROUP_COLUMNS, observed=True, sort=False)
    ranked["model_rank"] = group.cumcount().add(1).astype("int32")
    ranked["_is_explore"] = ranked["candidate_type"].astype("string").eq(
        "explore"
    )
    ranked["_in_base_topk"] = ranked["model_rank"].le(top_k)

    explore_rank = (
        ranked["_is_explore"]
        .astype("int32")
        .groupby([ranked[column] for column in GROUP_COLUMNS], observed=True)
        .cumsum()
    )
    ranked["_explore_rank"] = explore_rank
    group_keys = [ranked[column] for column in GROUP_COLUMNS]
    available_explore = ranked["_is_explore"].groupby(group_keys, observed=True).transform(
        "sum"
    )
    current_explore = (
        (ranked["_is_explore"] & ranked["_in_base_topk"])
        .groupby(group_keys, observed=True)
        .transform("sum")
    )
    effective_quota = np.minimum(available_explore, min_explore)
    promotions = np.maximum(effective_quota - current_explore, 0).astype("int32")

    promote = (
        ranked["_is_explore"]
        & ~ranked["_in_base_topk"]
        & ranked["_explore_rank"].le(effective_quota)
    )
    top_repeat = ranked["_in_base_topk"] & ~ranked["_is_explore"]
    weak_repeat_rank = pd.Series(0, index=ranked.index, dtype="int32")
    weak_repeat_rank.loc[top_repeat] = (
        ranked.loc[top_repeat]
        .sort_values(
            [*GROUP_COLUMNS, "model_rank"],
            ascending=[True, True, False],
        )
        .groupby(GROUP_COLUMNS, observed=True, sort=False)
        .cumcount()
        .add(1)
        .astype("int32")
        .sort_index()
    )
    remove = top_repeat & weak_repeat_rank.le(promotions) & promotions.gt(0)
    selected = (ranked["_in_base_topk"] & ~remove) | promote
    result = ranked.loc[selected].sort_values(
        [*GROUP_COLUMNS, "model_rank"], ignore_index=True
    )
    result["policy_rank"] = (
        result.groupby(GROUP_COLUMNS, observed=True, sort=False)
        .cumcount()
        .add(1)
        .astype("int16")
    )
    result["policy_score"] = -result["policy_rank"].astype("float32")
    result["explore_quota"] = np.int8(min_explore)
    return result.drop(
        columns=[
            "_is_explore",
            "_in_base_topk",
            "_explore_rank",
        ]
    )


def build_candidate_pool(
    personal_candidates: pd.DataFrame,
    covis_candidates: pd.DataFrame,
    component_k: int,
) -> pd.DataFrame:
    """Union repeat and co-visitation candidates without combining their scores."""
    if component_k <= 0:
        raise ValueError("component_k must be positive")
    required = {"user_id", "product_id", "score", "rank"}
    for name, frame in (
        ("personal_candidates", personal_candidates),
        ("covis_candidates", covis_candidates),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    repeat = personal_candidates.loc[
        personal_candidates["rank"] <= component_k,
        ["user_id", "product_id", "score", "rank"],
    ].rename(columns={"score": "repeat_score", "rank": "repeat_rank"})
    repeat["covis_score"] = np.float32(np.nan)
    repeat["covis_rank"] = np.float32(np.nan)

    covis = covis_candidates.loc[
        covis_candidates["rank"] <= component_k,
        ["user_id", "product_id", "score", "rank"],
    ].rename(columns={"score": "covis_score", "rank": "covis_rank"})
    covis["repeat_score"] = np.float32(np.nan)
    covis["repeat_rank"] = np.float32(np.nan)

    columns = [
        "user_id",
        "product_id",
        "repeat_score",
        "covis_score",
        "repeat_rank",
        "covis_rank",
    ]
    pool = pd.concat([repeat[columns], covis[columns]], ignore_index=True)
    duplicate_mask = pool.duplicated(["user_id", "product_id"], keep=False)
    if duplicate_mask.any():
        duplicates = (
            pool.loc[duplicate_mask]
            .groupby(["user_id", "product_id"], observed=True, as_index=False)
            .agg(
                repeat_score=("repeat_score", "max"),
                covis_score=("covis_score", "max"),
                repeat_rank=("repeat_rank", "min"),
                covis_rank=("covis_rank", "min"),
            )
        )
        pool = pd.concat([pool.loc[~duplicate_mask], duplicates], ignore_index=True)

    has_repeat = pool["repeat_score"].notna()
    has_covis = pool["covis_score"].notna()
    pool["candidate_source"] = pd.Categorical(
        np.select(
            [has_repeat & has_covis, has_repeat],
            ["repeat+covis", "repeat"],
            default="covis",
        ),
        categories=["repeat", "covis", "repeat+covis"],
    )
    for column in ("repeat_score", "covis_score", "repeat_rank", "covis_rank"):
        pool[column] = pool[column].astype("float32")
    assert not pool.duplicated(["user_id", "product_id"]).any()
    return pool.sort_values(["user_id", "product_id"], ignore_index=True)


def attach_groups_and_labels(
    pool: pd.DataFrame,
    examples: pd.DataFrame,
    target: pd.DataFrame,
    *,
    stage: Stage = "ranker",
) -> pd.DataFrame:
    """Create one labeled row per (user, prediction moment, candidate item)."""
    target_id_column = f"{stage}_target_order_id"
    _require_columns(pool, {"user_id", "product_id"}, "pool")
    _require_columns(
        examples, {"user_id", target_id_column}, "examples"
    )
    _require_columns(target, {"user_id", "order_id", "product_id"}, "target")
    groups = examples[["user_id", target_id_column]].rename(
        columns={target_id_column: "target_order_id"}
    )
    if groups.duplicated(GROUP_COLUMNS).any():
        raise ValueError("ranking groups must be unique")
    result = pool.merge(groups, on="user_id", how="inner", validate="many_to_one")
    labels = target[["user_id", "order_id", "product_id"]].rename(
        columns={"order_id": "target_order_id"}
    )
    labels = labels.drop_duplicates().assign(label=np.int8(1))
    result = result.merge(
        labels,
        on=[*GROUP_COLUMNS, "product_id"],
        how="left",
        validate="one_to_one",
    )
    result["label"] = result["label"].fillna(0).astype("int8")
    result["query_id"] = result["target_order_id"].astype("int32")
    return result.sort_values(
        ["query_id", "product_id"], ignore_index=True
    )


def candidate_recall_ceiling(
    pool: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Macro recall of an unordered candidate pool against the full target."""
    _require_columns(pool, {"user_id", "product_id"}, "pool")
    _require_columns(target, {"user_id", "product_id", "item_type"}, "target")
    labels = target[["user_id", "product_id", "item_type"]].drop_duplicates().copy()
    labels["item_type"] = labels["item_type"].astype("string")
    hits = labels.merge(
        pool[["user_id", "product_id"]].drop_duplicates().assign(hit=np.int8(1)),
        on=["user_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    hits["hit"] = hits["hit"].fillna(0)
    rows = []
    for segment in ("overall", "repeat", "explore"):
        selected = hits if segment == "overall" else hits[hits["item_type"].eq(segment)]
        per_user = selected.groupby("user_id", observed=True)["hit"].mean()
        rows.append(
            {
                "segment": segment,
                "recall": float(per_user.mean()) if len(per_user) else np.nan,
                "users": int(len(per_user)),
            }
        )
    return pd.DataFrame(rows)


def split_group_users(
    user_ids: Iterable[int],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split complete user-level ranking groups."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    users = np.asarray(sorted(set(user_ids)))
    if len(users) < 2:
        raise ValueError("at least two users are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(users)
    validation_size = max(1, round(len(users) * validation_fraction))
    validation = np.sort(shuffled[:validation_size])
    train = np.sort(shuffled[validation_size:])
    assert np.intersect1d(train, validation).size == 0
    return train, validation


def assert_groups_contiguous(frame: pd.DataFrame, group_column: str = "query_id") -> None:
    """Assert that every group occupies one contiguous block of rows."""
    _require_columns(frame, {group_column}, "frame")
    groups = frame[group_column].to_numpy()
    if len(groups) == 0:
        return
    run_starts = np.r_[True, groups[1:] != groups[:-1]]
    runs = groups[run_starts]
    if len(runs) != len(np.unique(runs)):
        raise AssertionError(f"{group_column} is not contiguous")


def heuristic_scores(
    candidates: pd.DataFrame,
    *,
    repeat_slots: int = 2,
    covis_slots: int = 1,
) -> np.ndarray:
    """Return a deterministic interleaving key without mixing retrieval scores.

    The default schedule emits two frequency-ranked repeat candidates followed
    by one similarity-ranked explore candidate in each three-position cycle.
    """
    _require_columns(candidates, {"repeat_rank", "covis_rank"}, "candidates")
    if repeat_slots <= 0 or covis_slots <= 0:
        raise ValueError("slot counts must be positive")
    cycle_size = repeat_slots + covis_slots
    repeat_rank = candidates["repeat_rank"].to_numpy(dtype=np.float32)
    covis_rank = candidates["covis_rank"].to_numpy(dtype=np.float32)
    repeat_position = (
        np.floor((repeat_rank - 1) / repeat_slots) * cycle_size
        + np.mod(repeat_rank - 1, repeat_slots)
        + 1
    )
    covis_position = (
        np.floor((covis_rank - 1) / covis_slots) * cycle_size
        + repeat_slots
        + np.mod(covis_rank - 1, covis_slots)
        + 1
    )
    position = np.fmin(repeat_position, covis_position)
    return -position.astype(np.float32)


def ranking_metrics(
    predictions: pd.DataFrame,
    target: pd.DataFrame,
    *,
    score_column: str,
    ks: tuple[int, ...] = (10, 20),
) -> pd.DataFrame:
    """Compute full-target macro nDCG/Recall, including missed retrieval items."""
    _require_columns(
        predictions,
        {"user_id", "target_order_id", "product_id", score_column},
        "predictions",
    )
    _require_columns(
        target, {"user_id", "order_id", "product_id", "item_type"}, "target"
    )
    if any(k <= 0 for k in ks):
        raise ValueError("ks must be positive")
    if predictions.duplicated([*GROUP_COLUMNS, "product_id"]).any():
        raise ValueError("prediction candidates must be unique within a group")

    ranked = predictions.sort_values(
        [*GROUP_COLUMNS, score_column, "product_id"],
        ascending=[True, True, False, True],
    ).copy()
    ranked["position"] = ranked.groupby(GROUP_COLUMNS, observed=True).cumcount().add(1)
    labels = target[["user_id", "order_id", "product_id", "item_type"]].rename(
        columns={"order_id": "target_order_id"}
    )
    labels = labels.drop_duplicates()
    target_counts = labels.groupby(GROUP_COLUMNS, observed=True).size().rename("overall")
    segment_counts = (
        labels.groupby([*GROUP_COLUMNS, "item_type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    denominators = target_counts.to_frame().join(segment_counts, how="left").fillna(0)

    max_k = max(ks)
    top = ranked.loc[ranked["position"] <= max_k].merge(
        labels,
        on=[*GROUP_COLUMNS, "product_id"],
        how="left",
        validate="one_to_one",
    )
    top["relevant"] = top["item_type"].notna().astype("int8")
    rows = []
    for k in sorted(set(ks)):
        selected = top[top["position"] <= k].copy()
        selected["discounted_gain"] = selected["relevant"] / np.log2(
            selected["position"] + 1
        )
        dcg = selected.groupby(GROUP_COLUMNS, observed=True)["discounted_gain"].sum()
        overall_denominator = denominators["overall"]
        ideal = overall_denominator.clip(upper=k).map(_ideal_dcg)
        ndcg = dcg.reindex(denominators.index, fill_value=0).div(ideal)
        hits = (
            selected.loc[selected["relevant"].eq(1)]
            .groupby(GROUP_COLUMNS, observed=True)
            .size()
        )
        recall = hits.reindex(denominators.index, fill_value=0).div(overall_denominator)
        rows.extend(
            [
                {"k": k, "metric": "ndcg", "segment": "overall", "value": ndcg.mean(), "groups": len(ndcg)},
                {"k": k, "metric": "recall", "segment": "overall", "value": recall.mean(), "groups": len(recall)},
            ]
        )
        for segment in ("repeat", "explore"):
            denominator = denominators.get(segment, pd.Series(0, index=denominators.index))
            eligible = denominator.gt(0)
            segment_hits = (
                selected.loc[selected["item_type"].astype("string").eq(segment)]
                .groupby(GROUP_COLUMNS, observed=True)
                .size()
                .reindex(denominators.index, fill_value=0)
            )
            segment_recall = segment_hits[eligible].div(denominator[eligible])
            segment_gain = selected.loc[
                selected["item_type"].astype("string").eq(segment)
            ].assign(gain=lambda frame: 1 / np.log2(frame["position"] + 1))
            segment_dcg = (
                segment_gain.groupby(GROUP_COLUMNS, observed=True)["gain"]
                .sum()
                .reindex(denominators.index, fill_value=0)
            )
            segment_ideal = denominator[eligible].clip(upper=k).map(_ideal_dcg)
            segment_ndcg = segment_dcg[eligible].div(segment_ideal)
            rows.extend(
                [
                    {"k": k, "metric": "ndcg", "segment": segment, "value": segment_ndcg.mean(), "groups": int(eligible.sum())},
                    {"k": k, "metric": "recall", "segment": segment, "value": segment_recall.mean(), "groups": int(eligible.sum())},
                ]
            )
    return pd.DataFrame(rows)


def _ideal_dcg(relevant_items: int) -> float:
    if relevant_items <= 0:
        return np.nan
    positions = np.arange(1, int(relevant_items) + 1)
    return float(np.sum(1 / np.log2(positions + 1)))


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
