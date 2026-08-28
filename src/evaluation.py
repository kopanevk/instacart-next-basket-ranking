"""Chronological evaluation protocol and repeat/explore decomposition."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


Stage = Literal["ranker", "final"]


def build_evaluation_examples(orders: pd.DataFrame) -> pd.DataFrame:
    """Define the two chronological examples for every labeled user.

    ``ranker`` targets order n-1 and sees history through n-2. ``final`` targets
    order n and sees the observed prior history through n-1. Final labels are
    deliberately not accepted by this function.
    """
    required = {"order_id", "user_id", "eval_set", "order_number"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"orders is missing columns: {sorted(missing)}")

    labeled = orders.loc[
        orders["eval_set"].astype("string").eq("train"),
        ["user_id", "order_id", "order_number"],
    ].copy()
    if labeled.empty:
        raise ValueError("orders contains no eval_set == 'train' users")
    if labeled["user_id"].duplicated().any():
        raise ValueError("Each evaluation user must have exactly one train order")
    labeled = labeled.rename(
        columns={
            "order_id": "final_target_order_id",
            "order_number": "final_target_order_number",
        }
    )
    labeled["ranker_target_order_number"] = (
        labeled["final_target_order_number"] - 1
    )

    prior_orders = orders.loc[
        orders["eval_set"].astype("string").eq("prior"),
        ["user_id", "order_id", "order_number"],
    ].rename(
        columns={
            "order_id": "ranker_target_order_id",
            "order_number": "ranker_target_order_number",
        }
    )
    examples = labeled.merge(
        prior_orders,
        on=["user_id", "ranker_target_order_number"],
        how="left",
        validate="one_to_one",
    )
    if examples["ranker_target_order_id"].isna().any():
        bad = examples.loc[
            examples["ranker_target_order_id"].isna(), "user_id"
        ].head().tolist()
        raise ValueError(f"Missing order n-1 for evaluation users, e.g. {bad}")

    examples["ranker_target_order_id"] = examples[
        "ranker_target_order_id"
    ].astype(orders["order_id"].dtype)
    examples["ranker_history_max_order_number"] = (
        examples["ranker_target_order_number"] - 1
    )
    examples["final_history_max_order_number"] = (
        examples["final_target_order_number"] - 1
    )
    return examples[
        [
            "user_id",
            "ranker_target_order_id",
            "ranker_target_order_number",
            "ranker_history_max_order_number",
            "final_target_order_id",
            "final_target_order_number",
            "final_history_max_order_number",
        ]
    ].sort_values("user_id", ignore_index=True)


def history_orders(
    orders: pd.DataFrame,
    examples: pd.DataFrame,
    stage: Stage,
) -> pd.DataFrame:
    """Return only order metadata visible before each stage's target basket."""
    _check_stage(stage)
    max_column = f"{stage}_history_max_order_number"
    visible = orders.loc[
        orders["eval_set"].astype("string").eq("prior")
    ].merge(examples[["user_id", max_column]], on="user_id", how="inner")
    visible = visible.loc[visible["order_number"] <= visible[max_column]].copy()
    return visible.drop(columns=max_column).reset_index(drop=True)


def history_items(
    orders: pd.DataFrame,
    prior_items: pd.DataFrame,
    examples: pd.DataFrame,
    stage: Stage,
) -> pd.DataFrame:
    """Return product rows visible at a cutoff; accepts prior labels only."""
    visible_orders = history_orders(orders, examples, stage)
    result = visible_orders[["order_id", "user_id", "order_number"]].merge(
        prior_items,
        on="order_id",
        how="inner",
        validate="one_to_many",
    )
    assert_no_target_in_history(result, examples, stage)
    return result


def target_items(
    prior_items: pd.DataFrame,
    train_items: pd.DataFrame,
    examples: pd.DataFrame,
    stage: Stage,
) -> pd.DataFrame:
    """Return labels for a stage, keeping their source explicit."""
    _check_stage(stage)
    id_column = f"{stage}_target_order_id"
    source = prior_items if stage == "ranker" else train_items
    return examples[["user_id", id_column]].merge(
        source,
        left_on=id_column,
        right_on="order_id",
        how="inner",
        validate="one_to_many",
    )


def classify_repeat_explore(
    history: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Label each target item by whether the user bought it in visible history."""
    required = {"user_id", "product_id"}
    for name, frame in (("history", history), ("target", target)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    seen = history[["user_id", "product_id"]].drop_duplicates().assign(
        item_type="repeat"
    )
    result = target.merge(
        seen,
        on=["user_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    result["item_type"] = result["item_type"].fillna("explore").astype("category")
    return result


def retrieval_recall(
    candidates: pd.DataFrame,
    target: pd.DataFrame,
    ks: tuple[int, ...] = (10, 50, 100),
) -> pd.DataFrame:
    """Compute macro Recall@K for overall, repeat, and explore targets.

    Denominators always contain the complete relevant target segment. Users with
    no repeat (or explore) target items are excluded only from that segment's
    macro average. Users with targets but no retrieved candidates receive zero.
    """
    candidate_columns = {"user_id", "product_id", "rank"}
    target_columns = {"user_id", "product_id", "item_type"}
    for name, frame, required in (
        ("candidates", candidates, candidate_columns),
        ("target", target, target_columns),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if candidates.duplicated(["user_id", "product_id"]).any():
        raise ValueError("candidates must contain unique user-product pairs")
    if target.duplicated(["user_id", "product_id"]).any():
        raise ValueError("target must contain unique user-product pairs")
    if target["item_type"].isna().any():
        raise ValueError("target item_type cannot be missing")
    item_types = set(target["item_type"].astype("string").dropna().unique())
    if not item_types.issubset({"repeat", "explore"}):
        raise ValueError("target item_type must contain only repeat/explore")

    labels = target[["user_id", "product_id", "item_type"]].copy()
    labels["item_type"] = labels["item_type"].astype("string")
    retrieved = labels.merge(
        candidates[["user_id", "product_id", "rank"]],
        on=["user_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    results: list[dict[str, float | int | str]] = []
    for k in sorted(set(ks)):
        hit_labels = retrieved.assign(hit=retrieved["rank"].le(k).astype("int8"))

        for segment in ("overall", "repeat", "explore"):
            segment_labels = (
                hit_labels
                if segment == "overall"
                else hit_labels.loc[hit_labels["item_type"].eq(segment)]
            )
            per_user = segment_labels.groupby("user_id", observed=True)["hit"].mean()
            results.append(
                {
                    "k": k,
                    "segment": segment,
                    "recall": float(per_user.mean()) if len(per_user) else np.nan,
                    "users": int(len(per_user)),
                }
            )
    return pd.DataFrame(results, columns=["k", "segment", "recall", "users"])


def assert_no_target_in_history(
    history: pd.DataFrame,
    examples: pd.DataFrame,
    stage: Stage,
) -> None:
    """Assert that no source row belongs to the same example's target order."""
    _check_stage(stage)
    target_ids = set(examples[f"{stage}_target_order_id"].tolist())
    overlap = target_ids.intersection(history["order_id"].unique())
    assert not overlap, f"{stage} target orders leaked into history: {list(overlap)[:5]}"


def assert_protocol_integrity(
    orders: pd.DataFrame,
    prior_items: pd.DataFrame,
    train_items: pd.DataFrame,
    examples: pd.DataFrame,
) -> None:
    """Run high-value assertions for the temporal protocol and label sources."""
    train_users = set(
        orders.loc[orders["eval_set"].astype("string").eq("train"), "user_id"]
    )
    assert set(examples["user_id"]) == train_users
    assert examples["user_id"].is_unique
    assert examples[
        ["ranker_target_order_id", "final_target_order_id"]
    ].notna().all().all()
    assert (
        examples["ranker_target_order_number"]
        == examples["final_target_order_number"] - 1
    ).all()
    assert (
        examples["ranker_history_max_order_number"]
        < examples["ranker_target_order_number"]
    ).all()
    assert (
        examples["final_history_max_order_number"]
        < examples["final_target_order_number"]
    ).all()

    order_source = orders.set_index("order_id")["eval_set"].astype("string")
    prior_order_ids = set(prior_items["order_id"].unique())
    train_order_ids = set(train_items["order_id"].unique())
    assert prior_order_ids.issubset(set(order_source[order_source.eq("prior")].index))
    assert train_order_ids.issubset(set(order_source[order_source.eq("train")].index))
    assert prior_order_ids.isdisjoint(train_order_ids)
    assert set(examples["ranker_target_order_id"]).issubset(prior_order_ids)
    assert set(examples["final_target_order_id"]).issubset(train_order_ids)

    for stage in ("ranker", "final"):
        visible = history_orders(orders, examples, stage)
        assert visible["eval_set"].astype("string").eq("prior").all()
        assert_no_target_in_history(visible, examples, stage)
        max_column = f"{stage}_history_max_order_number"
        checked = visible.merge(
            examples[["user_id", max_column]], on="user_id", validate="many_to_one"
        )
        assert (checked["order_number"] <= checked[max_column]).all()


def _check_stage(stage: str) -> None:
    if stage not in {"ranker", "final"}:
        raise ValueError("stage must be 'ranker' or 'final'")
