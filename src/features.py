"""Compact leakage-safe feature tables for the development ranking cutoff."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation import Stage, assert_no_target_in_history


USER_ITEM_FEATURES = [
    "ui_n_orders",
    "ui_order_share",
    "ui_orders_since_last",
    "ui_in_last_basket",
]
ITEM_FEATURES = ["i_n_orders", "i_reorder_rate", "aisle_id", "department_id"]
USER_FEATURES = ["u_n_orders", "u_mean_basket_size", "u_reorder_ratio"]
RETRIEVAL_FEATURES = ["repeat_score", "covis_score"]
INTERVAL_FEATURES = ["days_since_prior_order"]
CLOCK_FEATURES = ["order_hour_of_day", "order_dow"]
BASE_FEATURES = USER_ITEM_FEATURES + ITEM_FEATURES + USER_FEATURES + RETRIEVAL_FEATURES
ALL_FEATURES = BASE_FEATURES + INTERVAL_FEATURES + CLOCK_FEATURES


@dataclass(frozen=True)
class FeatureTables:
    user_item: pd.DataFrame
    item: pd.DataFrame
    user: pd.DataFrame
    context: pd.DataFrame


def build_feature_tables(
    history: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    examples: pd.DataFrame,
    *,
    stage: Stage = "ranker",
) -> FeatureTables:
    """Aggregate every feature from one centrally validated history cutoff."""
    _validate_feature_cutoff(history, examples, stage)
    target_id_column = f"{stage}_target_order_id"
    target_number_column = f"{stage}_target_order_number"
    history_max_column = f"{stage}_history_max_order_number"
    required_history = {
        "order_id",
        "user_id",
        "order_number",
        "product_id",
        "reordered",
    }
    _require_columns(history, required_history, "history")
    _require_columns(products, {"product_id", "aisle_id", "department_id"}, "products")

    user = (
        history.groupby("user_id", observed=True)
        .agg(
            u_n_orders=("order_id", "nunique"),
            u_n_items=("product_id", "size"),
            u_reorder_ratio=("reordered", "mean"),
        )
        .reset_index()
    )
    user["u_mean_basket_size"] = user["u_n_items"] / user["u_n_orders"]
    user = user.drop(columns="u_n_items")

    user_item = (
        history.groupby(["user_id", "product_id"], observed=True)
        .agg(
            ui_n_orders=("order_id", "nunique"),
            ui_last_order_number=("order_number", "max"),
        )
        .reset_index()
        .merge(user[["user_id", "u_n_orders"]], on="user_id", validate="many_to_one")
        .merge(
            examples[
                [
                    "user_id",
                    target_number_column,
                    history_max_column,
                ]
            ],
            on="user_id",
            validate="many_to_one",
        )
    )
    user_item["ui_order_share"] = (
        user_item["ui_n_orders"] / user_item["u_n_orders"]
    )
    user_item["ui_orders_since_last"] = (
        user_item[target_number_column]
        - user_item["ui_last_order_number"]
    )
    user_item["ui_in_last_basket"] = user_item["ui_last_order_number"].eq(
        user_item[history_max_column]
    )
    user_item = user_item[["user_id", "product_id", *USER_ITEM_FEATURES]]

    item = (
        history.groupby("product_id", observed=True)
        .agg(i_n_orders=("order_id", "nunique"), i_reorder_rate=("reordered", "mean"))
        .reset_index()
        .merge(
            products[["product_id", "aisle_id", "department_id"]],
            on="product_id",
            how="left",
            validate="one_to_one",
        )
    )

    context = examples[["user_id", target_id_column]].merge(
        orders[
            [
                "order_id",
                "days_since_prior_order",
                "order_hour_of_day",
                "order_dow",
            ]
        ],
        left_on=target_id_column,
        right_on="order_id",
        validate="one_to_one",
    )
    context = context.rename(columns={target_id_column: "target_order_id"})[
        ["user_id", "target_order_id", *INTERVAL_FEATURES, *CLOCK_FEATURES]
    ]

    return FeatureTables(
        user_item=_compact_feature_dtypes(user_item),
        item=_compact_feature_dtypes(item),
        user=_compact_feature_dtypes(user[["user_id", *USER_FEATURES]]),
        context=_compact_feature_dtypes(context),
    )


def assemble_ranking_features(
    labeled_candidates: pd.DataFrame,
    tables: FeatureTables,
) -> pd.DataFrame:
    """Join compact aggregates to labeled candidates without changing groups."""
    required = {
        "user_id",
        "target_order_id",
        "query_id",
        "product_id",
        "label",
        "repeat_score",
        "covis_score",
    }
    _require_columns(labeled_candidates, required, "labeled_candidates")
    result = labeled_candidates.merge(
        tables.user_item,
        on=["user_id", "product_id"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        tables.item, on="product_id", how="left", validate="many_to_one"
    )
    result = result.merge(
        tables.user, on="user_id", how="left", validate="many_to_one"
    )
    result = result.merge(
        tables.context,
        on=["user_id", "target_order_id"],
        how="left",
        validate="many_to_one",
    )
    result["ui_n_orders"] = result["ui_n_orders"].fillna(0)
    result["ui_order_share"] = result["ui_order_share"].fillna(0)
    result["ui_in_last_basket"] = result["ui_in_last_basket"].fillna(0)
    result["i_n_orders"] = result["i_n_orders"].fillna(0)
    result["aisle_id"] = result["aisle_id"].fillna(-1)
    result["department_id"] = result["department_id"].fillna(-1)
    result = _compact_feature_dtypes(result)
    return result.sort_values(["query_id", "product_id"], ignore_index=True)


def _validate_feature_cutoff(
    history: pd.DataFrame,
    examples: pd.DataFrame,
    stage: Stage,
) -> None:
    target_id_column = f"{stage}_target_order_id"
    target_number_column = f"{stage}_target_order_number"
    history_max_column = f"{stage}_history_max_order_number"
    _require_columns(history, {"user_id", "order_id", "order_number"}, "history")
    _require_columns(
        examples,
        {
            "user_id",
            target_id_column,
            target_number_column,
            history_max_column,
        },
        "examples",
    )
    assert_no_target_in_history(history, examples, stage)
    checked = history[["user_id", "order_number"]].merge(
        examples[["user_id", history_max_column]],
        on="user_id",
        validate="many_to_one",
    )
    if not (checked["order_number"] <= checked[history_max_column]).all():
        raise AssertionError(
            f"feature history exceeds the centralized {stage} cutoff"
        )


def _compact_feature_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    int8_columns = ["ui_in_last_basket", "department_id", "order_hour_of_day", "order_dow"]
    int16_columns = ["ui_n_orders", "u_n_orders", "aisle_id"]
    int32_columns = ["i_n_orders"]
    float_columns = [
        "ui_order_share",
        "ui_orders_since_last",
        "i_reorder_rate",
        "u_mean_basket_size",
        "u_reorder_ratio",
        "days_since_prior_order",
        "repeat_score",
        "covis_score",
    ]
    for column in int8_columns:
        if column in result:
            result[column] = result[column].astype("int8")
    for column in int16_columns:
        if column in result:
            result[column] = result[column].astype("int16")
    for column in int32_columns:
        if column in result:
            result[column] = result[column].astype("int32")
    for column in float_columns:
        if column in result:
            result[column] = result[column].astype("float32")
    return result


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
