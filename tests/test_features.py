import pandas as pd
import pytest

from src.features import (
    ALL_FEATURES,
    assemble_ranking_features,
    build_feature_tables,
)


def _feature_inputs():
    history = pd.DataFrame(
        [
            (11, 1, 1, 10, 0),
            (11, 1, 1, 20, 0),
            (12, 1, 2, 10, 1),
            (21, 2, 1, 30, 0),
        ],
        columns=["order_id", "user_id", "order_number", "product_id", "reordered"],
    )
    orders = pd.DataFrame(
        [
            (11, 1, 1, None, 8, 1),
            (12, 1, 2, 7.0, 9, 2),
            (13, 1, 3, 8.0, 10, 3),
            (21, 2, 1, None, 11, 4),
            (22, 2, 2, 5.0, 12, 5),
        ],
        columns=[
            "order_id", "user_id", "order_number", "days_since_prior_order",
            "order_hour_of_day", "order_dow",
        ],
    )
    products = pd.DataFrame(
        [(10, 1, 1), (20, 2, 1), (30, 3, 2), (99, 4, 2)],
        columns=["product_id", "aisle_id", "department_id"],
    )
    examples = pd.DataFrame(
        [(1, 13, 3, 2), (2, 22, 2, 1)],
        columns=[
            "user_id", "ranker_target_order_id", "ranker_target_order_number",
            "ranker_history_max_order_number",
        ],
    )
    return history, orders, products, examples


def test_target_order_is_rejected_by_central_feature_cutoff():
    history, orders, products, examples = _feature_inputs()
    leaked = pd.concat(
        [
            history,
            pd.DataFrame([(13, 1, 3, 99, 0)], columns=history.columns),
        ],
        ignore_index=True,
    )
    with pytest.raises(AssertionError):
        build_feature_tables(leaked, orders, products, examples)


def test_target_only_product_does_not_affect_allowed_feature_tables():
    history, orders, products, examples = _feature_inputs()
    tables = build_feature_tables(history, orders, products, examples)

    assert 99 not in set(tables.item["product_id"])
    user_one_item_ten = tables.user_item.loc[
        tables.user_item["user_id"].eq(1) & tables.user_item["product_id"].eq(10)
    ].iloc[0]
    assert user_one_item_ten["ui_n_orders"] == 2
    assert user_one_item_ten["ui_orders_since_last"] == 1
    assert user_one_item_ten["ui_in_last_basket"] == 1


def test_feature_assembly_has_all_declared_features_and_safe_unseen_values():
    history, orders, products, examples = _feature_inputs()
    tables = build_feature_tables(history, orders, products, examples)
    candidates = pd.DataFrame(
        [
            (1, 13, 13, 10, 1, 2.0, None),
            (1, 13, 13, 99, 0, None, 0.5),
        ],
        columns=[
            "user_id", "target_order_id", "query_id", "product_id", "label",
            "repeat_score", "covis_score",
        ],
    )
    features = assemble_ranking_features(candidates, tables)

    assert set(ALL_FEATURES).issubset(features.columns)
    unseen = features.loc[features["product_id"].eq(99)].iloc[0]
    assert unseen["ui_n_orders"] == 0
    assert unseen["ui_order_share"] == 0
    assert pd.isna(unseen["ui_orders_since_last"])

