import pandas as pd
import pytest

from src.evaluation import (
    build_evaluation_examples,
    history_items,
    target_items,
)
from src.features import ALL_FEATURES, build_feature_tables
from src.final import (
    FROZEN_CLASSIFIER_PARAMS,
    FROZEN_COMPONENT_K,
    FROZEN_FEATURES,
    FROZEN_TOP_K,
    assert_frozen_feature_contract,
    target_basket_segment,
)
from src.ranking import attach_groups_and_labels
from src.retrieval import personal_repeat_candidates


def _final_inputs():
    orders = pd.DataFrame(
        [
            (11, 1, "prior", 1, 0, 8, None),
            (12, 1, "prior", 2, 1, 9, 7.0),
            (13, 1, "prior", 3, 2, 10, 8.0),
            (14, 1, "train", 4, 3, 11, 9.0),
        ],
        columns=[
            "order_id",
            "user_id",
            "eval_set",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
        ],
    )
    prior = pd.DataFrame(
        [
            (11, 10, 1, 0),
            (12, 20, 1, 0),
            (13, 10, 1, 1),
        ],
        columns=["order_id", "product_id", "add_to_cart_order", "reordered"],
    )
    train = pd.DataFrame(
        [(14, 10, 1, 1), (14, 77, 2, 0)],
        columns=["order_id", "product_id", "add_to_cart_order", "reordered"],
    )
    products = pd.DataFrame(
        [(10, 1, 1), (20, 2, 1), (77, 3, 2)],
        columns=["product_id", "aisle_id", "department_id"],
    )
    return orders, prior, train, products


def test_final_features_stop_at_n_minus_one_and_train_is_labels_only():
    orders, prior, train, products = _final_inputs()
    examples = build_evaluation_examples(orders)
    history = history_items(orders, prior, examples, "final")
    labels = target_items(prior, train, examples, "final")
    tables = build_feature_tables(history, orders, products, examples, stage="final")

    assert history["order_number"].max() == 3
    assert examples["final_history_max_order_number"].item() == 3
    assert set(history["order_id"]).isdisjoint(labels["order_id"])
    assert 77 not in set(tables.item["product_id"])
    assert 77 not in set(tables.user_item["product_id"])
    item_ten = tables.user_item.loc[tables.user_item["product_id"].eq(10)].iloc[0]
    assert item_ten["ui_orders_since_last"] == 1
    assert item_ten["ui_in_last_basket"] == 1
    assert tables.context["target_order_id"].item() == 14


def test_final_target_order_is_rejected_by_feature_cutoff():
    orders, prior, train, products = _final_inputs()
    examples = build_evaluation_examples(orders)
    history = history_items(orders, prior, examples, "final")
    leaked_target = train.assign(user_id=1, order_number=4)[history.columns]
    leaked = pd.concat([history, leaked_target], ignore_index=True)

    with pytest.raises(AssertionError, match="final target orders leaked"):
        build_feature_tables(leaked, orders, products, examples, stage="final")


def test_final_candidates_and_labels_use_separate_sources():
    orders, prior, train, _ = _final_inputs()
    examples = build_evaluation_examples(orders)
    history = history_items(orders, prior, examples, "final")
    labels = target_items(prior, train, examples, "final")
    candidates = personal_repeat_candidates(history, FROZEN_COMPONENT_K)
    pool = candidates[["user_id", "product_id"]].assign(
        repeat_score=1.0,
        covis_score=float("nan"),
    )
    labeled = attach_groups_and_labels(
        pool, examples, labels, stage="final"
    )

    assert 77 not in set(candidates["product_id"])
    assert labeled["target_order_id"].unique().tolist() == [14]
    assert labeled.loc[labeled["product_id"].eq(10), "label"].item() == 1


def test_frozen_final_configuration_has_not_drifted():
    assert FROZEN_COMPONENT_K == 150
    assert FROZEN_TOP_K == 10
    assert FROZEN_CLASSIFIER_PARAMS == {
        "iterations": 120,
        "depth": 7,
        "learning_rate": 0.15,
        "loss_function": "Logloss",
        "random_seed": 42,
    }
    assert tuple(ALL_FEATURES) == FROZEN_FEATURES
    assert_frozen_feature_contract(ALL_FEATURES)
    with pytest.raises(AssertionError):
        assert_frozen_feature_contract(ALL_FEATURES[:-1])


def test_target_basket_segments_are_predeclared():
    values = pd.Series([1, 5, 6, 10, 11, 30])
    assert target_basket_segment(values).astype("string").tolist() == [
        "small",
        "small",
        "medium",
        "medium",
        "large",
        "large",
    ]
