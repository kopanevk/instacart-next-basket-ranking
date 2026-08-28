import pandas as pd
import pytest

from src.evaluation import (
    assert_no_target_in_history,
    assert_protocol_integrity,
    build_evaluation_examples,
    classify_repeat_explore,
    history_items,
    history_orders,
    target_items,
)


@pytest.fixture
def tiny_data():
    orders = pd.DataFrame(
        [
            (11, 1, "prior", 1),
            (12, 1, "prior", 2),
            (13, 1, "prior", 3),
            (14, 1, "train", 4),
            (21, 2, "prior", 1),
            (22, 2, "prior", 2),
            (23, 2, "train", 3),
            (31, 3, "prior", 1),
            (32, 3, "test", 2),
        ],
        columns=["order_id", "user_id", "eval_set", "order_number"],
    )
    prior = pd.DataFrame(
        [
            (11, 100, 1, 0),
            (12, 101, 1, 0),
            (13, 100, 1, 1),
            (13, 102, 2, 0),
            (21, 200, 1, 0),
            (22, 201, 1, 0),
            (31, 300, 1, 0),
        ],
        columns=["order_id", "product_id", "add_to_cart_order", "reordered"],
    )
    train = pd.DataFrame(
        [(14, 100, 1, 1), (14, 103, 2, 0), (23, 200, 1, 1)],
        columns=["order_id", "product_id", "add_to_cart_order", "reordered"],
    )
    return orders, prior, train


def test_examples_cover_only_labeled_users_with_correct_cutoffs(tiny_data):
    orders, _, _ = tiny_data
    examples = build_evaluation_examples(orders)

    assert examples["user_id"].tolist() == [1, 2]
    assert examples["ranker_target_order_id"].tolist() == [13, 22]
    assert examples["ranker_history_max_order_number"].tolist() == [2, 1]
    assert examples["final_target_order_id"].tolist() == [14, 23]
    assert examples["final_history_max_order_number"].tolist() == [3, 2]


def test_ranker_history_stops_before_n_minus_one(tiny_data):
    orders, prior, _ = tiny_data
    examples = build_evaluation_examples(orders)
    visible_orders = history_orders(orders, examples, "ranker")
    visible_items = history_items(orders, prior, examples, "ranker")

    assert set(visible_orders["order_id"]) == {11, 12, 21}
    assert set(visible_items["order_id"]) == {11, 12, 21}
    assert set(examples["ranker_target_order_id"]).isdisjoint(visible_items["order_id"])


def test_final_history_uses_prior_only_and_not_final_labels(tiny_data):
    orders, prior, train = tiny_data
    examples = build_evaluation_examples(orders)
    visible = history_items(orders, prior, examples, "final")
    labels = target_items(prior, train, examples, "final")

    assert set(visible["order_id"]) == {11, 12, 13, 21, 22}
    assert set(visible["order_id"]).isdisjoint(train["order_id"])
    assert set(labels["order_id"]) == {14, 23}


def test_target_rows_never_appear_in_same_example_history(tiny_data):
    orders, prior, _ = tiny_data
    examples = build_evaluation_examples(orders)
    visible = history_items(orders, prior, examples, "ranker")
    assert_no_target_in_history(visible, examples, "ranker")

    leaked = pd.concat([visible, prior.loc[prior["order_id"].eq(13)]])
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_target_in_history(leaked, examples, "ranker")


def test_repeat_explore_uses_only_visible_history(tiny_data):
    orders, prior, train = tiny_data
    examples = build_evaluation_examples(orders)
    visible = history_items(orders, prior, examples, "final")
    labels = target_items(prior, train, examples, "final")
    classified = classify_repeat_explore(visible, labels)

    actual = dict(zip(classified["product_id"], classified["item_type"].astype(str)))
    assert actual == {100: "repeat", 103: "explore", 200: "repeat"}


def test_full_protocol_assertions(tiny_data):
    orders, prior, train = tiny_data
    examples = build_evaluation_examples(orders)
    assert_protocol_integrity(orders, prior, train, examples)


def test_mislabeled_prior_source_is_rejected(tiny_data):
    orders, prior, train = tiny_data
    examples = build_evaluation_examples(orders)
    contaminated = pd.concat([prior, train.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError):
        assert_protocol_integrity(orders, contaminated, train, examples)

