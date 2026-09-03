"""Immutable configuration and post-hoc segments for the final MVP run."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import ALL_FEATURES


FROZEN_COMPONENT_K = 150
FROZEN_MAX_NEIGHBORS = 100
FROZEN_TOP_K = 10
FROZEN_BATCH_SIZE = 2_000
FROZEN_FEATURES = (
    "ui_n_orders",
    "ui_order_share",
    "ui_orders_since_last",
    "ui_in_last_basket",
    "i_n_orders",
    "i_reorder_rate",
    "aisle_id",
    "department_id",
    "u_n_orders",
    "u_mean_basket_size",
    "u_reorder_ratio",
    "repeat_score",
    "covis_score",
    "days_since_prior_order",
    "order_hour_of_day",
    "order_dow",
)
FROZEN_CLASSIFIER_PARAMS = {
    "iterations": 120,
    "depth": 7,
    "learning_rate": 0.15,
    "loss_function": "Logloss",
    "random_seed": 42,
}

# Frozen from the development policy analysis; reusing these prevents final
# labels from influencing the user-segment definitions.
REORDER_RATIO_LOW_UPPER = 0.2884615361690521
REORDER_RATIO_MEDIUM_UPPER = 0.5199999809265137


def assert_frozen_feature_contract(feature_names: list[str] | tuple[str, ...]) -> None:
    """Reject any feature drift before final model training or inference."""
    if tuple(feature_names) != FROZEN_FEATURES:
        raise AssertionError("final model feature list differs from frozen development")
    if tuple(ALL_FEATURES) != FROZEN_FEATURES:
        raise AssertionError("ALL_FEATURES drifted after development freeze")


def reorder_ratio_segment(values: pd.Series) -> pd.Categorical:
    """Apply the frozen development low/medium/high user boundaries."""
    return pd.Categorical(
        np.select(
            [
                values.le(REORDER_RATIO_LOW_UPPER),
                values.le(REORDER_RATIO_MEDIUM_UPPER),
            ],
            ["low-repeat", "medium-repeat"],
            default="high-repeat",
        ),
        categories=["low-repeat", "medium-repeat", "high-repeat"],
        ordered=True,
    )


def target_basket_segment(values: pd.Series) -> pd.Categorical:
    """Assign predeclared, interpretable final-basket size bands."""
    return pd.Categorical(
        np.select(
            [values.le(5), values.le(10)],
            ["small", "medium"],
            default="large",
        ),
        categories=["small", "medium", "large"],
        ordered=True,
    )
