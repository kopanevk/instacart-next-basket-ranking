"""Dataset discovery and loading for the Instacart competition tables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


TABLES = (
    "orders",
    "order_products__prior",
    "order_products__train",
    "products",
    "aisles",
    "departments",
)

REQUIRED_COLUMNS = {
    "orders": {
        "order_id",
        "user_id",
        "eval_set",
        "order_number",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
    },
    "order_products__prior": {
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
    },
    "order_products__train": {
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
    },
    "products": {"product_id", "product_name", "aisle_id", "department_id"},
    "aisles": {"aisle_id", "aisle"},
    "departments": {"department_id", "department"},
}

DTYPES = {
    "orders": {
        "order_id": "int32",
        "user_id": "int32",
        "eval_set": "category",
        "order_number": "int16",
        "order_dow": "int8",
        "order_hour_of_day": "int8",
        "days_since_prior_order": "float32",
    },
    "order_products__prior": {
        "order_id": "int32",
        "product_id": "int32",
        "add_to_cart_order": "int16",
        "reordered": "int8",
    },
    "order_products__train": {
        "order_id": "int32",
        "product_id": "int32",
        "add_to_cart_order": "int16",
        "reordered": "int8",
    },
    "products": {
        "product_id": "int32",
        "product_name": "string",
        "aisle_id": "int16",
        "department_id": "int8",
    },
    "aisles": {"aisle_id": "int16", "aisle": "string"},
    "departments": {"department_id": "int8", "department": "string"},
}


def _table_path(directory: Path, table: str) -> Path | None:
    """Return the CSV (plain or Kaggle-style zipped) for one table."""
    for suffix in (".csv", ".csv.zip"):
        path = directory / f"{table}{suffix}"
        if path.is_file():
            return path
    return None


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Locate a directory containing all six expected Instacart tables.

    Resolution order is: explicit argument, ``INSTACART_DATA_DIR``, common local
    directories, the standard Kaggle dataset directory, then direct children of
    ``/kaggle/input``. Both ``.csv`` and ``.csv.zip`` files are supported.
    """
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir).expanduser())
    if env_dir := os.getenv("INSTACART_DATA_DIR"):
        candidates.append(Path(env_dir).expanduser())

    candidates.extend(
        [
            Path("data/raw"),
            Path("data"),
            Path("/kaggle/input/instacart-market-basket-analysis"),
        ]
    )
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        candidates.extend(path for path in kaggle_root.iterdir() if path.is_dir())

    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if all(_table_path(candidate, table) is not None for table in TABLES):
            return candidate

    locations = "\n  - ".join(checked)
    raise FileNotFoundError(
        "Could not find all Instacart tables. Set INSTACART_DATA_DIR or pass "
        f"data_dir explicitly. Checked:\n  - {locations}"
    )


def load_table(
    table: str,
    data_dir: str | Path | None = None,
    *,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load and schema-check one table without hidden caching or global state."""
    if table not in TABLES:
        raise ValueError(f"Unknown table {table!r}; expected one of {TABLES}")
    directory = resolve_data_dir(data_dir)
    path = _table_path(directory, table)
    assert path is not None  # resolve_data_dir already checked every table

    columns = list(usecols) if usecols is not None else None
    dtype = DTYPES[table]
    if columns is not None:
        dtype = {column: value for column, value in dtype.items() if column in columns}
    frame = pd.read_csv(path, usecols=columns, dtype=dtype)

    expected = REQUIRED_COLUMNS[table] if columns is None else set(columns)
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def load_instacart(
    data_dir: str | Path | None = None,
    tables: Iterable[str] = TABLES,
) -> dict[str, pd.DataFrame]:
    """Load each requested table exactly once and return it by table name."""
    directory = resolve_data_dir(data_dir)
    requested = tuple(tables)
    unknown = set(requested).difference(TABLES)
    if unknown:
        raise ValueError(f"Unknown tables: {sorted(unknown)}")
    return {table: load_table(table, directory) for table in requested}

