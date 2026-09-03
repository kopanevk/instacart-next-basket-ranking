from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from src.data import TABLES, load_table, resolve_data_dir


def _write_minimal_dataset(directory, *, zipped=False):
    rows = {
        "orders": {
            "order_id": [1],
            "user_id": [1],
            "eval_set": ["prior"],
            "order_number": [1],
            "order_dow": [0],
            "order_hour_of_day": [10],
            "days_since_prior_order": [None],
        },
        "order_products__prior": {
            "order_id": [1],
            "product_id": [1],
            "add_to_cart_order": [1],
            "reordered": [0],
        },
        "order_products__train": {
            "order_id": [2],
            "product_id": [1],
            "add_to_cart_order": [1],
            "reordered": [1],
        },
        "products": {
            "product_id": [1],
            "product_name": ["apple"],
            "aisle_id": [1],
            "department_id": [1],
        },
        "aisles": {"aisle_id": [1], "aisle": ["fruit"]},
        "departments": {"department_id": [1], "department": ["produce"]},
    }
    for table in TABLES:
        frame = pd.DataFrame(rows[table])
        csv = frame.to_csv(index=False)
        if zipped:
            with ZipFile(
                directory / f"{table}.csv.zip", "w", compression=ZIP_DEFLATED
            ) as archive:
                archive.writestr(f"{table}.csv", csv)
        else:
            frame.to_csv(directory / f"{table}.csv", index=False)


def test_resolve_and_load_plain_csv(tmp_path):
    _write_minimal_dataset(tmp_path)
    assert resolve_data_dir(tmp_path) == tmp_path.resolve()
    orders = load_table("orders", tmp_path)
    assert orders.loc[0, "user_id"] == 1


def test_kaggle_style_csv_zip_is_supported(tmp_path):
    _write_minimal_dataset(tmp_path, zipped=True)
    products = load_table("products", tmp_path)
    assert products.loc[0, "product_name"] == "apple"
