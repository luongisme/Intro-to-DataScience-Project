"""Comprehensive Unit & Integration tests for Olist Merge Pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.paths import RAW_DIR
from src.data_loader import DATASETS, load_data
from src.pipeline.merger import OlistDataMerger
from src.pipeline.run_merge import _load_and_prepare, run_pipeline
from src.pipeline.schema import (
    EXPECTED_FINAL_COLUMNS,
    ORDER_ITEMS_SPEC,
    TableLoadSpec,
)
from src.pipeline.validator import MergeValidator


# 1. Multi-item order preservation
def test_row_count_preserved_with_multi_item_order(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    merger = OlistDataMerger()
    merged = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        synthetic_products,
        synthetic_sellers,
    )
    # Order O1 has 2 items, O2 has 1 item -> Exactly 3 rows
    assert len(merged) == 3
    assert len(merged) == len(synthetic_order_items)


# 2. Merge fails on duplicate key in right table (validate="m:1")
def test_merge_fails_on_duplicate_key_in_right_table(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    dup_products = pd.concat([synthetic_products, synthetic_products.iloc[[0]]], ignore_index=True)
    merger = OlistDataMerger()
    with pytest.raises(pd.errors.MergeError):
        merger.merge_tables(
            synthetic_order_items,
            synthetic_orders,
            synthetic_customers,
            dup_products,
            synthetic_sellers,
        )


# 3. Left join keeps item when product missing; orphan check catches it
def test_left_join_keeps_item_when_product_missing(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    # Remove P2 from products
    subset_products = synthetic_products[synthetic_products["product_id"] == "P1"]
    merger = OlistDataMerger()
    merged = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        subset_products,
        synthetic_sellers,
    )

    # Item with P2 is kept in output with NaN product columns
    assert len(merged) == len(synthetic_order_items)
    p2_row = merged[merged["product_id"] == "P2"]
    assert len(p2_row) == 1
    assert pd.isna(p2_row["product_category_name"].values[0])

    # Validator orphan check must fail and report orphan
    validator = MergeValidator()
    with pytest.raises(ValueError, match="orphan product_id"):
        validator.validate_hard_rules(
            merged,
            synthetic_order_items,
            synthetic_orders,
            subset_products,
            synthetic_sellers,
        )


# 4. Dropped columns are absent from output
def test_dropped_columns_absent(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    merger = OlistDataMerger()
    merged = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        synthetic_products,
        synthetic_sellers,
    )
    dropped = [
        "customer_unique_id",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
    ]
    for col in dropped:
        assert col not in merged.columns


# 5. Output has exactly 25 expected columns in order
def test_exact_columns_and_order(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    merger = OlistDataMerger()
    merged = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        synthetic_products,
        synthetic_sellers,
    )
    assert len(merged.columns) == 25
    assert list(merged.columns) == EXPECTED_FINAL_COLUMNS


# 6. No suffix columns (_x, _y)
def test_no_suffix_columns(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    merger = OlistDataMerger()
    merged = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        synthetic_products,
        synthetic_sellers,
    )
    assert not any(c.endswith("_x") or c.endswith("_y") for c in merged.columns)


# 7. Datetime parsing: valid parsed, empty string -> NaT, malformed -> raise
def test_datetime_parsing():
    valid_series = pd.Series(["2017-02-01 08:00:00", "2018-05-10 12:30:00"])
    parsed = pd.to_datetime(valid_series, errors="raise")
    assert pd.api.types.is_datetime64_any_dtype(parsed)

    empty_series = pd.Series(["", None, np.nan])
    parsed_empty = pd.to_datetime(empty_series, errors="raise")
    assert parsed_empty.isna().all()

    malformed_series = pd.Series(["2017-02-01", "not_a_valid_date"])
    with pytest.raises(Exception):
        pd.to_datetime(malformed_series, errors="raise")


# 8. Zip code leading zero preserved as string
def test_zip_code_leading_zero_preserved(tmp_path):
    csv_file = tmp_path / "zip_test.csv"
    csv_file.write_text("customer_id,customer_zip_code_prefix\nC1,01310\nC2,04571\n")
    df = pd.read_csv(csv_file, dtype={"customer_zip_code_prefix": "string"})
    assert df["customer_zip_code_prefix"].iloc[0] == "01310"
    assert df["customer_zip_code_prefix"].dtype == "string"


# 9a. Missing file raises clear FileNotFoundError naming file and folder
def test_missing_file_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr("src.data_loader.RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        load_data("order_items")
    error_msg = str(exc_info.value)
    assert "olist_order_items_dataset.csv" in error_msg
    assert str(tmp_path) in error_msg


# 9b. Missing expected column raises clear ValueError listing missing columns
def test_missing_column_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr("src.data_loader.RAW_DIR", tmp_path)
    csv_file = tmp_path / "olist_order_items_dataset.csv"
    # Write CSV missing 'price' and 'freight_value'
    csv_file.write_text(
        "order_id,order_item_id,product_id,seller_id,shipping_limit_date\n"
        "O1,1,P1,S1,2017-02-05 10:00:00\n"
    )
    with pytest.raises(ValueError) as exc_info:
        _load_and_prepare(ORDER_ITEMS_SPEC)
    assert "missing expected columns" in str(exc_info.value)


# 10. Validation catches broken merge (e.g. duplicated rows)
def test_validation_catches_broken_merge(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    merger = OlistDataMerger()
    merged = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        synthetic_products,
        synthetic_sellers,
    )
    # Deliberately duplicate a row
    broken_merged = pd.concat([merged, merged.iloc[[0]]], ignore_index=True)
    validator = MergeValidator()
    with pytest.raises(ValueError, match="Row count changed"):
        validator.validate_hard_rules(
            broken_merged,
            synthetic_order_items,
            synthetic_orders,
            synthetic_products,
            synthetic_sellers,
        )


# 11. Input DataFrames are not mutated
def test_inputs_not_mutated(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    items_copy = synthetic_order_items.copy(deep=True)
    orders_copy = synthetic_orders.copy(deep=True)
    customers_copy = synthetic_customers.copy(deep=True)
    products_copy = synthetic_products.copy(deep=True)
    sellers_copy = synthetic_sellers.copy(deep=True)

    merger = OlistDataMerger()
    _ = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        synthetic_products,
        synthetic_sellers,
    )

    pd.testing.assert_frame_equal(synthetic_order_items, items_copy)
    pd.testing.assert_frame_equal(synthetic_orders, orders_copy)
    pd.testing.assert_frame_equal(synthetic_customers, customers_copy)
    pd.testing.assert_frame_equal(synthetic_products, products_copy)
    pd.testing.assert_frame_equal(synthetic_sellers, sellers_copy)


# 12. Deterministic execution
def test_deterministic(
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
):
    merger = OlistDataMerger()
    run1 = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        synthetic_products,
        synthetic_sellers,
    )
    run2 = merger.merge_tables(
        synthetic_order_items,
        synthetic_orders,
        synthetic_customers,
        synthetic_products,
        synthetic_sellers,
    )
    pd.testing.assert_frame_equal(run1, run2)