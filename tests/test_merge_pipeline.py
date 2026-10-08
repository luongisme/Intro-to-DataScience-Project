"""Comprehensive Unit & Integration tests for Olist Merge Pipeline."""

import pandas as pd
import pytest

from src.pipeline.merger import OlistDataMerger
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