"""Comprehensive Unit & Integration tests for Olist Merge Pipeline."""

import pandas as pd
import pytest

from src.pipeline.merger import OlistDataMerger

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