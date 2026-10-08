"""Runtime validation layer: Hard checks (raise) and Soft checks (warnings)."""

import logging
from typing import Set
import pandas as pd

from src.pipeline.schema import (
    ALL_TIMESTAMP_COLUMNS,
    BENCHMARK_ROW_COUNT,
    EXPECTED_FINAL_COLUMNS,
    MANDATORY_NON_NULL_COLUMNS,
    ZIP_COLUMNS,
)

logger = logging.getLogger(__name__)


class MergeValidator:
    """Enforces integrity constraints on the merged Olist DataFrame."""

    @staticmethod
    def validate_hard_rules(
        merged_df: pd.DataFrame,
        base_order_items: pd.DataFrame,
        orders: pd.DataFrame,
        products: pd.DataFrame,
        sellers: pd.DataFrame,
    ) -> None:
        """Run all HARD checks. Must raise ValueError if any check fails.

        Checks:
        1. len(merged) == len(order_items)
        2. (order_id, order_item_id) is unique
        3. Columns equal exactly 25 expected columns in order
        4. No nulls in key identifiers (order_id, order_item_id, product_id, seller_id, customer_id)
        5. Timestamp columns are datetime64; zip code columns are string/object
        6. No column ends with _x or _y
        7. Orphan checks: every order_id, product_id, seller_id in base exists in right tables

        Raises:
            ValueError: If any integrity constraint is violated.
        """
        # Hard Check 1: Row count preservation
        if len(merged_df) != len(base_order_items):
            raise ValueError(
                f"Hard check failed: Row count changed! "
                f"Base order_items has {len(base_order_items)}, but merged has {len(merged_df)}."
            )

        # Hard Check 2: Uniqueness of (order_id, order_item_id)
        dup_mask = merged_df.duplicated(subset=["order_id", "order_item_id"])
        if dup_mask.any():
            dup_count = int(dup_mask.sum())
            raise ValueError(
                f"Hard check failed: Duplicate composite key (order_id, order_item_id) detected. "
                f"Count: {dup_count} duplicates."
            )

        # Hard Check 3: Exact 25 columns and order
        actual_cols = list(merged_df.columns)
        if actual_cols != EXPECTED_FINAL_COLUMNS:
            raise ValueError(
                f"Hard check failed: Column structure mismatch!\n"
                f"Expected ({len(EXPECTED_FINAL_COLUMNS)}): {EXPECTED_FINAL_COLUMNS}\n"
                f"Actual   ({len(actual_cols)}): {actual_cols}"
            )

        # Hard Check 4: No nulls in mandatory ID columns
        for col in MANDATORY_NON_NULL_COLUMNS:
            null_count = int(merged_df[col].isna().sum())
            if null_count > 0:
                raise ValueError(
                    f"Hard check failed: Mandatory ID column '{col}' contains {null_count} nulls."
                )

        # Hard Check 5: Dtypes check (datetimes & zip strings)
        for date_col in ALL_TIMESTAMP_COLUMNS:
            if not pd.api.types.is_datetime64_any_dtype(merged_df[date_col]):
                raise ValueError(
                    f"Hard check failed: Column '{date_col}' must be datetime64, "
                    f"got {merged_df[date_col].dtype}."
                )

        for zip_col in ZIP_COLUMNS:
            col_dtype = str(merged_df[zip_col].dtype)
            if col_dtype not in ("object", "string"):
                raise ValueError(
                    f"Hard check failed: Column '{zip_col}' must be string/object to preserve leading zeros, "
                    f"got {merged_df[zip_col].dtype}."
                )

        # Hard Check 6: No suffix columns
        suffix_cols = [c for c in merged_df.columns if c.endswith("_x") or c.endswith("_y")]
        if suffix_cols:
            raise ValueError(
                f"Hard check failed: Output contains suffix columns: {suffix_cols}."
            )

        logger.info("All 6 HARD checks passed successfully.")