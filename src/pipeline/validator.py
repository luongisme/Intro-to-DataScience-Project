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

        # Hard Check 7: Orphan checks
        # 7a. order_id vs orders
        items_order_ids: Set[str] = set(base_order_items["order_id"])
        valid_order_ids: Set[str] = set(orders["order_id"])
        orphan_orders = items_order_ids - valid_order_ids
        if orphan_orders:
            raise ValueError(
                f"Hard check failed: {len(orphan_orders)} orphan order_id(s) found in order_items "
                "that do not exist in orders."
            )

        # 7b. product_id vs products
        items_product_ids: Set[str] = set(base_order_items["product_id"])
        valid_product_ids: Set[str] = set(products["product_id"])
        orphan_products = items_product_ids - valid_product_ids
        if orphan_products:
            raise ValueError(
                f"Hard check failed: {len(orphan_products)} orphan product_id(s) found in order_items "
                "that do not exist in products."
            )

        # 7c. seller_id vs sellers
        items_seller_ids: Set[str] = set(base_order_items["seller_id"])
        valid_seller_ids: Set[str] = set(sellers["seller_id"])
        orphan_sellers = items_seller_ids - valid_seller_ids
        if orphan_sellers:
            raise ValueError(
                f"Hard check failed: {len(orphan_sellers)} orphan seller_id(s) found in order_items "
                "that do not exist in sellers."
            )

        logger.info("All 7 HARD checks passed successfully.")

    @staticmethod
    def run_soft_checks(merged_df: pd.DataFrame) -> None:
        """Run soft quality checks and log warnings. Does not raise exceptions."""
        total_rows = len(merged_df)

        # 1. Row count benchmark check (~112,650 +- 1%)
        pct_diff = abs(total_rows - BENCHMARK_ROW_COUNT) / BENCHMARK_ROW_COUNT * 100
        if pct_diff > 1.0:
            logger.warning(
                "SOFT CHECK WARNING: Total rows (%d) differs from benchmark (%d) by %.2f%% (> 1%%).",
                total_rows,
                BENCHMARK_ROW_COUNT,
                pct_diff,
            )
        else:
            logger.info("Soft check: Row count %d within 1%% of benchmark.", total_rows)

        # 2. Distinct order count
        distinct_orders = merged_df["order_id"].nunique()
        logger.info("Soft check: Distinct order count: %d", distinct_orders)

        # 3. Null counts report table
        null_counts = merged_df.isna().sum()
        null_report_lines = [
            f"{col:32s} | Null Count: {count:7d} | Null Pct: {count / total_rows * 100:6.2f}%"
            for col, count in null_counts.items()
        ]
        logger.info(
            "=== NULL VALUE REPORT (ALL 25 COLUMNS) ===\n%s",
            "\n".join(null_report_lines),
        )

        # 4. order_status distribution
        status_counts = merged_df["order_status"].value_counts(dropna=False).to_dict()
        logger.info("Soft check: order_status distribution: %s", status_counts)

        # 5. Delivered status but null delivery date anomaly
        delivered_missing_date = int(
            (
                (merged_df["order_status"] == "delivered")
                & (merged_df["order_delivered_customer_date"].isna())
            ).sum()
        )
        if delivered_missing_date > 0:
            logger.warning(
                "SOFT CHECK WARNING: Found %d rows where order_status == 'delivered' "
                "but order_delivered_customer_date is NaT.",
                delivered_missing_date,
            )
        else:
            logger.info("Soft check: 0 delivered orders have missing customer delivery dates.")