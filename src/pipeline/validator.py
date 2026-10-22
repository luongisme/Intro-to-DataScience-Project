"""Runtime validation layer: Hard checks (raise) and Soft checks (warnings)."""

import logging
from typing import Optional, Sequence, Set
import pandas as pd

from src.pipeline.schema import (
    ALL_TIMESTAMP_COLUMNS,
    BENCHMARK_ROW_COUNT,
    EXPECTED_FINAL_COLUMNS,
    FULL_MERGE_SCHEMA,
    SOURCE_PRIMARY_KEYS,
    ZIP_COLUMNS,
    TableSchema,
)

logger = logging.getLogger(__name__)


def validate_table_schema(
    df: pd.DataFrame,
    schema: TableSchema,
    *,
    table_name: str = "dataset",
) -> None:
    """Check exact column structure and dtypes without modifying the table.

    Require unique column names, complete dtype declarations, exact column
    order and canonical pandas dtypes, including datetime precision. Empty
    tables are valid when their columns and dtypes satisfy the schema.
    Row uniqueness, missing values and foreign keys are checked separately.

    Args:
        df: Table to inspect without conversion or mutation.
        schema: Expected ordered columns and pandas dtypes.
        table_name: Table label included in error messages.

    Raises:
        TypeError: If df or schema has an unexpected type.
        ValueError: If the schema declarations or table structure are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Table '{table_name}' must be a pandas DataFrame.")
    if not isinstance(schema, TableSchema):
        raise TypeError(f"Schema for '{table_name}' must be a TableSchema.")

    expected_columns = list(schema.columns)
    if not all(isinstance(column, str) for column in expected_columns):
        raise ValueError(f"Schema for '{table_name}' requires string column names.")
    if len(expected_columns) != len(set(expected_columns)):
        raise ValueError(f"Schema for '{table_name}' declares duplicate columns.")
    missing_dtypes = [col for col in expected_columns if col not in schema.dtypes]
    extra_dtypes = [col for col in schema.dtypes if col not in expected_columns]
    if missing_dtypes or extra_dtypes:
        raise ValueError(
            f"Schema for '{table_name}' has inconsistent dtype declarations: "
            f"missing={missing_dtypes}, extra={extra_dtypes}."
        )
    if df.columns.has_duplicates:
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"Table '{table_name}' has duplicate column names: {duplicate_columns}."
        )

    actual_columns = list(df.columns)
    missing_columns = [col for col in expected_columns if col not in df.columns]
    extra_columns = [col for col in actual_columns if col not in expected_columns]
    if missing_columns or extra_columns:
        raise ValueError(
            f"Table '{table_name}' has a column mismatch: "
            f"missing={missing_columns}, extra={extra_columns}."
        )
    if actual_columns != expected_columns:
        raise ValueError(
            f"Table '{table_name}' has an incorrect column order. "
            f"Expected: {expected_columns}. Actual: {actual_columns}."
        )

    mismatches = []
    for column in expected_columns:
        declared_dtype = schema.dtypes[column]
        try:
            expected_dtype = pd.api.types.pandas_dtype(declared_dtype)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Schema for '{table_name}' declares an invalid dtype for "
                f"column '{column}': {declared_dtype!r}."
            ) from err
        actual_dtype = df[column].dtype
        if not pd.api.types.is_dtype_equal(actual_dtype, expected_dtype):
            mismatches.append(
                f"{column}: expected {expected_dtype}, got {actual_dtype}"
            )
    if mismatches:
        raise ValueError(
            f"Table '{table_name}' has dtype mismatches: " + "; ".join(mismatches)
        )


def validate_source_keys(
    df: pd.DataFrame,
    table_name: str,
    *,
    key_columns: Optional[Sequence[str]] = None,
) -> None:
    """Check that a table has a complete, unique primary key without mutation.

    Use SOURCE_PRIMARY_KEYS unless explicit key columns are supplied. A key
    with multiple columns is checked as one composite value, not as separate
    unique columns. Null values and blank strings are invalid key components.
    Empty tables are valid if the declared key columns exist.

    Args:
        df: Source table or custom table to inspect.
        table_name: Registered source name or label for an explicit key.
        key_columns: Optional key override, such as a list or tuple of names.
            Custom tables must supply this instead of guessing a key.

    Raises:
        TypeError: If df or table_name has an unexpected type.
        ValueError: If key declarations, columns or values are invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Table '{table_name}' must be a pandas DataFrame.")
    if not isinstance(table_name, str):
        raise TypeError("table_name must be a string.")
    if key_columns is None:
        if table_name not in SOURCE_PRIMARY_KEYS:
            raise ValueError(
                f"No primary key is registered for table '{table_name}'. "
                "Supply key_columns explicitly for a custom table."
            )
        declared_keys = SOURCE_PRIMARY_KEYS[table_name]
    else:
        declared_keys = key_columns
    if isinstance(declared_keys, str) or not isinstance(declared_keys, Sequence):
        raise ValueError(
            f"Primary key for '{table_name}' must be a non-empty sequence "
            "of column names."
        )
    keys = list(declared_keys)
    if not keys or not all(isinstance(key, str) and key.strip() for key in keys):
        raise ValueError(
            f"Primary key for '{table_name}' requires non-empty string column names."
        )
    if len(keys) != len(set(keys)):
        raise ValueError(f"Primary key for '{table_name}' repeats a column name.")
    if df.columns.has_duplicates:
        raise ValueError(f"Table '{table_name}' has duplicate column names.")
    missing_columns = [key for key in keys if key not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Table '{table_name}' is missing primary key columns: {missing_columns}."
        )

    missing_rows = int(df[keys].isna().any(axis=1).sum())
    if missing_rows:
        raise ValueError(
            f"Primary key {keys} in table '{table_name}' has missing values "
            f"in {missing_rows} row(s)."
        )
    blank_counts = {}
    for key in keys:
        count = sum(
            isinstance(value, str) and not value.strip()
            for value in df[key]
        )
        if count:
            blank_counts[key] = count
    if blank_counts:
        raise ValueError(
            f"Primary key in table '{table_name}' has blank string values: "
            f"{blank_counts}."
        )
    duplicate_count = int(df.duplicated(subset=keys).sum())
    if duplicate_count:
        raise ValueError(
            f"Primary key {keys} in table '{table_name}' is not unique: "
            f"{duplicate_count} repeated row(s) after the first occurrence."
        )


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
        2. The declared item-level primary key is complete and unique
        3. Columns equal exactly 25 expected columns in order
        4. No nulls in the remaining mandatory identifiers
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

        # Hard Check 2: Validate the declared item-level primary key
        primary_key = FULL_MERGE_SCHEMA.unique_key
        validate_source_keys(
            merged_df,
            table_name="merged_order_items",
            key_columns=primary_key,
        )

        # Hard Check 3: Exact 25 columns and order
        actual_cols = list(merged_df.columns)
        if actual_cols != EXPECTED_FINAL_COLUMNS:
            raise ValueError(
                f"Hard check failed: Column structure mismatch!\n"
                f"Expected ({len(EXPECTED_FINAL_COLUMNS)}): {EXPECTED_FINAL_COLUMNS}\n"
                f"Actual   ({len(actual_cols)}): {actual_cols}"
            )

        # Hard Check 4: Validate IDs not already covered by the primary key
        for col in FULL_MERGE_SCHEMA.non_nullable_columns:
            if col in primary_key:
                continue
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
            if not pd.api.types.is_string_dtype(merged_df[zip_col]):
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