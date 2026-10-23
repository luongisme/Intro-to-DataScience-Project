"""Merge logic for combining Olist tables into a single item-level DataFrame."""

import logging
from typing import List, Set
import pandas as pd

from src.pipeline.normalizer import normalize_table_types
from src.pipeline.schema import EXPECTED_FINAL_COLUMNS, JoinSpec, MergeResult, TableSchema
from src.pipeline.schema_derivation import derive_join_schema
from src.pipeline.validator import (
    validate_foreign_keys,
    validate_source_keys,
    validate_table_schema,
)

logger = logging.getLogger(__name__)


def execute_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_schema: TableSchema,
    right_schema: TableSchema,
    join: JoinSpec,
    *,
    left_table_name: str = "left",
) -> MergeResult:
    """Execute one configured join on canonical input tables without mutation.

    Derive the output contract, validate both inputs and enforce foreign-key
    membership when required. Let pandas enforce the declared cardinality,
    including explicit m:m joins. Retain selected right columns in source
    order, always keep join keys and rename only configured non-key columns.

    Null or blank key components never match, even on both sides. Preserve
    those rows as unmatched when the join type requires it. A temporary guard
    key prevents pandas from matching invalid keys; it never reaches output.
    Convert numpy integers/booleans to nullable types before merging, including
    join keys, so missing-side rows cannot round large integers through float.
    Normalize the result back to its derived contract and verify constraints.

    Return data, schema and a serializable in-memory join report. Row-count
    preservation is checked only when guaranteed by how, validate and match
    policy. Column-based merging returns a fresh index as in pandas.merge.
    Do not load/save files, filter statuses, fill attributes or remove duplicates.

    Raises:
        TypeError: If inputs or configuration have invalid types.
        ValueError: If schemas, constraints or required links are invalid.
        pandas.errors.MergeError: If join cardinality is violated.
    """
    output_schema = derive_join_schema(left_schema, right_schema, join)
    for frame, schema, label in (
        (left_df, left_schema, left_table_name),
        (right_df, right_schema, join.table),
    ):
        validate_table_schema(frame, schema, table_name=label)
        if schema.unique_key:
            validate_source_keys(frame, label, key_columns=schema.unique_key)
        missing = {
            name: int(frame[name].isna().sum())
            for name in schema.non_nullable_columns
            if frame[name].isna().any()
        }
        if missing:
            raise ValueError(f"Table '{label}' violates non-null constraints: {missing}.")
    matching = validate_foreign_keys(
        left_df, right_df, join, left_table_name=left_table_name
    )
    declared_left = join.on if join.on is not None else join.left_on
    declared_right = join.on if join.on is not None else join.right_on
    left_keys = [declared_left] if isinstance(declared_left, str) else list(declared_left)
    right_keys = [declared_right] if isinstance(declared_right, str) else list(declared_right)
    included = set(right_schema.columns if join.columns is None else join.columns)
    included.update(right_keys)
    right_columns = [name for name in right_schema.columns if name in included]
    left_work = left_df.copy(deep=True).reset_index(drop=True)
    right_work = right_df.loc[:, right_columns].copy(deep=True).reset_index(drop=True)
    for frame in (left_work, right_work):
        for name in frame.columns:
            dtype = frame[name].dtype
            if isinstance(dtype, pd.api.extensions.ExtensionDtype):
                continue
            if pd.api.types.is_integer_dtype(dtype):
                prefix = "UInt" if pd.api.types.is_unsigned_integer_dtype(dtype) else "Int"
                frame[name] = frame[name].astype(f"{prefix}{dtype.itemsize * 8}")
            elif pd.api.types.is_bool_dtype(dtype):
                frame[name] = frame[name].astype("boolean")

    guard = "__merge_validity__"
    reserved = set(left_work.columns) | set(right_work.columns) | set(output_schema.columns)
    while guard in reserved:
        guard += "_"
    invalid_counts = []
    for frame, keys, marker in ((left_work, left_keys, 1), (right_work, right_keys, 2)):
        invalid = frame[keys].isna().any(axis=1).to_numpy(dtype=bool, copy=True)
        for key in keys:
            invalid |= frame[key].map(
                lambda value: isinstance(value, str) and not value.strip()
            ).to_numpy(dtype=bool)
        frame[guard] = invalid.astype("int8") * marker
        invalid_counts.append(int(invalid.sum()))
    right_work = right_work.rename(columns=join.rename)
    try:
        merged = left_work.merge(
            right_work,
            left_on=left_keys + [guard],
            right_on=right_keys + [guard],
            how=join.how,
            validate=join.validate,
            suffixes=(None, None),
            sort=False,
        ).drop(columns=[guard])
    except pd.errors.MergeError as err:
        raise pd.errors.MergeError(
            f"Join '{left_table_name}' -> '{join.table}' violates "
            f"cardinality {join.validate!r}: {err}"
        ) from err

    expected_rows = None
    if join.validate in ("1:1", "m:1") and (
        join.how == "left" or (join.how == "inner" and join.require_match)
    ):
        expected_rows = len(left_df)
    elif join.validate in ("1:1", "1:m") and (
        join.how == "right" or (join.how == "outer" and join.require_match)
    ):
        expected_rows = len(right_df)
    if expected_rows is not None and len(merged) != expected_rows:
        raise ValueError(
            f"Join '{join.table}' changed a guaranteed row count: "
            f"expected {expected_rows}, got {len(merged)}."
        )
    if len(merged.columns) != len(output_schema.columns) or set(merged.columns) != set(
        output_schema.columns
    ):
        raise ValueError(f"Join '{join.table}' produced unexpected output columns.")
    result = normalize_table_types(
        merged.loc[:, output_schema.columns], output_schema, table_name="join_result"
    )
    validate_table_schema(result, output_schema, table_name="join_result")
    if output_schema.unique_key:
        validate_source_keys(result, "join_result", key_columns=output_schema.unique_key)
    missing = {
        name: int(result[name].isna().sum())
        for name in output_schema.non_nullable_columns
        if result[name].isna().any()
    }
    if missing:
        raise ValueError(f"Join output violates non-null constraints: {missing}.")
    return MergeResult(
        data=result,
        schema=output_schema,
        report={
            "left_table": left_table_name,
            "right_table": join.table,
            "left_on": left_keys,
            "right_on": right_keys,
            "how": join.how,
            "cardinality": join.validate,
            "require_match": join.require_match,
            "selected_right_columns": right_columns,
            "rename": dict(join.rename),
            "left_rows": len(left_df),
            "right_rows": len(right_df),
            "output_rows": len(result),
            "output_columns": len(result.columns),
            "left_duplicate_key_rows": int(left_df.duplicated(left_keys).sum()),
            "right_duplicate_key_rows": int(right_df.duplicated(right_keys).sum()),
            "left_invalid_key_rows": invalid_counts[0],
            "right_invalid_key_rows": invalid_counts[1],
            "matching": matching,
            "expected_rows": expected_rows,
            "validation": {
                "input_schemas": True,
                "input_constraints": True,
                "cardinality": True,
                "output_schema": True,
                "output_constraints": True,
                "row_count_checked": expected_rows is not None,
            },
        },
    )


class OlistDataMerger:
    """Executes sequential many-to-one left joins starting from order_items."""

    def __init__(self) -> None:
        self.expected_columns: List[str] = EXPECTED_FINAL_COLUMNS

    @staticmethod
    def _safe_left_join(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        on: str,
        table_name: str,
    ) -> pd.DataFrame:
        """Perform a validated many-to-one left join without column collision.

        Args:
            left_df: Base DataFrame.
            right_df: Dimension table to join.
            on: Common key column.
            table_name: Name of the right table for error logging.

        Returns:
            pd.DataFrame: Merged DataFrame.

        Raises:
            ValueError: If non-key column names collide or if right table contains duplicate keys.
        """
        # Guard against suffix collisions (_x, _y)
        overlapping_cols: Set[str] = (set(left_df.columns) & set(right_df.columns)) - {on}
        if overlapping_cols:
            raise ValueError(
                f"Column name collision detected when joining '{table_name}' on key '{on}'. "
                f"Colliding columns: {sorted(overlapping_cols)}. "
                "Adding suffixes (_x, _y) is strictly forbidden."
            )

        merged = left_df.merge(right_df, on=on, how="left", validate="m:1")
        return merged

    def merge_tables(
        self,
        order_items: pd.DataFrame,
        orders: pd.DataFrame,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        sellers: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all 5 Olist tables strictly in left-join order.

        Join sequence:
        1. order_items LEFT JOIN orders on order_id
        2. + LEFT JOIN customers on customer_id
        3. + LEFT JOIN products on product_id
        4. + LEFT JOIN sellers on seller_id

        Args:
            order_items: Base item table.
            orders: Orders table.
            customers: Customers table.
            products: Products table.
            sellers: Sellers table.

        Returns:
            pd.DataFrame: New merged DataFrame matching EXPECTED_FINAL_COLUMNS.
        """
        # Immutability guarantee: clone input base table
        current_df = order_items.copy(deep=True)
        initial_rows = len(current_df)
        logger.info("Starting merge pipeline. Base table rows: %d", initial_rows)

        # 1. Join orders on order_id
        current_df = self._safe_left_join(
            left_df=current_df,
            right_df=orders,
            on="order_id",
            table_name="orders",
        )
        logger.info("Joined 'orders'. Current shape: %s", current_df.shape)

        # 2. Join customers on customer_id
        current_df = self._safe_left_join(
            left_df=current_df,
            right_df=customers,
            on="customer_id",
            table_name="customers",
        )
        logger.info("Joined 'customers'. Current shape: %s", current_df.shape)

        # 3. Join products on product_id
        current_df = self._safe_left_join(
            left_df=current_df,
            right_df=products,
            on="product_id",
            table_name="products",
        )
        logger.info("Joined 'products'. Current shape: %s", current_df.shape)

        # 4. Join sellers on seller_id
        current_df = self._safe_left_join(
            left_df=current_df,
            right_df=sellers,
            on="seller_id",
            table_name="sellers",
        )
        logger.info("Joined 'sellers'. Current shape: %s", current_df.shape)

        # Ensure column ordering matches specification exactly
        result = current_df[self.expected_columns].copy(deep=True)
        logger.info("Merge completed successfully. Output shape: %s", result.shape)
        return result