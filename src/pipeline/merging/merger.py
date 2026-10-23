"""Merge logic for combining Olist tables into a single item-level DataFrame."""

from dataclasses import replace
from typing import List, Mapping, Sequence
import pandas as pd

from pipeline.contracts.normalizer import normalize_table_types
from src.pipeline.planner import MERGE_TABLE_SPECS, build_combination_plan
from pipeline.contracts.schema import (
    EXPECTED_FINAL_COLUMNS,
    JoinSpec,
    MergeResult,
    TableSchema,
    build_source_schema,
)
from pipeline.contracts.schema_derivation import derive_join_schema
from pipeline.contracts.validator import (
    validate_foreign_keys,
    validate_table_contract,
)


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
        validate_table_contract(frame, schema, table_name=label)
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
    if len(merged.columns) != len(output_schema.columns) or set(merged.columns) != set(
        output_schema.columns
    ):
        raise ValueError(f"Join '{join.table}' produced unexpected output columns.")
    result = normalize_table_types(
        merged.loc[:, output_schema.columns], output_schema, table_name="join_result"
    )
    validate_table_contract(
        result,
        output_schema,
        table_name="join_result",
        expected_rows=expected_rows,
    )
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


def execute_join_plan(
    base: str,
    joins: Sequence[JoinSpec],
    tables: Mapping[str, pd.DataFrame],
    schemas: Mapping[str, TableSchema],
) -> MergeResult:
    """Execute an ordered join plan using named canonical tables and schemas.

    Inspect only the base and referenced right tables. Validate their contracts
    and preflight the entire schema sequence before performing the first join.
    Then delegate every step to execute_join; carry each resulting table and
    schema into the next step without reimplementing joining or validation.
    Preserve explicit join order, including repeated use of a named source
    when its selected/renamed columns form a valid sequence.

    Return the final data/schema plus source shapes, join order and per-step
    reports in memory. An empty plan returns independent copies of the checked
    base data and schema. Do not infer global row preservation: each step
    enforces only its own guarantees. Input tables/configuration are unchanged.
    Do not load/save data, reorder the plan, select bridge tables or clean rows.

    Args:
        base: Name of the initial table in both input mappings.
        joins: JoinSpec sequence, executed exactly in the supplied order.
        tables: Already loaded canonical DataFrames indexed by source name.
        schemas: Corresponding TableSchema contracts; extra entries are ignored.

    Raises:
        TypeError: If arguments or plan steps have unexpected types.
        ValueError: If inputs, schema sequence or table constraints are invalid.
        pandas.errors.MergeError: If a step violates its declared cardinality.
    """
    if not isinstance(base, str):
        raise TypeError("The join-plan base must be a table name.")
    if not base.strip():
        raise ValueError("The join-plan base must be a non-empty table name.")
    if isinstance(joins, (str, bytes)) or not isinstance(joins, Sequence):
        raise TypeError("joins must be a sequence of JoinSpec objects.")
    if not isinstance(tables, Mapping) or not isinstance(schemas, Mapping):
        raise TypeError("tables and schemas must be mappings indexed by source name.")
    steps = list(joins)
    for position, join in enumerate(steps, 1):
        if not isinstance(join, JoinSpec):
            raise TypeError(f"Join plan step {position} must be a JoinSpec.")
        if not isinstance(join.table, str) or not join.table.strip():
            raise ValueError(f"Join plan step {position} requires a non-empty table name.")
    required = list(dict.fromkeys([base] + [join.table for join in steps]))
    missing_tables = [name for name in required if name not in tables]
    missing_schemas = [name for name in required if name not in schemas]
    if missing_tables or missing_schemas:
        raise ValueError(
            f"Missing join-plan inputs: tables={missing_tables}, schemas={missing_schemas}."
        )
    for name in required:
        validate_table_contract(tables[name], schemas[name], table_name=name)
    planned_schema = schemas[base]
    for position, join in enumerate(steps, 1):
        try:
            planned_schema = derive_join_schema(planned_schema, schemas[join.table], join)
        except (TypeError, ValueError) as err:
            raise type(err)(
                f"Join plan step {position} ('{join.table}') has an invalid schema: {err}"
            ) from err

    current = tables[base].copy(deep=True)
    source_schema = schemas[base]
    current_schema = TableSchema(
        columns=list(source_schema.columns),
        dtypes=dict(source_schema.dtypes),
        unique_key=list(source_schema.unique_key),
        non_nullable_columns=list(source_schema.non_nullable_columns),
    )
    reports = []
    for position, join in enumerate(steps, 1):
        try:
            result = execute_join(
                current,
                tables[join.table],
                current_schema,
                schemas[join.table],
                join,
                left_table_name=base,
            )
        except (TypeError, ValueError) as err:
            raise type(err)(f"Join plan step {position} ('{join.table}') failed: {err}") from err
        current, current_schema = result.data, result.schema
        reports.append(result.report)
    if current_schema != planned_schema:
        raise ValueError("Executed join-plan schema differs from its preflight contract.")
    validate_table_contract(current, current_schema, table_name="plan_result")
    return MergeResult(
        data=current,
        schema=current_schema,
        report={
            "base": base,
            "tables": required,
            "join_order": [join.table for join in steps],
            "input_shapes": {
                name: {"rows": len(tables[name]), "columns": len(tables[name].columns)}
                for name in required
            },
            "output_rows": len(current),
            "output_columns": len(current.columns),
            "steps": reports,
            "validation": {
                "source_contracts": True,
                "plan_schema": True,
                "output_contract": True,
            },
        },
    )


class OlistDataMerger:
    """Executes sequential many-to-one left joins starting from order_items."""

    def __init__(self) -> None:
        self.expected_columns: List[str] = EXPECTED_FINAL_COLUMNS

    def merge_tables(
        self,
        order_items: pd.DataFrame,
        orders: pd.DataFrame,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        sellers: pd.DataFrame,
    ) -> pd.DataFrame:
        """Adapt the five-table legacy API to the shared join-plan executor.

        Preserve the positional/keyword arguments, DataFrame return type,
        configured output-column order and non-mutating behavior. Normalize
        supplied tables with their registered contracts so existing callers
        can pass pandas-inferred strings, timestamp precision or column order.
        Retain unexpected columns for explicit contract errors. Conversion
        must remain lossless; never fill attributes, filter rows or load files.

        Use the registered full preset for table order and join keys, but
        retain the legacy left-join policy: unmatched items remain available
        for the caller's subsequent validation. The public preset API keeps
        its strict require_match=True policy. Preserve MergeError for repeated
        right keys before source-contract validation would raise ValueError.

        Delegate all joining, null-key handling, schema derivation and result
        checks to execute_join_plan; do not keep a second merge algorithm.
        Additional source-contract violations now fail explicitly rather than
        passing unchecked. This adapter does not save data or reports.

        Raises:
            TypeError: If an input is not a DataFrame.
            ValueError: If conversion, source or result contracts are invalid.
            pandas.errors.MergeError: If right-side join keys are duplicated.
        """
        sources = {
            "order_items": order_items,
            "orders": orders,
            "customers": customers,
            "products": products,
            "sellers": sellers,
        }
        plan = build_combination_plan(list(sources))
        schemas = {
            name: build_source_schema(MERGE_TABLE_SPECS[name]) for name in sources
        }
        frames = {}
        for name, frame in sources.items():
            normalized = normalize_table_types(frame, schemas[name], table_name=name)
            columns = schemas[name].columns + [
                column for column in normalized.columns
                if column not in schemas[name].columns
            ]
            frames[name] = normalized.loc[:, columns]
        for join in plan["joins"]:
            keys = schemas[join.table].unique_key
            if frames[join.table].duplicated(keys).any():
                raise pd.errors.MergeError(
                    f"Join '{join.table}' violates cardinality 'm:1': "
                    f"right keys {keys} are not unique."
                )
        result = execute_join_plan(
            plan["base"],
            [replace(join, require_match=False) for join in plan["joins"]],
            frames,
            schemas,
        )
        return result.data.loc[:, self.expected_columns].copy(deep=True)
