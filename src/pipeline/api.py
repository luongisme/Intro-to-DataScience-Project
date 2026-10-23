"""Public entry point for preset or explicitly configured table merges."""

from typing import Mapping, Optional, Sequence

import pandas as pd

from src.pipeline.loader import load_table
from src.pipeline.merger import execute_join_plan
from src.pipeline.planner import MERGE_TABLE_SPECS, build_combination_plan
from src.pipeline.schema import JoinSpec, MergeResult, TableSchema, build_source_schema


def merge_data(
    *,
    tables: Optional[Sequence[str]] = None,
    base: Optional[str] = None,
    joins: Optional[Sequence[JoinSpec]] = None,
    source_tables: Optional[Mapping[str, pd.DataFrame]] = None,
    source_schemas: Optional[Mapping[str, TableSchema]] = None,
) -> MergeResult:
    """Merge selected Olist tables or execute an explicitly ordered join plan.

    Choose exactly one configuration mode:
    - tables=[...] builds a canonical preset with minimum bridge tables and
      strict left m:1 joins. Keep bridge columns and report the actual grain.
    - base=..., joins=[JoinSpec(...)] runs exactly that sequence without
      adding bridges or reordering steps. An empty joins list is supported.
      Explicit plans report grain=None rather than guessing a business grain.

    By default, load only the required registered raw tables using their
    canonical contracts. Alternatively, supply both source_tables and
    source_schemas to work entirely in memory, including custom named tables
    in explicit mode. Memory inputs must already satisfy their contracts;
    preset mode additionally requires the canonical registered contracts.
    Ignore unused mapping entries. Do not mix partial memory inputs with
    implicit disk loading, mutate inputs or silently normalize supplied data.

    Delegate preset planning, typed loading and plan execution to the shared
    functions. Return final data/schema plus the plan report, configuration
    mode, requested/resolved/bridge tables, grain and source mode. Reports stay
    in memory. Do not save files, clean rows or use interim output as a cache.

    Examples:
        merge_data(tables=["customers", "products"])
        merge_data(base="order_items", joins=[
            JoinSpec("products", on="product_id",
                     columns=["product_weight_g"])
        ])

    Raises:
        TypeError: If configuration or memory arguments have invalid types.
        ValueError: If modes conflict, configuration/contracts are invalid,
            a required table is unavailable or required links are missing.
        FileNotFoundError: If a required registered raw file is missing.
        pandas.errors.MergeError: If declared join cardinality is violated.
    """
    if tables is not None:
        if base is not None or joins is not None:
            raise ValueError("Choose either tables or base plus joins, not both.")
        plan = build_combination_plan(tables)
        mode = "preset"
    else:
        if base is None or joins is None:
            raise ValueError("Provide tables, or provide both base and joins.")
        if not isinstance(base, str):
            raise TypeError("The merge base must be a table name.")
        if not base.strip():
            raise ValueError("The merge base must be a non-empty table name.")
        if isinstance(joins, (str, bytes)) or not isinstance(joins, Sequence):
            raise TypeError("joins must be a sequence of JoinSpec objects.")
        steps = list(joins)
        for position, join in enumerate(steps, 1):
            if not isinstance(join, JoinSpec):
                raise TypeError(f"Join plan step {position} must be a JoinSpec.")
            if not isinstance(join.table, str) or not join.table.strip():
                raise ValueError(
                    f"Join plan step {position} requires a non-empty table name."
                )
        required = list(dict.fromkeys([base] + [join.table for join in steps]))
        plan = {
            "base": base,
            "joins": steps,
            "requested_tables": required.copy(),
            "bridge_tables": [],
            "resolved_tables": required,
            "grain": None,
        }
        mode = "explicit"

    required = plan["resolved_tables"]
    if (source_tables is None) != (source_schemas is None):
        raise ValueError("Provide both source_tables and source_schemas, or neither.")
    if source_tables is None:
        unknown = [name for name in required if name not in MERGE_TABLE_SPECS]
        if unknown:
            raise ValueError(
                f"Unknown raw merge tables: {unknown}. "
                f"Available tables: {list(MERGE_TABLE_SPECS)}. "
                "For custom tables, provide source_tables plus source_schemas."
            )
        schemas = {
            name: build_source_schema(MERGE_TABLE_SPECS[name]) for name in required
        }
        frames = {name: load_table(MERGE_TABLE_SPECS[name]) for name in required}
        source = "raw"
    else:
        if not isinstance(source_tables, Mapping) or not isinstance(source_schemas, Mapping):
            raise TypeError("source_tables and source_schemas must be mappings.")
        if mode == "preset":
            for name in required:
                if name in source_schemas and source_schemas[name] != build_source_schema(
                    MERGE_TABLE_SPECS[name]
                ):
                    raise ValueError(
                        f"Preset table '{name}' requires its canonical source schema. "
                        "Use base plus joins for custom contracts."
                    )
        frames, schemas = source_tables, source_schemas
        source = "memory"

    result = execute_join_plan(plan["base"], plan["joins"], frames, schemas)
    return MergeResult(
        data=result.data,
        schema=result.schema,
        report={
            **result.report,
            "mode": mode,
            "requested_tables": list(plan["requested_tables"]),
            "bridge_tables": list(plan["bridge_tables"]),
            "resolved_tables": list(required),
            "grain": plan["grain"],
            "source": source,
        },
    )
