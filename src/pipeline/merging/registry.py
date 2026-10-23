"""Build named storage contracts for the standard Olist merge combinations."""

from itertools import combinations
from typing import Any, Dict

from pipeline.merging.planner import MERGE_TABLE_SPECS, build_combination_plan
from pipeline.contracts.schema import build_source_schema
from pipeline.contracts.schema_derivation import derive_join_schema


FULL_DATASET_NAME = "merged_order_items"


def build_dataset_registry() -> Dict[str, Dict[str, Any]]:
    """Return fresh dataset configurations for all 26 standard combinations.

    Enumerate two through five requested tables in MERGE_TABLE_SPECS order.
    Reuse build_combination_plan for bridge resolution, base, joins and grain;
    derive each output contract from the registered source schemas by applying
    derive_join_schema in that exact join order. Do not duplicate column/key
    declarations, join logic or observed row-count benchmarks here.

    Name non-full combinations 'merged_' plus requested table names separated
    by double underscores, for example 'merged_orders__customers'. Requested
    tables determine identity, even when different requests resolve to the
    same table set/schema through bridges. Keep the legacy full filename stem
    FULL_DATASET_NAME, without an additional alias for the five-table request.

    Map each extension-free name to requested_tables, bridge_tables,
    resolved_tables, base, joins, grain and schema. Lists/dicts/schema contents
    are independent between entries and calls; consumers can use a returned
    schema without mutating the shared source declarations or another entry.
    Registry order is combination size, then canonical source order.

    Build declarations only: do not load/merge/save data, inspect files, infer
    whether data-level validation passed, create sidecars/reports or change
    the legacy loader registry. Custom joins remain outside this catalog and
    use their explicit result schema with the dataset I/O functions.

    Returns:
        Dataset filename stems mapped to fresh plans and TableSchema objects.

    Raises:
        TypeError: If shared source/schema declarations have invalid types.
        ValueError: If the shared preset graph or contracts are inconsistent.
    """
    sources = tuple(MERGE_TABLE_SPECS)
    source_schemas = {
        name: build_source_schema(spec) for name, spec in MERGE_TABLE_SPECS.items()
    }
    registry = {}
    for size in range(2, len(sources) + 1):
        for selected in combinations(sources, size):
            plan = build_combination_plan(selected)
            schema = source_schemas[plan["base"]]
            for join in plan["joins"]:
                schema = derive_join_schema(schema, source_schemas[join.table], join)
            dataset_name = (
                FULL_DATASET_NAME if size == len(sources)
                else "merged_" + "__".join(plan["requested_tables"])
            )
            registry[dataset_name] = {**plan, "schema": schema}
    return registry
