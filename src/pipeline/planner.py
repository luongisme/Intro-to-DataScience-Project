"""Resolve the table dependencies of preset Olist merges."""

from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

from src.pipeline.schema import (
    CUSTOMERS_SPEC,
    ORDER_ITEMS_SPEC,
    ORDERS_SPEC,
    PRODUCTS_SPEC,
    SELLERS_SPEC,
    SOURCE_PRIMARY_KEYS,
    JoinSpec,
    TableLoadSpec,
)

# Insertion order defines the canonical table order for preset merges.
MERGE_TABLE_SPECS: Dict[str, TableLoadSpec] = {
    spec.dataset_name: spec
    for spec in (
        ORDER_ITEMS_SPEC,
        ORDERS_SPEC,
        CUSTOMERS_SPEC,
        PRODUCTS_SPEC,
        SELLERS_SPEC,
    )
}

# Each edge connects a foreign-key table to its referenced primary-key table.
# Treat these edges as undirected when finding the required bridge tables.
MERGE_RELATIONSHIPS: Tuple[Tuple[str, str], ...] = (
    (ORDER_ITEMS_SPEC.dataset_name, ORDERS_SPEC.dataset_name),
    (ORDERS_SPEC.dataset_name, CUSTOMERS_SPEC.dataset_name),
    (ORDER_ITEMS_SPEC.dataset_name, PRODUCTS_SPEC.dataset_name),
    (ORDER_ITEMS_SPEC.dataset_name, SELLERS_SPEC.dataset_name),
)


def resolve_bridge_tables(tables: Sequence[str]) -> List[str]:
    """Return selected tables plus the minimum bridges in canonical order.

    Accept two to five distinct registered merge table names. Input order
    does not affect the result. The preset relationship graph is a tree, so
    each pair has exactly one path; their union is the minimum connected set
    containing all selected tables. Do not add unrelated tables or infer
    relationships from shared column names.

    Return a fresh list of names, including both selected and bridge tables.
    The caller can identify bridges by subtracting the selected names. This
    function does not choose JoinSpec options, load data, execute joins or
    mutate the input sequence or registered specifications.

    Raises:
        TypeError: If tables is not a sequence of table names.
        ValueError: If names are invalid, duplicated, unknown, insufficient
            or disconnected in the registered relationship graph.
    """
    if isinstance(tables, (str, bytes)) or not isinstance(tables, Sequence):
        raise TypeError("tables must be a sequence of merge table names.")
    selected_names = list(tables)
    if not all(isinstance(name, str) and name.strip() for name in selected_names):
        raise ValueError("Merge table names must be non-empty strings.")
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("Merge table names must be distinct; duplicate names found.")
    unknown = [name for name in selected_names if name not in MERGE_TABLE_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown merge tables: {unknown}. "
            f"Available tables: {list(MERGE_TABLE_SPECS)}."
        )
    if len(selected_names) < 2:
        raise ValueError("Select at least two distinct merge tables.")

    selected = set(selected_names)
    ordered_selected = [name for name in MERGE_TABLE_SPECS if name in selected]
    adjacency = {name: [] for name in MERGE_TABLE_SPECS}
    for left, right in MERGE_RELATIONSHIPS:
        adjacency[left].append(right)
        adjacency[right].append(left)

    anchor = ordered_selected[0]
    required = selected.copy()
    for target in ordered_selected[1:]:
        parents = {anchor: None}
        pending = deque([anchor])
        while pending and target not in parents:
            current = pending.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in parents:
                    parents[neighbor] = current
                    pending.append(neighbor)
        if target not in parents:
            raise ValueError(
                f"No registered merge path connects '{anchor}' to '{target}'."
            )
        current = target
        while current is not None:
            required.add(current)
            current = parents[current]

    return [name for name in MERGE_TABLE_SPECS if name in required]


def build_combination_plan(tables: Sequence[str]) -> Dict[str, Any]:
    """Build a deterministic, executable configuration for a preset merge.

    Reuse resolve_bridge_tables for name validation and the minimum connected
    table set. Choose order_items as the item-level base whenever it is in
    that set; otherwise the supported orders/customers pair uses orders.
    Join remaining tables in canonical order with left m:1 joins that require
    every left-side key to match. Each registered edge references the primary
    key of its right-hand table; never infer a link from column-name overlap.

    Return requested_tables, bridge_tables, resolved_tables, base, joins and
    grain (order_item or order). All table lists use canonical order. The
    joins value is a fresh list of JoinSpec objects with independent key
    lists. The returned configuration does not load, join or save data, and
    does not imply that any data-level validation has already passed.

    Raises:
        TypeError: If tables is not a sequence of table names.
        ValueError: If table selection or the registered preset topology
            cannot produce a valid sequence of primary-key references.
    """
    resolved = resolve_bridge_tables(tables)
    selected = set(tables)
    base = resolved[0]
    if base not in (ORDER_ITEMS_SPEC.dataset_name, ORDERS_SPEC.dataset_name):
        raise ValueError(f"Unsupported base table for a preset merge: '{base}'.")

    joined = {base}
    joins = []
    for table_name in resolved[1:]:
        sources = [
            source
            for source, target in MERGE_RELATIONSHIPS
            if target == table_name and source in joined
        ]
        if len(sources) != 1:
            raise ValueError(
                f"Preset table '{table_name}' requires exactly one registered "
                "source table already present in the join sequence."
            )
        keys = SOURCE_PRIMARY_KEYS.get(table_name)
        if not keys:
            raise ValueError(f"No primary key is registered for '{table_name}'.")
        source_name = sources[0]
        for name in (source_name, table_name):
            missing_keys = [
                key for key in keys if key not in MERGE_TABLE_SPECS[name].usecols
            ]
            if missing_keys:
                raise ValueError(
                    f"Preset link '{source_name}' -> '{table_name}' requires "
                    f"missing columns in '{name}': {missing_keys}."
                )
        joins.append(
            JoinSpec(
                table=table_name,
                on=keys.copy(),
                how="left",
                validate="m:1",
                require_match=True,
            )
        )
        joined.add(table_name)

    return {
        "requested_tables": [name for name in resolved if name in selected],
        "bridge_tables": [name for name in resolved if name not in selected],
        "resolved_tables": resolved,
        "base": base,
        "joins": joins,
        "grain": "order_item" if base == ORDER_ITEMS_SPEC.dataset_name else "order",
    }
