"""Resolve the table dependencies of preset Olist merges."""

from collections import deque
from typing import Dict, List, Sequence, Tuple

from src.pipeline.schema import (
    CUSTOMERS_SPEC,
    ORDER_ITEMS_SPEC,
    ORDERS_SPEC,
    PRODUCTS_SPEC,
    SELLERS_SPEC,
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
