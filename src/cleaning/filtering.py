"""Exclude complete orders using the agreed historical-data eligibility rules."""

import pandas as pd

from src.cleaning.measurements import INVALID_PHYSICAL_FLAG
from src.cleaning.schema import (
    FLAGGED_ORDER_ITEMS_SCHEMA, QUALITY_FLAG_COLUMNS, REQUIRED_TIMESTAMPS,
)
from src.cleaning.timeline import (
    DEADLINE_REVIEW_FLAG, INVALID_TIMELINE_FLAG,
)
from src.pipeline.contracts.validator import validate_table_contract


EXCLUSION_REASONS = (
    "not_delivered",
    "missing_required_timestamp",
    INVALID_TIMELINE_FLAG,
    INVALID_PHYSICAL_FLAG,
    DEADLINE_REVIEW_FLAG,
)


def filter_eligible_orders(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Filter a schema-valid inspected item table by whole-order eligibility.

    Require the 28-column result with quality flags already computed on the full
    input. Exclude an entire order if any of its items is not delivered, lacks
    a core/estimated-delivery timestamp, or has any of the three quality flags.
    A missing status also fails the delivered requirement. Optional missing
    attributes remain allowed. Do not recalculate flags on a filtered subset.

    Return an independent retained table with the same columns, dtypes, index
    labels and relative row order, plus three in-memory diagnostic DataFrames:
    reasons counts every affected order and all its item rows per reason;
    flow counts newly excluded orders/items in EXCLUSION_REASONS order;
    excluded_orders lists removed IDs with item counts and every reason flag.
    Reason counts can overlap. Flow counts are disjoint, and its final row gives
    the retained totals. Empty input or an empty retained result is supported.
    Column selection, saving and display are explicit caller responsibilities.
    """
    validate_table_contract(df, FLAGGED_ORDER_ITEMS_SCHEMA, table_name="inspected_order_items")
    item_rules = df[["order_id", *QUALITY_FLAG_COLUMNS]].copy(deep=True)
    item_rules["not_delivered"] = df["order_status"].ne("delivered").fillna(True).to_numpy(dtype=bool)
    item_rules["missing_required_timestamp"] = df[list(REQUIRED_TIMESTAMPS)].isna().any(axis=1).to_numpy(dtype=bool)
    groups = item_rules.groupby("order_id", sort=False, observed=True, dropna=False)
    order_rules = groups[list(EXCLUSION_REASONS)].any()
    order_rules.insert(0, "item_rows", groups.size())

    reason_rows = []
    for reason in EXCLUSION_REASONS:
        affected = order_rules[reason]
        reason_rows.append([
            reason, int(affected.sum()), int(order_rules.loc[affected, "item_rows"].sum()),
        ])
    reasons = pd.DataFrame(reason_rows, columns=[
        "reason", "affected_orders", "affected_item_rows",
    ])

    remaining = pd.Series(True, index=order_rules.index, dtype=bool)
    flow_rows = [["input", 0, 0, len(order_rules), len(df)]]
    for reason in EXCLUSION_REASONS:
        newly_excluded = remaining & order_rules[reason]
        removed_orders = int(newly_excluded.sum())
        removed_items = int(order_rules.loc[newly_excluded, "item_rows"].sum())
        remaining = remaining & ~order_rules[reason]
        flow_rows.append([
            reason, removed_orders, removed_items, int(remaining.sum()),
            int(order_rules.loc[remaining, "item_rows"].sum()),
        ])
    flow = pd.DataFrame(flow_rows, columns=[
        "step", "removed_orders", "removed_item_rows", "remaining_orders",
        "remaining_item_rows",
    ])
    excluded_orders = order_rules.loc[~remaining].reset_index()
    retained_ids = order_rules.index[remaining]
    retained = df.loc[df["order_id"].isin(retained_ids)].copy(deep=True)
    return retained, {
        "reasons": reasons,
        "flow": flow,
        "excluded_orders": excluded_orders,
    }
