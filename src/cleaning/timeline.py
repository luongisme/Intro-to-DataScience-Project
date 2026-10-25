"""Detect adjacent core timestamp reversals at order grain."""

import pandas as pd

from src.eda.checks import build_entity_views, profile_timestamp_ordering
from src.eda.profiling import require_columns


CORE_TIMESTAMP_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
)
ADJACENT_TIMESTAMP_CHECKS = (
    "approved_before_purchase",
    "carrier_before_approval",
    "customer_before_carrier",
)
INVALID_TIMELINE_FLAG = "has_invalid_timeline"


def flag_timeline_issues(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag observed adjacent core timestamp reversals without changing dates.

    Build validated order-grain views, evaluate the three adjacent timestamp
    comparisons and retain missing comparisons as unknown in the summary.
    A missing core timestamp is reported separately and does not by itself set
    the chronology flag. Return a clean item-level copy, the per-check order
    summary and the core-timestamp completeness summary.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    require_columns(df, ["order_id", *CORE_TIMESTAMP_COLUMNS])
    if INVALID_TIMELINE_FLAG in df.columns:
        raise ValueError(f"Input already contains the reserved column '{INVALID_TIMELINE_FLAG}'.")

    orders = build_entity_views(df)["orders"]
    profile = profile_timestamp_ordering(orders)
    conditions = profile["conditions"].loc[:, list(ADJACENT_TIMESTAMP_CHECKS)]
    order_summary = profile["summary"].set_index("check")
    precision_rows = []
    for check in ADJACENT_TIMESTAMP_CHECKS:
        row = order_summary.loc[check]
        flagged = conditions[check].fillna(False)
        start = orders[row["start_column"]]
        end = orders[row["end_column"]]
        same_day = flagged & start.dt.normalize().eq(end.dt.normalize()).fillna(False)
        earlier_date = flagged & end.dt.normalize().lt(start.dt.normalize()).fillna(False)
        precision_rows.append([
            check,
            int(row["orders"]),
            int(row["evaluated_orders"]),
            int(row["unassessable_orders"]),
            int(row["flagged_orders"]),
            int(same_day.sum()),
            int(earlier_date.sum()),
        ])

    timeline_summary = pd.DataFrame(precision_rows, columns=[
        "check", "orders", "comparable_orders", "unknown_orders",
        "affected_orders", "same_day_reversals", "earlier_date_reversals",
    ])
    missing_core = orders.loc[:, list(CORE_TIMESTAMP_COLUMNS)].isna().any(axis=1)
    core_summary = pd.DataFrame([{
        "orders": len(orders),
        "orders_missing_any_core_timestamp": int(missing_core.sum()),
        "orders_with_all_core_timestamps": int((~missing_core).sum()),
    }])

    affected_order_ids = orders.loc[conditions.fillna(False).any(axis=1), "order_id"]
    item_flag = df["order_id"].isin(affected_order_ids).to_numpy(dtype=bool)
    cleaned = df.copy(deep=True)
    cleaned[INVALID_TIMELINE_FLAG] = item_flag
    return cleaned, timeline_summary, core_summary
