"""Descriptive delivery statistics by approval time slot."""

import numpy as np
import pandas as pd

from .orders import SLOT_LABELS


def summarize_delivery_by_slot(orders: pd.DataFrame) -> pd.DataFrame:
    """Summarize one row per order in all four approval-time slots.

    Late rate uses all orders in the slot. Delay statistics use late orders
    because the order view leaves delay_days missing for on-time deliveries.
    Empty slots retain zero counts and undefined rates and delays.
    """
    required = {"order_id", "approval_slot", "is_late", "delay_days"}
    missing_columns = required.difference(orders.columns)
    if missing_columns:
        raise ValueError(f"Missing summary columns: {sorted(missing_columns)}")
    if orders["order_id"].duplicated().any():
        raise ValueError("Expected one row per order_id before slot aggregation.")

    summary = orders.groupby("approval_slot", observed=False).agg(
        orders=("order_id", "size"),
        late_orders=("is_late", "sum"),
        mean_delay_days=("delay_days", "mean"),
        median_delay_days=("delay_days", "median"),
    ).reindex(SLOT_LABELS)
    summary["orders"] = summary["orders"].fillna(0).astype("int64")
    summary["late_orders"] = summary["late_orders"].fillna(0).astype("int64")
    summary["on_time_orders"] = summary["orders"] - summary["late_orders"]
    summary["late_rate_pct"] = np.divide(
        100 * summary["late_orders"],
        summary["orders"],
        out=np.full(len(summary), np.nan),
        where=summary["orders"].ne(0),
    )
    return summary.reset_index()[
        ["approval_slot", "orders", "on_time_orders", "late_orders",
         "late_rate_pct", "mean_delay_days", "median_delay_days"]
    ]
