"""Display labels and numeric formats for delivery analysis tables."""

from pandas import DataFrame
from pandas.io.formats.style import Styler


_LABELS = {
    "customer_state": "Customer state", "late_share_pct": "Share of scope late orders",
    "approval_slot": "Approval slot", "late_rate_pct": "Late rate",
    "month": "Purchase month", "year": "Purchase year", "stage": "Stage",
    "order_id": "Order ID", "route_type": "Route", "status": "Interval status",
    "is_late": "Delivery outcome", "evaluated": "Valid intervals",
    "unavailable": "Unavailable intervals", "on_time_orders": "On-time orders",
    "order_approved_at": "Approved at", "order_estimated_delivery_date": "Promised date",
    "on_time_calendar_hours": "On time · elapsed (h)", "late_calendar_hours": "Late · elapsed (h)",
    "on_time_eligible_hours": "On time · eligible (h)", "late_eligible_hours": "Late · eligible (h)",
}


def format_analysis_table(table: DataFrame) -> Styler:
    """Format order counts, rates and durations without changing the analytical table."""
    formats = {}
    for column in table:
        if column.endswith("orders") or column in {"evaluated", "unavailable"}:
            formats[column] = "{:,.0f}"
        elif column in {"late_rate_pct", "late_share_pct"}:
            formats[column] = "{:.2f}%"
        elif column.endswith(("_hours", "_days")):
            formats[column] = "{:,.2f}"
        elif column == "is_late":
            formats[column] = {False: "On time", True: "Late"}.__getitem__
    def label(column):
        return _LABELS.get(column, column.replace("_", " ").capitalize())

    return (table.style.format(formats, na_rep="—")
            .format_index(label, axis="columns").hide(axis="index"))
