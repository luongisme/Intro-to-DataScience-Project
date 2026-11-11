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
    "raw_late_rate_pct": "Full-cohort raw late rate",
    "standardized_late_rate_pct": "Standardized late rate",
    "difference_pp_vs_12_18": "Difference from 12:00–18:00 (pp)",
    "common_months": "Common purchase months",
    "valid_intervals": "Valid stage intervals",
    "unavailable_intervals": "Unavailable stage intervals",
    "inter_state_orders": "Inter-state orders",
    "inter_state_unknown_destination": "Inter-state orders with unknown destination",
    "inter_state_to_destination": "Inter-state orders to destination",
    "selected_pair": "Selected pair orders",
    "other_to_destination": "Other inter-state orders to destination",
    "median_calendar_hours": "Median elapsed time (h)",
    "p75_calendar_hours": "P75 elapsed time (h)",
    "median_eligible_hours": "Median eligible-day time (h)",
    "p75_eligible_hours": "P75 eligible-day time (h)",
    "median_processing_hours": "Median approval-to-handoff time (h)",
    "median_transit_hours": "Median handoff-to-delivery time (h)",
    "median_promised_days": "Median promised window (days)",
    "median_days": "Median promised window (days)",
    "p75_days": "P75 promised window (days)",
    "negative_windows": "Negative promised windows",
    "unavailable_windows": "Unavailable promised windows",
}


def format_analysis_table(table: DataFrame) -> Styler:
    """Format order counts, rates and durations without changing the analytical table."""
    formats = {}
    for column in table:
        if column == "approval_weekday":
            formats[column] = lambda value: ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[int(value)]
        elif column.endswith("_pct"):
            formats[column] = "{:.2f}%"
        elif column == "difference_pp_vs_12_18":
            formats[column] = "{:.2f}"
        elif (column.endswith("orders") or column in {
                "evaluated", "unavailable", "common_months", "valid_intervals",
                "unavailable_intervals", "inter_state_unknown_destination",
                "inter_state_to_destination", "selected_pair", "other_to_destination",
                "negative_windows", "unavailable_windows"}):
            formats[column] = "{:,.0f}"
        elif column.endswith(("_hours", "_days")):
            formats[column] = "{:,.2f}"
        elif column == "is_late":
            formats[column] = {False: "On time", True: "Late"}.__getitem__
    def label(column):
        if column == "evaluated" and "median_days" in table.columns:
            return "Evaluated promised windows"
        return _LABELS.get(column, column.replace("_", " ").capitalize())

    return (table.style.format(formats, na_rep="—")
            .format_index(label, axis="columns").hide(axis="index"))
