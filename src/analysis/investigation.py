"""Cohort stability and recorded delivery-stage durations."""

import numpy as np
import pandas as pd

from .orders import SLOT_LABELS
from .summary import summarize_delivery_by_slot


def summarize_time_periods(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Use purchase months for cohorts and approval times for slot assignment."""
    monthly = []
    purchase = orders["order_purchase_timestamp"]
    months = purchase.dt.to_period("M")
    if not orders.empty:
        for month in pd.period_range(months.min(), months.max(), freq="M"):
            monthly.append(summarize_delivery_by_slot(orders.loc[months.eq(month)]).assign(month=str(month)))
    columns = [*summarize_delivery_by_slot(orders.iloc[:0]).columns, "month"]
    monthly_table = pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame(columns=columns)
    matched = []
    for year in (2017, 2018):
        selected = orders.loc[purchase.dt.year.eq(year) & purchase.dt.month.le(7)]
        matched.append(summarize_delivery_by_slot(selected).assign(year=year))
    return {"monthly": monthly_table, "matched_years": pd.concat(matched, ignore_index=True)}


def compare_cleaning_cohorts(merged: pd.DataFrame, orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare delivered orders with known label inputs against the clean cohort."""
    fields = ["order_status", "order_approved_at", "order_delivered_customer_date",
              "order_estimated_delivery_date"]
    distinct = merged.groupby("order_id")[fields].nunique(dropna=False)
    conflicts = distinct.index[distinct.gt(1).any(axis=1)]
    source = merged.loc[~merged["order_id"].isin(conflicts), ["order_id", *fields]].drop_duplicates("order_id").copy()
    delivered = source.loc[source["order_status"].eq("delivered")].copy()
    comparable = delivered.dropna(subset=fields[1:]).copy()
    comparable["approval_slot"] = pd.Categorical.from_codes(
        comparable["order_approved_at"].dt.hour // 6, categories=SLOT_LABELS, ordered=True)
    days = (comparable["order_delivered_customer_date"].dt.normalize() -
            comparable["order_estimated_delivery_date"].dt.normalize()).dt.days
    comparable["is_late"] = days.gt(0)
    comparable["delay_days"] = days.where(days.gt(0)).astype("Int64")
    tables = [summarize_delivery_by_slot(comparable).assign(cohort="Before cleaning"),
              summarize_delivery_by_slot(orders).assign(cohort="After cleaning")]
    coverage = pd.DataFrame([
        ["Orders with conflicting comparison fields", len(conflicts)],
        ["Delivered orders with consistent comparison fields", len(delivered)],
        ["Delivered orders missing comparison timestamps", len(delivered) - len(comparable)],
        ["Comparable delivered orders before cleaning", len(comparable)],
        ["Analyzable orders after cleaning", len(orders)],
    ], columns=["measure", "orders"])
    return {"coverage": coverage, "slots": pd.concat(tables, ignore_index=True)}


def measure_stage_hours(start: pd.Series, end: pd.Series) -> pd.DataFrame:
    """Measure [start, end) in recorded-clock and eligible-day hours.

    Eligible days are Monday-Friday excluding Jan 1, May 1 and Dec 25.
    A full eligible day contributes 24 hours, with partial endpoints preserved.
    Missing or reversed intervals receive null durations and an explicit status.
    """
    if not start.index.equals(end.index):
        raise ValueError("Stage endpoints must have matching indexes.")
    if start.dt.tz is not None or end.dt.tz is not None:
        raise ValueError("Use the timezone-naive recorded timestamps for this convention.")
    valid = start.notna() & end.notna() & end.ge(start)
    result = pd.DataFrame({"calendar_hours": np.nan, "eligible_hours": np.nan,
                           "status": "missing_endpoint"}, index=start.index)
    result.loc[start.notna() & end.notna() & end.lt(start), "status"] = "reversed"
    if valid.any():
        first_year = int(start.loc[valid].dt.year.min())
        last_year = int(end.loc[valid].dt.year.max())
        holidays = [f"{year}-{day}" for year in range(first_year, last_year + 1)
                    for day in ("01-01", "05-01", "12-25")]
        calendar = np.busdaycalendar(holidays=holidays)

        def accumulated_hours(timestamps):
            times = timestamps.to_numpy(dtype="datetime64[us]")
            dates = times.astype("datetime64[D]")
            days = np.busday_count("1970-01-01", dates, busdaycal=calendar)
            fractions = (times - dates) / np.timedelta64(1, "h")
            return 24 * days + np.is_busday(dates, busdaycal=calendar) * fractions

        result.loc[valid, "calendar_hours"] = (end.loc[valid] - start.loc[valid]).dt.total_seconds() / 3600
        result.loc[valid, "eligible_hours"] = accumulated_hours(end.loc[valid]) - accumulated_hours(start.loc[valid])
        result.loc[valid, "status"] = "valid"
    return result


def investigate_delivery_stages(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return stage records, valid-duration summaries and promised-date windows."""
    records = []
    for name, start, end in (
        ("Processing", "order_approved_at", "order_delivered_carrier_date"),
        ("Carrier transit", "order_delivered_carrier_date", "order_delivered_customer_date"),
    ):
        frame = orders[["order_id", "approval_slot", "is_late"]].copy()
        durations = measure_stage_hours(orders[start], orders[end])
        records.append(frame.join(durations).assign(stage=name))
    stages = pd.concat(records, ignore_index=True)
    summary = stages.groupby(["stage", "approval_slot", "is_late"], observed=True).agg(
        orders=("order_id", "size"), evaluated=("calendar_hours", "count"),
        mean_calendar_hours=("calendar_hours", "mean"), median_calendar_hours=("calendar_hours", "median"),
        mean_eligible_hours=("eligible_hours", "mean"), median_eligible_hours=("eligible_hours", "median"),
    ).reset_index()
    windows = orders[["order_id", "approval_slot", "order_approved_at",
                      "order_estimated_delivery_date"]].copy()
    windows["promised_days"] = (orders["order_estimated_delivery_date"].dt.normalize() -
                               orders["order_approved_at"].dt.normalize()).dt.days
    promised = windows.groupby("approval_slot", observed=False).agg(
        orders=("order_id", "size"), mean_days=("promised_days", "mean"),
        median_days=("promised_days", "median"), min_days=("promised_days", "min"),
        max_days=("promised_days", "max"),
    ).reset_index()
    quality = stages.groupby(["stage", "status"]).size().rename("orders").reset_index()
    return {"records": stages, "summary": summary, "quality": quality,
            "windows": windows, "promised_summary": promised,
            "negative_windows": windows.loc[windows["promised_days"].lt(0)].copy()}
