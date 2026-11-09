"""Independent boundary and cohort checks for delivery investigations."""

import pandas as pd
import pytest

from src.analysis.investigation import (
    measure_stage_hours, summarize_time_periods, compare_cleaning_cohorts,
    investigate_delivery_stages,
)
from src.analysis.orders import SLOT_LABELS


@pytest.mark.parametrize("start,end,elapsed,eligible", [
    ("2018-01-05 18:00", "2018-01-08 09:00", 63, 15),
    ("2017-12-29 18:00", "2018-01-01 09:00", 63, 6),
    ("2018-01-06 10:00", "2018-01-07 10:00", 24, 0),
    ("2018-01-08 09:30", "2018-01-08 17:15", 7.75, 7.75),
    ("2018-01-08 09:30", "2018-01-08 09:30", 0, 0),
    ("2018-04-30 18:00", "2018-05-02 09:00", 39, 15),
    ("2021-12-24 18:00", "2021-12-27 09:00", 63, 15),
])
def test_stage_hours_preserve_partial_days_and_remove_holidays_once(start, end, elapsed, eligible):
    a = pd.Series(pd.to_datetime([start]))
    b = pd.Series(pd.to_datetime([end]))
    before = a.copy(deep=True)
    result = measure_stage_hours(a, b).iloc[0]
    assert result["calendar_hours"] == pytest.approx(elapsed)
    assert result["eligible_hours"] == pytest.approx(eligible)
    assert result["status"] == "valid"
    pd.testing.assert_series_equal(a, before)


def test_missing_and_reversed_intervals_are_unassessable():
    a = pd.Series(pd.to_datetime([None, "2018-01-08 12:00"]))
    b = pd.Series(pd.to_datetime(["2018-01-09 12:00", "2018-01-08 09:00"]))
    result = measure_stage_hours(a, b)
    assert result["status"].tolist() == ["missing_endpoint", "reversed"]
    assert result[["calendar_hours", "eligible_hours"]].isna().all().all()


def sample_orders():
    frame = pd.DataFrame({
        "order_id": ["A", "B", "C"],
        "order_status": ["delivered"] * 3,
        "approval_slot": pd.Categorical([SLOT_LABELS[1]] * 3, categories=SLOT_LABELS, ordered=True),
        "is_late": [False, True, False],
        "delay_days": pd.array([pd.NA, 1, pd.NA], dtype="Int64"),
    })
    for column in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                   "order_delivered_customer_date", "order_estimated_delivery_date"]:
        frame[column] = pd.to_datetime(["2017-01-31 10:00", "2018-07-01 10:00", "2018-08-01 10:00"])
    # Purchase cohort and approval calendar month intentionally differ.
    frame.loc[0, "order_approved_at"] = pd.Timestamp("2017-02-01 10:00")
    return frame


def test_periods_use_purchase_month_and_keep_empty_months_and_slots():
    orders = sample_orders()
    result = summarize_time_periods(orders)
    assert result["monthly"]["orders"].sum() == 3
    assert result["monthly"].loc[result["monthly"]["month"].eq("2017-02"), "orders"].sum() == 0
    assert result["matched_years"]["orders"].sum() == 2
    assert len(result["matched_years"]) == 8


def test_cohort_comparison_counts_missing_and_conflicted_orders_without_dropping_valid_ones():
    orders = sample_orders()
    merged = pd.concat([orders, orders.iloc[[0]]], ignore_index=True)
    merged.loc[2, "order_approved_at"] = pd.NaT
    merged.loc[3, "order_approved_at"] = pd.Timestamp("2017-02-01 11:00")
    before = merged.copy(deep=True)
    retained = orders.loc[orders["order_id"].eq("B")].copy()
    retained["is_late"] = False
    retained.loc[:, "delay_days"] = pd.NA
    result = compare_cleaning_cohorts(merged, retained)
    counts = result["coverage"].set_index("measure")["orders"]
    assert counts["Orders with conflicting comparison fields"] == 1
    assert counts["Delivered orders missing comparison timestamps"] == 1
    assert counts["Comparable delivered orders before cleaning"] == 1
    assert result["slots"].groupby("cohort")["orders"].sum().to_dict() == {
        "Before cleaning": 1, "After cleaning": 1,
    }
    pd.testing.assert_frame_equal(merged, before)


def test_disjoint_order_ids_cannot_be_compared_even_if_counts_match():
    source = sample_orders()
    clean = source.assign(order_id=["X", "Y", "Z"])
    result = compare_cleaning_cohorts(source, clean)
    coverage = result["coverage"].set_index("measure")["orders"]
    assert coverage["Clean orders absent from the comparable source"] == 3
    assert result["slots"].empty


@pytest.mark.parametrize("field", ["order_approved_at", "order_delivered_customer_date",
                                    "order_estimated_delivery_date", "order_status"])
def test_changed_comparison_fields_are_counted_by_order(field):
    source = sample_orders()
    clean = source.iloc[::-1].copy()
    clean.loc[0, field] = "canceled" if field == "order_status" else pd.NaT
    result = compare_cleaning_cohorts(source, clean)
    coverage = result["coverage"].set_index("measure")["orders"]
    assert coverage["Matched orders with different comparison fields"] == 1
    assert coverage["Clean orders absent from the comparable source"] == 0
    assert result["slots"].empty


def test_stage_summaries_preserve_orders_and_report_invalid_endpoints():
    orders = sample_orders()
    before = orders.copy(deep=True)
    result = investigate_delivery_stages(orders)
    assert len(result["records"]) == 2 * len(orders)
    assert result["quality"]["orders"].sum() == 6
    assert result["summary"]["orders"].sum() == 6
    assert result["promised_summary"]["orders"].sum() == 3
    assert len(result["negative_windows"]) == 1
    pd.testing.assert_frame_equal(orders, before)


def test_stage_intervals_follow_orders_with_repeated_dataframe_indexes():
    orders = sample_orders()
    orders["order_delivered_carrier_date"] = orders["order_approved_at"] + pd.to_timedelta([1, 2, 3], unit="h")
    orders["order_delivered_customer_date"] = orders["order_delivered_carrier_date"] + pd.to_timedelta([7, 5, 4], unit="h")
    orders.index = [0, 0, 1]
    result = investigate_delivery_stages(orders)
    by_order = result["records"].pivot(index="order_id", columns="stage", values="calendar_hours")
    assert by_order["Processing"].to_dict() == {"A": 1, "B": 2, "C": 3}
    assert by_order["Carrier transit"].to_dict() == {"A": 7, "B": 5, "C": 4}


def test_pooled_median_uses_each_valid_interval_once_across_slots():
    orders = sample_orders()
    orders["is_late"] = False
    orders["approval_slot"] = [SLOT_LABELS[0], SLOT_LABELS[0], SLOT_LABELS[1]]
    orders["order_approved_at"] = pd.Timestamp("2018-07-02 00:00")
    orders["order_delivered_carrier_date"] = orders["order_approved_at"] + pd.to_timedelta([1, 3, 21], unit="h")
    pooled = investigate_delivery_stages(orders)["pooled_medians"].set_index("stage")
    assert pooled.loc["Processing", "on_time_calendar_hours"] == 3
    assert pooled.loc["Processing", "on_time_eligible_hours"] == 3
    assert pd.isna(pooled.loc["Processing", "late_calendar_hours"])


def test_investigation_figures_render_for_an_empty_cohort():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.investigation_plots import (
        plot_monthly_slot_rates, plot_matched_years, plot_stage_comparison, plot_promised_windows,
    )

    orders = sample_orders().iloc[:0]
    periods = summarize_time_periods(orders)
    stages = investigate_delivery_stages(orders)
    figures = [plot_monthly_slot_rates(periods["monthly"]),
               plot_matched_years(periods["matched_years"]),
               plot_stage_comparison(stages["summary"], "Processing"),
               plot_promised_windows(stages["windows"])]
    try:
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)
