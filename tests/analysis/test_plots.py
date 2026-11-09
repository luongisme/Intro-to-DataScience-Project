"""Check temporal plot inputs and empty-group behavior."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.plots import (
    build_temporal_profiles, plot_delay_distribution, plot_hour_weekday_heatmap,
    plot_rate_profile, plot_slot_counts,
)
from src.analysis.summary import summarize_delivery_by_slot
from src.analysis.orders import SLOT_LABELS


def sample_orders() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["A", "B", "C"],
        "approval_slot": pd.Categorical(
            [SLOT_LABELS[0], SLOT_LABELS[0], SLOT_LABELS[2]],
            categories=SLOT_LABELS, ordered=True,
        ),
        "approval_hour": [0, 0, 15],
        "approval_weekday": [0, 0, 4],
        "is_late": [True, False, True],
        "delay_days": pd.array([2, pd.NA, 5], dtype="Int64"),
    })


def test_temporal_profiles_reconcile_counts_and_leave_empty_rates_undefined():
    orders = sample_orders()
    before = orders.copy(deep=True)
    tables = build_temporal_profiles(orders)

    assert len(tables["hours"]) == 24
    assert len(tables["weekdays"]) == 7
    assert len(tables["hour_weekday"]) == 168
    for table in tables.values():
        assert table["orders"].sum() == 3
        assert table["late_orders"].sum() == 2
        assert table.loc[table["orders"].eq(0), "late_rate_pct"].isna().all()
    monday_midnight = tables["hour_weekday"].iloc[0]
    assert monday_midnight["orders"] == 2
    assert monday_midnight["late_orders"] == 1
    assert monday_midnight["late_rate_pct"] == 50
    pd.testing.assert_frame_equal(orders, before)


def test_figures_render_with_sparse_and_no_late_orders():
    orders = sample_orders()
    tables = build_temporal_profiles(orders)
    figures = [
        plot_slot_counts(summarize_delivery_by_slot(orders)),
        plot_rate_profile(tables["hours"], "approval_hour"),
        plot_rate_profile(tables["weekdays"], "approval_weekday"),
        plot_hour_weekday_heatmap(tables["hour_weekday"]),
        plot_delay_distribution(orders),
    ]
    empty = orders.iloc[:0]
    figures.extend([
        plot_rate_profile(build_temporal_profiles(empty)["hours"], "approval_hour"),
        plot_hour_weekday_heatmap(build_temporal_profiles(empty)["hour_weekday"]),
        plot_delay_distribution(empty),
    ])
    try:
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)
