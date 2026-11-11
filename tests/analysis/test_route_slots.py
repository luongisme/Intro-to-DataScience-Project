"""Checks for all-slot state-pair comparisons and monthly standardization."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.analysis.orders import SLOT_LABELS
from src.analysis.route_slots import build_state_pair_orders, summarize_state_pair_slots
from src.analysis.investigation_plots import plot_state_pair_slots, plot_state_pair_months


def test_pair_selection_uses_every_item_without_counting_orders_twice():
    orders = pd.DataFrame({
        "order_id": ["pair", "multi_seller", "other", "blank"],
        "approval_slot": list(SLOT_LABELS), "is_late": [True, False, False, True],
        "order_purchase_timestamp": pd.to_datetime(["2018-02-01"] * 4),
    })
    items = pd.DataFrame({
        "order_id": ["pair", "pair", "multi_seller", "multi_seller", "other", "blank"],
        "order_item_id": [1, 2, 1, 2, 1, 1],
        "seller_state": [" SP ", "SP", "SP", "MG", "MG", "SP"],
        "customer_state": ["RJ", " RJ ", "RJ", "RJ", "RJ", " "],
    })
    routes = pd.DataFrame({"order_id": orders["order_id"],
                           "route_type": ["Inter-state"] * 4,
                           "items": [2, 2, 1, 1]})
    originals = [frame.copy(deep=True) for frame in (items, orders, routes)]
    result = build_state_pair_orders(items, orders, routes, ("SP", "RJ"))
    assert result["order_id"].tolist() == ["pair"]
    assert build_state_pair_orders(items, orders, routes, None).empty
    for frame, before in zip((items, orders, routes), originals):
        pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="complete items"):
        build_state_pair_orders(items.loc[~items["order_item_id"].eq(2)], orders,
                                routes, ("SP", "RJ"))
    duplicate = pd.concat([items, items.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique, non-null item keys"):
        build_state_pair_orders(duplicate, orders, routes, ("SP", "RJ"))


def test_common_month_rates_use_pooled_month_weights_and_purchase_date():
    rows = []
    for month, slot, count, late in (
        ("2018-01", 0, 1, 1), ("2018-01", 1, 1, 0),
        ("2018-01", 2, 1, 0), ("2018-01", 3, 1, 0),
        ("2018-02", 0, 3, 0), ("2018-02", 1, 3, 0),
        ("2018-02", 2, 3, 0), ("2018-02", 3, 3, 0),
    ):
        for i in range(count):
            rows.append((f"{month}-{slot}-{i}", SLOT_LABELS[slot], i < late,
                         pd.Timestamp(f"{month}-15")))
    frame = pd.DataFrame(rows, columns=["order_id", "approval_slot", "is_late",
                                        "order_purchase_timestamp"])
    before = frame.copy(deep=True)
    results = summarize_state_pair_slots(frame)
    raw = results["raw"].set_index("approval_slot")
    weighted = results["standardized"].query("scope == 'Common months'").set_index("approval_slot")
    assert raw.loc[SLOT_LABELS[0], "late_rate_pct"] == pytest.approx(25)
    assert raw.loc[SLOT_LABELS[0], "late_rate_95_lower_pct"] < 25
    assert raw.loc[SLOT_LABELS[0], "late_rate_95_upper_pct"] > 25
    assert weighted.loc[SLOT_LABELS[0], "standardized_late_rate_pct"] == pytest.approx(25)
    assert weighted.loc[SLOT_LABELS[0], "difference_pp_vs_12_18"] == pytest.approx(25)
    assert weighted["common_months"].eq(2).all()
    assert weighted["excluded_orders"].eq(0).all()
    assert results["monthly"]["orders"].sum() == len(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_missing_cells_and_endpoint_sensitivity_remain_explicit():
    frame = pd.DataFrame({
        "order_id": list("ABCDEF"),
        "approval_slot": [*SLOT_LABELS, SLOT_LABELS[0], SLOT_LABELS[2]],
        "is_late": [True, False, False, False, False, True],
        "order_purchase_timestamp": pd.to_datetime([
            "2018-01-01", "2018-02-01", "2018-02-01", "2018-02-01",
            "2018-03-01", "2018-03-01",
        ]),
    })
    results = summarize_state_pair_slots(frame)
    missing = results["monthly"].query("purchase_month == '2018-01' and approval_slot == @SLOT_LABELS[1]")
    assert missing.iloc[0]["orders"] == 0
    assert pd.isna(missing.iloc[0]["late_rate_pct"])
    assert results["standardized"]["standardized_late_rate_pct"].isna().all()
    empty = summarize_state_pair_slots(frame.iloc[:0])
    assert empty["raw"]["orders"].eq(0).all()
    assert empty["raw"]["late_rate_95_lower_pct"].isna().all()
    assert empty["standardized"]["standardized_late_rate_pct"].isna().all()

    complete = pd.DataFrame({
        "order_id": [f"{month}-{slot}" for month in range(3) for slot in range(4)],
        "approval_slot": list(SLOT_LABELS) * 3,
        "is_late": [True] * 4 + [False] * 4 + [True] * 4,
        "order_purchase_timestamp": pd.to_datetime(
            ["2018-01-01"] * 4 + ["2018-02-01"] * 4 + ["2018-03-01"] * 4),
    })
    sensitivity = summarize_state_pair_slots(complete)["standardized"]
    inner = sensitivity.query("scope == 'Interior common months'")
    assert inner["common_months"].eq(1).all()
    assert inner["included_orders"].eq(1).all()
    assert inner["excluded_orders"].eq(2).all()
    assert inner["standardized_late_rate_pct"].eq(0).all()


def test_pair_figures_render_from_summary_tables():
    frame = pd.DataFrame({
        "order_id": list("ABCD"), "approval_slot": list(SLOT_LABELS),
        "is_late": [True, False, True, False],
        "order_purchase_timestamp": pd.to_datetime(["2018-02-01"] * 4),
    })
    summary = summarize_state_pair_slots(frame)
    figures = [plot_state_pair_slots(summary["raw"], ("SP", "RJ")),
               plot_state_pair_months(summary["monthly"], ("SP", "RJ"))]
    try:
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)
