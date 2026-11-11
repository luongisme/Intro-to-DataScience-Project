"""Check destination-matched delivery-stage comparisons."""

import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.investigation import investigate_delivery_stages
from src.analysis.orders import SLOT_LABELS
from src.analysis.route_stages import compare_route_delivery_stages, summarize_afternoon_route_evidence
from src.analysis.investigation_plots import plot_route_period_slot_rates


def stage_fixture():
    orders = pd.DataFrame({
        "order_id": list("ABCDE"),
        "order_purchase_timestamp": pd.to_datetime([
            "2017-11-01 09:00", "2017-11-02 09:00", "2017-11-03 09:00",
            "2018-02-01 09:00", "2017-10-31 09:00"]),
        "order_approved_at": pd.to_datetime([
            "2017-11-01 13:00", "2017-11-02 13:00", "2017-11-03 13:00",
            "2018-02-01 01:00", "2017-11-01 13:00"]),
        "order_delivered_carrier_date": pd.to_datetime([
            "2017-11-02 13:00", "2017-11-04 13:00", "2017-11-04 13:00",
            "2018-02-02 01:00", "2017-11-02 13:00"]),
        "order_delivered_customer_date": pd.to_datetime([
            "2017-11-03 13:00", "2017-11-06 13:00", "2017-11-06 13:00",
            "2018-02-03 01:00", "2017-11-03 13:00"]),
        "order_estimated_delivery_date": pd.to_datetime([
            "2017-11-04", "2017-11-04", "2017-11-04", "2018-02-04", "2017-11-04"]),
        "approval_slot": [SLOT_LABELS[2]] * 3 + [SLOT_LABELS[0], SLOT_LABELS[2]],
        "is_late": [False, True, True, False, False],
    })
    items = pd.DataFrame({
        "order_id": ["A", "A", "B", "C", "D", "E"],
        "customer_state": ["RJ", "RJ", "RJ", "RJ", "RJ", "MG"],
        "seller_state": ["SP", "SP", "MG", "ES", "MG", "SP"],
    })
    routes = pd.DataFrame({"order_id": list("ABCDE"),
                           "route_type": ["Inter-state"] * 5, "items": [2, 1, 1, 1, 1]})
    return items, orders, routes


def test_route_stages_keep_disjoint_orders_and_purchase_periods():
    items, orders, routes = stage_fixture()
    profiles = investigate_delivery_stages(orders)
    profiles["records"].loc[
        profiles["records"]["order_id"].eq("B")
        & profiles["records"]["stage"].eq("Processing"),
        ["status", "calendar_hours", "eligible_hours"]] = ["reversed", float("nan"), float("nan")]
    originals = [frame.copy(deep=True) for frame in (items, orders, routes,
                                                      profiles["records"], profiles["windows"])]
    result = compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("A")],
                                            profiles, ("SP", "RJ"))
    evidence = summarize_afternoon_route_evidence(result)
    assert len(evidence) == 6
    selected = evidence.loc[evidence["group"].eq("SP → RJ") &
                            evidence["period"].eq("Nov–Dec 2017")].iloc[0]
    assert selected["orders"] == 1
    assert selected["median_processing_hours"] == 24
    assert selected["median_transit_hours"] == 24
    november = result["outcomes"].query("period == 'Nov–Dec 2017' and approval_slot == @SLOT_LABELS[2]")
    assert november["orders"].tolist() == [1, 2]
    assert november["late_orders"].tolist() == [0, 2]
    assert result["coverage"].iloc[0]["inter_state_to_destination"] == 3
    assert result["coverage"].iloc[0]["selected_pair"] == 1
    assert result["coverage"].iloc[0]["other_to_destination"] == 2
    processing = result["stages"].query(
        "period == 'Nov–Dec 2017' and group == 'Other inter-state to RJ' "
        "and approval_slot == @SLOT_LABELS[2] and stage == 'Processing'")
    assert processing.iloc[0]["valid_intervals"] == 1
    assert processing.iloc[0]["unavailable_intervals"] == 1
    assert result["promises"].loc[result["promises"].orders.gt(0), "evaluated"].sum() == 4
    assert result["outcomes"].loc[result["outcomes"].period.eq("Apr–Jul 2018"), "orders"].eq(0).all()
    assert result["outcomes"].loc[result["outcomes"].period.eq("Apr–Jul 2018"), "late_rate_pct"].isna().all()
    for frame, original in zip((items, orders, routes, profiles["records"], profiles["windows"]), originals):
        pd.testing.assert_frame_equal(frame, original)


def test_route_stages_reject_partial_item_input():
    items, orders, routes = stage_fixture()
    with pytest.raises(ValueError, match="complete cleaned items"):
        compare_route_delivery_stages(items.iloc[1:], orders, routes,
                                      orders.loc[orders.order_id.eq("A")],
                                      investigate_delivery_stages(orders), ("SP", "RJ"))


def test_route_stages_reject_mislabelled_selected_pair():
    items, orders, routes = stage_fixture()
    with pytest.raises(ValueError, match="seller and customer states"):
        compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("B")],
                                      investigate_delivery_stages(orders), ("SP", "RJ"))


def test_period_slot_heatmap_shows_counts_and_undefined_cells():
    rows = []
    for period in ("First", "Second"):
        for group in ("SP → RJ", "Other inter-state to RJ"):
            for slot in SLOT_LABELS:
                count = 0 if period == "Second" and slot == SLOT_LABELS[3] else 4
                late = 1 if count and group == "SP → RJ" else 0
                rows.append((period, group, slot, count, late,
                             100 * late / count if count else float("nan")))
    table = pd.DataFrame(rows, columns=["period", "group", "approval_slot",
                                        "orders", "late_orders", "late_rate_pct"])
    before = table.copy(deep=True)
    figure = plot_route_period_slot_rates(table)
    try:
        figure.canvas.draw()
        assert figure.axes[0].images[0].norm.vmax == figure.axes[1].images[0].norm.vmax
        labels = [text.get_text() for text in figure.axes[0].texts]
        assert "25.0%\nN=4" in labels
        assert "—\nN=0" in labels
    finally:
        plt.close(figure)
    pd.testing.assert_frame_equal(table, before)
    with pytest.raises(ValueError, match="all purchase-period"):
        plot_route_period_slot_rates(table.iloc[1:])
