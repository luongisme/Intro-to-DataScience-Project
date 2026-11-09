"""Route classification, reconciliation and figure checks."""

import matplotlib
import pandas as pd
import pytest

from src.analysis.investigation import classify_order_routes, summarize_routes
from src.analysis.investigation_plots import plot_route_slot_rates, plot_route_transit_time
from src.analysis.orders import SLOT_LABELS


matplotlib.use("Agg")


def route_fixture():
    orders = pd.DataFrame({
        "order_id": ["intra", "inter", "mixed", "unknown", "second_intra"],
        "approval_slot": pd.Categorical(
            [SLOT_LABELS[0], SLOT_LABELS[0], SLOT_LABELS[1], SLOT_LABELS[2], SLOT_LABELS[3]],
            categories=SLOT_LABELS, ordered=True,
        ),
        "is_late": [False, True, True, False, False],
    })
    items = pd.DataFrame({
        "order_id": ["intra", "intra", "inter", "mixed", "mixed", "unknown", "unknown",
                     "second_intra", "reviewed"],
        "customer_state": ["SP", "SP", "RJ", "SP", "SP", "SP", "SP", "MG", "BA"],
        "seller_state": ["SP", "SP", "SP", "SP", "RJ", "RJ", " ", "MG", "BA"],
    })
    stages = pd.DataFrame({
        "order_id": orders["order_id"], "stage": ["Carrier transit"] * len(orders),
        "status": ["valid", "valid", "valid", "missing_endpoint", "valid"],
        "calendar_hours": [10.0, 20.0, 30.0, float("nan"), 14.0],
        "eligible_hours": [8.0, 16.0, 24.0, float("nan"), 12.0],
    })
    return orders, items, stages


def test_route_classification_uses_every_item_and_excludes_reviewed_orders():
    orders, items, _ = route_fixture()
    original = items.copy(deep=True)
    routes = classify_order_routes(items, orders).set_index("order_id")
    assert routes["route_type"].astype(str).to_dict() == {
        "intra": "Intra-state", "inter": "Inter-state", "mixed": "Mixed",
        "unknown": "Unknown", "second_intra": "Intra-state",
    }
    assert routes.loc["intra", "items"] == 2
    assert "reviewed" not in routes.index
    pd.testing.assert_frame_equal(items, original)
    shuffled = classify_order_routes(items.sample(frac=1, random_state=17), orders).set_index("order_id")
    pd.testing.assert_series_equal(routes["route_type"].sort_index(),
                                   shuffled["route_type"].sort_index())


def test_unknown_takes_precedence_over_mixed_and_missing_items_fail():
    orders, items, _ = route_fixture()
    items.loc[len(items)] = ["mixed", "SP", None]
    routes = classify_order_routes(items, orders).set_index("order_id")
    assert routes.loc["mixed", "route_type"] == "Unknown"
    with pytest.raises(ValueError, match="No item route"):
        classify_order_routes(items.loc[items["order_id"].ne("intra")], orders)


def test_route_tables_reconcile_and_preserve_empty_cells_and_invalid_transit():
    orders, items, stages = route_fixture()
    before = stages.copy(deep=True)
    routes = classify_order_routes(items, orders)
    summary = summarize_routes(orders, routes, stages)
    totals = summary["totals"].set_index("route_type")
    assert totals["orders"].sum() == 5
    assert totals["late_orders"].sum() == 2
    assert len(summary["slots"]) == 16
    assert summary["slots"]["orders"].sum() == 5
    empty = summary["slots"].loc[summary["slots"]["route_type"].eq("Unknown")]
    assert empty["orders"].sum() == 1
    assert empty.loc[empty["orders"].eq(0), "late_rate_pct"].isna().all()
    transit = summary["transit"].set_index("route_type")
    assert transit.loc["Intra-state", "median_calendar_hours"] == 12
    assert transit.loc["Unknown", "unavailable"] == 1
    assert transit["evaluated"].sum() + transit["unavailable"].sum() == len(orders)
    pd.testing.assert_frame_equal(stages, before)


def test_route_summary_rejects_misaligned_inputs():
    orders, items, stages = route_fixture()
    routes = classify_order_routes(items, orders)
    with pytest.raises(ValueError, match="Route order IDs"):
        summarize_routes(orders, routes.iloc[:-1], stages)
    with pytest.raises(ValueError, match="Carrier transit order IDs"):
        summarize_routes(orders, routes, stages.iloc[:-1])


def test_route_figures_render_with_empty_groups():
    import matplotlib.pyplot as plt

    orders, items, stages = route_fixture()
    summary = summarize_routes(orders, classify_order_routes(items, orders), stages)
    figures = [plot_route_slot_rates(summary["slots"]),
               plot_route_transit_time(summary["transit"])]
    try:
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)


def test_empty_cohort_retains_four_routes_and_four_slots():
    import matplotlib.pyplot as plt

    orders, items, stages = route_fixture()
    orders, items, stages = orders.iloc[:0], items.iloc[:0], stages.iloc[:0]
    routes = classify_order_routes(items, orders)
    summary = summarize_routes(orders, routes, stages)
    assert len(summary["totals"]) == 4
    assert len(summary["slots"]) == 16
    assert summary["slots"]["orders"].eq(0).all()
    assert summary["slots"]["late_rate_pct"].isna().all()
    assert summary["transit"]["median_calendar_hours"].isna().all()
    figures = [plot_route_slot_rates(summary["slots"]),
               plot_route_transit_time(summary["transit"])]
    try:
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)
