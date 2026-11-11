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
from src.analysis.presentation import format_analysis_table


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
        "order_item_id": [1, 2, 1, 1, 1, 1],
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


def test_route_stages_reject_missing_pair_orders():
    items, orders, routes = stage_fixture()
    with pytest.raises(ValueError, match="complete state-pair cohort"):
        compare_route_delivery_stages(items, orders, routes, orders.iloc[:0],
                                      investigate_delivery_stages(orders), ("SP", "RJ"))


def test_route_stages_reject_partially_selected_pair():
    items, orders, routes = stage_fixture()
    items.loc[items.order_id.eq("B"), "seller_state"] = "SP"
    with pytest.raises(ValueError, match="complete state-pair cohort"):
        compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("A")],
                                      investigate_delivery_stages(orders), ("SP", "RJ"))


def test_route_stages_allow_a_genuinely_empty_pair():
    items, orders, routes = stage_fixture()
    items.loc[items.order_id.eq("A"), "seller_state"] = "MG"
    result = compare_route_delivery_stages(items, orders, routes, orders.iloc[:0],
                                            investigate_delivery_stages(orders), ("SP", "RJ"))
    assert result["coverage"]["selected_pair"].eq(0).all()
    assert result["coverage"].iloc[0]["other_to_destination"] == 3


@pytest.mark.parametrize("bad_key", ["duplicate", "missing"])
def test_route_stages_reject_invalid_item_keys(bad_key):
    items, orders, routes = stage_fixture()
    items.loc[1, "order_item_id"] = 1 if bad_key == "duplicate" else None
    with pytest.raises(ValueError, match="unique, non-null item keys"):
        compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("A")],
                                      investigate_delivery_stages(orders), ("SP", "RJ"))


@pytest.mark.parametrize("column,value", [
    ("calendar_hours", -24),
    ("eligible_hours", float("inf")),
    ("eligible_hours", 25),
])
def test_route_stages_reject_invalid_valid_durations(column, value):
    items, orders, routes = stage_fixture()
    profiles = investigate_delivery_stages(orders)
    mask = profiles["records"].order_id.eq("A") & profiles["records"].stage.eq("Processing")
    profiles["records"].loc[mask, column] = value
    with pytest.raises(ValueError, match="Valid stage durations"):
        compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("A")],
                                      profiles, ("SP", "RJ"))


def test_route_stages_accept_zero_valid_duration():
    items, orders, routes = stage_fixture()
    profiles = investigate_delivery_stages(orders)
    mask = profiles["records"].order_id.eq("A") & profiles["records"].stage.eq("Processing")
    profiles["records"].loc[mask, ["calendar_hours", "eligible_hours"]] = 0.0
    result = compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("A")],
                                            profiles, ("SP", "RJ"))
    row = result["stages"].query("period == 'Nov–Dec 2017' and group == 'SP → RJ' "
                                  "and approval_slot == @SLOT_LABELS[2] and stage == 'Processing'")
    assert row.iloc[0]["median_calendar_hours"] == 0


@pytest.mark.parametrize("excess_hours,accepted", [
    (5e-11, True),
    (2e-9, False),
])
def test_route_stages_limit_floating_point_duration_tolerance(excess_hours, accepted):
    items, orders, routes = stage_fixture()
    profiles = investigate_delivery_stages(orders)
    mask = profiles["records"].order_id.eq("A") & profiles["records"].stage.eq("Processing")
    records = profiles["records"]
    records.loc[mask, "eligible_hours"] = records.loc[mask, "calendar_hours"] + excess_hours
    args = (items, orders, routes, orders.loc[orders.order_id.eq("A")], profiles, ("SP", "RJ"))
    if accepted:
        result = compare_route_delivery_stages(*args)
        selected = result["stages"].query("period == 'Nov–Dec 2017' and group == 'SP → RJ' "
                                           "and approval_slot == @SLOT_LABELS[2] and stage == 'Processing'")
        assert selected.iloc[0]["valid_intervals"] == 1
    else:
        with pytest.raises(ValueError, match="Valid stage durations"):
            compare_route_delivery_stages(*args)


@pytest.mark.parametrize("purchase_date,expected_period", [
    ("2017-10-31", None),
    ("2017-11-01", "Nov–Dec 2017"),
    ("2017-12-31", "Nov–Dec 2017"),
    ("2018-01-01", None),
    ("2018-01-31", None),
    ("2018-02-01", "Feb–Mar 2018"),
    ("2018-03-31", "Feb–Mar 2018"),
    ("2018-04-01", "Apr–Jul 2018"),
    ("2018-07-31", "Apr–Jul 2018"),
    ("2018-08-01", None),
])
def test_route_stages_use_inclusive_purchase_period_boundaries(purchase_date, expected_period):
    items, orders, routes = stage_fixture()
    b = orders.order_id.eq("B")
    shift = pd.Timestamp(purchase_date) - orders.loc[b, "order_purchase_timestamp"].iloc[0].normalize()
    for column in ("order_purchase_timestamp", "order_approved_at",
                   "order_delivered_carrier_date", "order_delivered_customer_date",
                   "order_estimated_delivery_date"):
        orders.loc[b, column] += shift
    result = compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("A")],
                                            investigate_delivery_stages(orders), ("SP", "RJ"))
    counts = result["coverage"].set_index("period")["other_to_destination"].to_dict()
    assert counts == {
        "Nov–Dec 2017": 1 + (expected_period == "Nov–Dec 2017"),
        "Feb–Mar 2018": 1 + (expected_period == "Feb–Mar 2018"),
        "Apr–Jul 2018": int(expected_period == "Apr–Jul 2018"),
    }
    assert sum(counts.values()) == 2 + int(expected_period is not None)


def test_route_stages_calculate_p75_from_order_values():
    items, orders, routes = stage_fixture()
    for new_id in ("F", "G"):
        extra_order = orders.loc[orders.order_id.eq("C")].copy()
        extra_order["order_id"] = new_id
        orders = pd.concat([orders, extra_order], ignore_index=True)
        extra_item = items.loc[items.order_id.eq("C")].copy()
        extra_item["order_id"] = new_id
        items = pd.concat([items, extra_item], ignore_index=True)
        extra_route = routes.loc[routes.order_id.eq("C")].copy()
        extra_route["order_id"] = new_id
        routes = pd.concat([routes, extra_route], ignore_index=True)
    profiles = investigate_delivery_stages(orders)
    for order_id, hours, promised in zip("BCFG", (0, 10, 20, 40), (1, 2, 4, 8)):
        processing = (profiles["records"].order_id.eq(order_id)
                      & profiles["records"].stage.eq("Processing"))
        profiles["records"].loc[processing, ["calendar_hours", "eligible_hours"]] = hours
        profiles["windows"].loc[profiles["windows"].order_id.eq(order_id), "promised_days"] = promised
    result = compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("A")],
                                            profiles, ("SP", "RJ"))
    group = "Other inter-state to RJ"
    stage = result["stages"].query("period == 'Nov–Dec 2017' and group == @group "
                                    "and approval_slot == @SLOT_LABELS[2] and stage == 'Processing'").iloc[0]
    promise = result["promises"].query("period == 'Nov–Dec 2017' and group == @group "
                                        "and approval_slot == @SLOT_LABELS[2]").iloc[0]
    assert (stage["orders"], stage["valid_intervals"]) == (4, 4)
    assert stage["median_calendar_hours"] == pytest.approx(15)
    assert stage["p75_calendar_hours"] == pytest.approx(25)
    assert promise["median_days"] == pytest.approx(3)
    assert promise["p75_days"] == pytest.approx(5)


def test_afternoon_evidence_joins_matching_period_group_and_slot():
    items, orders, routes = stage_fixture()
    result = compare_route_delivery_stages(items, orders, routes, orders.loc[orders.order_id.eq("A")],
                                            investigate_delivery_stages(orders), ("SP", "RJ"))
    evidence = summarize_afternoon_route_evidence(result)
    assert len(evidence) == 6
    pair = evidence.query("period == 'Nov–Dec 2017' and group == 'SP → RJ'").iloc[0]
    other = evidence.query("period == 'Nov–Dec 2017' and group == 'Other inter-state to RJ'").iloc[0]
    assert (pair["orders"], pair["late_orders"], pair["late_rate_pct"]) == (1, 0, 0)
    assert (pair["median_processing_hours"], pair["median_transit_hours"],
            pair["median_promised_days"]) == (24, 24, 3)
    assert (other["orders"], other["late_orders"], other["late_rate_pct"]) == (2, 2, 100)
    assert (other["median_processing_hours"], other["median_transit_hours"],
            other["median_promised_days"]) == (36, 48, 1.5)
    assert evidence.loc[evidence["period"].eq("Apr–Jul 2018"), "orders"].eq(0).all()


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


def test_route_tables_show_units_scope_and_formatted_counts_without_mutation():
    tables = [
        pd.DataFrame({"raw_late_rate_pct": [3.14159],
                      "standardized_late_rate_pct": [4.5],
                      "difference_pp_vs_12_18": [1.23456],
                      "common_months": [21]}),
        pd.DataFrame({"inter_state_orders": [12345], "selected_pair": [2345],
                      "inter_state_unknown_destination": [1234],
                      "other_to_destination": [7890]}),
        pd.DataFrame({"valid_intervals": [1234], "unavailable_intervals": [2],
                      "median_calendar_hours": [24.234], "p75_eligible_hours": [35.678]}),
        pd.DataFrame({"evaluated": [2345], "negative_windows": [3],
                      "median_days": [25.5], "p75_days": [30.25]}),
    ]
    originals = [table.copy(deep=True) for table in tables]
    rendered = [format_analysis_table(table).to_html() for table in tables]
    for label in ("Full-cohort raw late rate", "Standardized late rate",
                  "Difference from 12:00–18:00 (pp)", "Common purchase months",
                  "Inter-state orders with unknown destination", "Selected pair orders",
                  "Other inter-state orders to destination", "Valid stage intervals",
                  "Unavailable stage intervals", "Median elapsed time (h)",
                  "P75 eligible-day time (h)", "Evaluated promised windows",
                  "Median promised window (days)", "P75 promised window (days)"):
        assert any(label in html for html in rendered)
    assert "3.14%" in rendered[0]
    assert "1.23" in rendered[0]
    assert "1.23456" not in rendered[0]
    assert all(value in rendered[1] for value in ("12,345", "2,345", "1,234", "7,890"))
    assert "24.23" in rendered[2]
    assert "35.68" in rendered[2]
    assert "30.25" in rendered[3]
    for table, original in zip(tables, originals):
        pd.testing.assert_frame_equal(table, original)
