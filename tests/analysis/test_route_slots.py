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


def test_pair_selection_allows_intra_state_only_with_explicit_option():
    orders = pd.DataFrame({
        "order_id": ["intra", "inter"],
        "approval_slot": [SLOT_LABELS[0]] * 2, "is_late": [False, True],
        "order_purchase_timestamp": pd.to_datetime(["2018-02-01"] * 2),
    })
    items = pd.DataFrame({
        "order_id": ["intra", "intra", "inter", "inter"], "order_item_id": [1, 2, 1, 2],
        "seller_state": [" SP ", "SP", "SP", "SP"],
        "customer_state": ["SP", " SP ", "RJ", "RJ"],
    })
    routes = pd.DataFrame({
        "order_id": orders["order_id"], "route_type": ["Intra-state", "Inter-state"],
        "items": [2, 2],
    })
    originals = [frame.copy(deep=True) for frame in (items, orders, routes)]
    with pytest.raises(ValueError, match="must be inter-state"):
        build_state_pair_orders(items, orders, routes, ("SP", "SP"))
    chosen = build_state_pair_orders(items, orders, routes, (" SP ", "SP"), include_intra_state=True)
    assert chosen["order_id"].tolist() == ["intra"]
    legacy = build_state_pair_orders(items, orders, routes, ("SP", "RJ"))
    expanded = build_state_pair_orders(items, orders, routes, ("SP", "RJ"), include_intra_state=True)
    pd.testing.assert_frame_equal(legacy, expanded)
    assert build_state_pair_orders(items, orders, routes, None, include_intra_state=True).empty
    for frame, before in zip((items, orders, routes), originals):
        pd.testing.assert_frame_equal(frame, before)

    for defect in ("partial", "duplicate", "wrong_route"):
        altered = items.copy()
        altered_routes = routes.copy()
        if defect == "partial":
            altered = altered.iloc[1:]
            message = "complete items"
        elif defect == "duplicate":
            altered.loc[1, "order_item_id"] = 1
            message = "unique, non-null item keys"
        else:
            altered_routes.loc[0, "route_type"] = "Inter-state"
            message = "conflicts with the route classification"
        with pytest.raises(ValueError, match=message):
            build_state_pair_orders(altered, orders, altered_routes, ("SP", "SP"),
                                    include_intra_state=True)


@pytest.mark.parametrize("flag", [None, 1, "yes"])
def test_pair_scope_requires_an_explicit_boolean(flag):
    from src.analysis.investigation import summarize_state_pairs

    for function, args in [(build_state_pair_orders, (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None)),
                           (summarize_state_pairs, (pd.DataFrame(), pd.DataFrame()))]:
        with pytest.raises(ValueError, match="include_intra_state must be a boolean"):
            function(*args, include_intra_state=flag)


def test_shared_month_helpers_keep_two_slot_and_four_slot_weights_distinct():
    from src.analysis.route_slots import _purchase_month_profile, _standardize_month_profile

    rows = []
    for month, slot, count, late in [
        ("2018-01", 0, 1, 1), ("2018-01", 1, 9, 0),
        ("2018-01", 2, 1, 0), ("2018-01", 3, 9, 0),
        ("2018-02", 0, 3, 0), ("2018-02", 1, 1, 0),
        ("2018-02", 2, 1, 1), ("2018-02", 3, 1, 0),
    ]:
        for index in range(count):
            rows.append({"order_id": f"{month}-{slot}-{index}",
                         "approval_slot": SLOT_LABELS[slot], "is_late": index < late,
                         "order_purchase_timestamp": pd.Timestamp(f"{month}-15")})
    orders = pd.DataFrame(rows)
    before = orders.copy(deep=True)
    four = summarize_state_pair_slots(orders)
    four_weighted = four["standardized"].query("scope == 'Common months'").set_index("approval_slot")
    assert four_weighted.loc[SLOT_LABELS[0], "standardized_late_rate_pct"] == pytest.approx(100 * 20 / 26)
    slots = (SLOT_LABELS[0], SLOT_LABELS[2])
    two_orders = orders.loc[orders["approval_slot"].isin(slots)]
    profile = _purchase_month_profile(two_orders, slots)
    two = _standardize_month_profile(profile, slots)
    assert two["weights"]["pooled_orders"].tolist() == [2, 4]
    assert two["summary"]["standardized_late_rate_pct"].tolist() == pytest.approx([100 / 3, 200 / 3])
    interior = _standardize_month_profile(profile, slots, excluded_months=("2018-01",))
    assert interior["weights"]["weight"].tolist() == [1]
    assert interior["summary"]["included_orders"].tolist() == [3, 1]
    assert interior["summary"]["excluded_late_orders"].tolist() == [1, 0]
    pd.testing.assert_frame_equal(orders, before)
