"""Independent boundary and cohort checks for delivery investigations."""

import pandas as pd
import pytest

from src.analysis.investigation import (
    summarize_focus_routes, measure_stage_hours, summarize_time_periods, compare_cleaning_cohorts,
    investigate_delivery_stages, classify_order_routes, build_focus_cohort,
    summarize_destination_states, summarize_state_pairs, summarize_focus_time, summarize_focus_comparisons, REVIEW_DESTINATION,
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


def focus_fixture():
    orders = pd.DataFrame({
        "order_id": ["known", "conflicting", "missing", "incomplete", "late", "outside"],
        "approval_slot": pd.Categorical(
            [SLOT_LABELS[2]] * 5 + [SLOT_LABELS[0]], categories=SLOT_LABELS, ordered=True),
        "is_late": [False, True, False, False, True, True],
        "approval_hour": [13, 14, 15, 16, 17, 2],
    })
    items = pd.DataFrame({
        "order_id": ["known", "known", "conflicting", "conflicting", "missing", "missing",
                     "incomplete", "incomplete", "late", "late", "outside"],
        "customer_state": ["SP", "SP", "RJ", "MG", None, " ", "SP", " ", "SP", "SP", "RJ"],
        "seller_state": ["SP"] * 11,
    })
    return items, orders, classify_order_routes(items, orders)


def test_focus_cohort_keeps_outcomes_and_separates_destination_review():
    items, orders, routes = focus_fixture()
    before = [frame.copy(deep=True) for frame in (items, orders, routes)]

    focus, diagnostics = build_focus_cohort(items, orders, routes)
    by_id = focus.set_index("order_id")
    review = diagnostics["location_review"].set_index("order_id")
    summary = diagnostics["summary"].iloc[0]

    assert len(focus) == 5
    assert "outside" not in by_id.index
    assert by_id["is_late"].sum() == 2
    assert by_id.loc["known", "customer_state"] == "SP"
    assert by_id.loc["known", "items"] == 2
    assert by_id.loc["known", "route_type"] == "Intra-state"
    assert by_id.loc["late", "approval_hour"] == 17
    assert by_id.loc["conflicting", "location_status"] == "conflicting"
    assert by_id.loc["missing", "location_status"] == "missing"
    assert by_id.loc["incomplete", "location_status"] == "incomplete"
    assert by_id.loc[["conflicting", "missing", "incomplete"], "customer_state"].isna().all()
    assert review.loc["conflicting", "observed_states"] == "MG, RJ"
    assert review.loc["missing", "missing_items"] == 2
    assert review.loc["incomplete", "observed_states"] == "SP"
    assert summary["orders"] == 5
    assert summary["late_orders"] == 2
    assert summary["known_orders"] == 2
    assert summary["review_orders"] == 3
    assert summary["late_rate_pct"] == pytest.approx(40)
    for frame, original in zip((items, orders, routes), before):
        pd.testing.assert_frame_equal(frame, original)


def test_focus_cohort_rejects_missing_routes_and_source_items():
    items, orders, routes = focus_fixture()
    with pytest.raises(ValueError, match="no route classification"):
        build_focus_cohort(items, orders, routes.loc[routes["order_id"].ne("late")])
    with pytest.raises(ValueError, match="no source items"):
        build_focus_cohort(items.loc[items["order_id"].ne("late")], orders, routes)


def test_focus_cohort_handles_empty_slot():
    items, orders, routes = focus_fixture()
    orders = orders.loc[orders["order_id"].eq("outside")]
    focus, diagnostics = build_focus_cohort(items, orders, routes)
    assert focus.empty
    assert diagnostics["location_review"].empty
    assert diagnostics["summary"].iloc[0]["orders"] == 0
    assert pd.isna(diagnostics["summary"].iloc[0]["late_rate_pct"])


@pytest.mark.parametrize("slot,expected_ids", [
    (SLOT_LABELS[0], ["outside"]),
    (SLOT_LABELS[1], []),
    (SLOT_LABELS[2], ["known", "conflicting", "missing", "incomplete", "late"]),
    (SLOT_LABELS[3], []),
])
def test_focus_cohort_selects_the_requested_slot_without_mutating_inputs(slot, expected_ids):
    items, orders, routes = focus_fixture()
    originals = [frame.copy(deep=True) for frame in (items, orders, routes)]
    focus, details = build_focus_cohort(items, orders, routes, approval_slot=slot)

    assert focus["order_id"].tolist() == expected_ids
    summary = details["summary"].iloc[0]
    assert summary["approval_slot"] == slot
    assert summary["orders"] == len(expected_ids)
    assert summary["known_orders"] + summary["review_orders"] == len(expected_ids)
    if slot == SLOT_LABELS[0]:
        assert summary["late_orders"] == 1
        assert summary["late_rate_pct"] == pytest.approx(100)
        assert focus.iloc[0]["customer_state"] == "RJ"
        assert focus.iloc[0]["route_type"] == "Inter-state"
    for frame, original in zip((items, orders, routes), originals):
        pd.testing.assert_frame_equal(frame, original)


def test_focus_cohort_default_matches_explicit_afternoon_and_empty_schema():
    items, orders, routes = focus_fixture()
    default, default_details = build_focus_cohort(items, orders, routes)
    explicit, explicit_details = build_focus_cohort(
        items, orders, routes, approval_slot=SLOT_LABELS[2],
    )
    pd.testing.assert_frame_equal(default, explicit)
    for name in default_details:
        pd.testing.assert_frame_equal(default_details[name], explicit_details[name])

    empty, empty_details = build_focus_cohort(
        items, orders, routes, approval_slot=SLOT_LABELS[1],
    )
    assert empty.empty
    assert empty.columns.tolist() == default.columns.tolist()
    for name in default_details:
        assert empty_details[name].columns.tolist() == default_details[name].columns.tolist()
    assert empty_details["location_review"].empty
    assert pd.isna(empty_details["summary"].iloc[0]["late_rate_pct"])


@pytest.mark.parametrize("slot", [None, "", "00:00-06:00", "06:00–00:00", 0, [], pd.NA])
def test_focus_cohort_rejects_invalid_slot_labels(slot):
    items, orders, routes = focus_fixture()
    with pytest.raises(ValueError, match="approval_slot must be one of"):
        build_focus_cohort(items, orders, routes, approval_slot=slot)


def test_focus_cohort_uses_derived_slots_at_exact_approval_boundaries():
    from src.analysis.orders import build_order_analysis_view

    clock_times = [
        "00:00:00", "05:59:59.999999", "06:00:00", "11:59:59.999999",
        "12:00:00", "17:59:59.999999", "18:00:00", "23:59:59.999999",
    ]
    rows = []
    for index, clock in enumerate(clock_times):
        for item_id in (1, 2):
            rows.append({
                "order_id": f"order-{index}", "order_item_id": item_id,
                "order_status": "delivered", "customer_state": "RJ", "seller_state": "SP",
                "order_purchase_timestamp": pd.Timestamp("2018-01-07 12:00"),
                "order_approved_at": pd.Timestamp(f"2018-01-08 {clock}"),
                "order_delivered_carrier_date": pd.Timestamp("2018-01-09 12:00"),
                "order_delivered_customer_date": pd.Timestamp("2018-01-10 12:00"),
                "order_estimated_delivery_date": pd.Timestamp("2018-01-09"),
            })
    items = pd.DataFrame(rows)
    orders, _ = build_order_analysis_view(items)
    routes = classify_order_routes(items, orders)

    for slot, expected_ids in [
        (SLOT_LABELS[0], ["order-0", "order-1"]),
        (SLOT_LABELS[1], ["order-2", "order-3"]),
        (SLOT_LABELS[2], ["order-4", "order-5"]),
        (SLOT_LABELS[3], ["order-6", "order-7"]),
    ]:
        focus, details = build_focus_cohort(items, orders, routes, approval_slot=slot)
        assert focus["order_id"].tolist() == expected_ids
        assert focus["items"].tolist() == [2, 2]
        assert details["summary"].iloc[0]["orders"] == 2


def test_destination_profiles_reconcile_review_orders_and_scope_denominators():
    focus = pd.DataFrame({
        "order_id": list("ABCDEFG"),
        "customer_state": ["SP", "SP", "RJ", "RJ", None, None, "MG"],
        "location_status": ["known"] * 4 + ["missing", "conflicting", "known"],
        "route_type": ["Intra-state", "Inter-state", "Inter-state", "Inter-state",
                       "Unknown", "Inter-state", "Mixed"],
        "is_late": [True, False, True, False, True, True, False],
    })
    before = focus.copy(deep=True)
    profiles = summarize_destination_states(focus)
    for name, expected_orders, expected_late in (("all_routes", 7, 4), ("inter_state", 4, 2)):
        table = profiles[name]
        assert table["orders"].sum() == expected_orders
        assert table["late_orders"].sum() == expected_late
        assert table["late_share_pct"].sum() == pytest.approx(100)
    all_routes = profiles["all_routes"].set_index("customer_state")
    inter = profiles["inter_state"].set_index("customer_state")
    assert all_routes.loc["RJ", "late_rate_pct"] == pytest.approx(50)
    assert all_routes.loc["RJ", "late_share_pct"] == pytest.approx(25)
    assert inter.loc["RJ", "late_share_pct"] == pytest.approx(50)
    assert all_routes.loc[REVIEW_DESTINATION, "orders"] == 2
    assert inter.loc[REVIEW_DESTINATION, "orders"] == 1
    assert profiles["inter_state"]["customer_state"].tolist() == ["RJ", REVIEW_DESTINATION, "SP"]
    pd.testing.assert_frame_equal(focus, before)
    shuffled = summarize_destination_states(focus.sample(frac=1, random_state=17))
    for name in profiles:
        pd.testing.assert_frame_equal(profiles[name], shuffled[name])


def test_destination_profiles_handle_empty_scope_and_no_late_orders():
    items, orders, routes = focus_fixture()
    focus, _ = build_focus_cohort(items, orders, routes)
    focus["is_late"] = False
    profiles = summarize_destination_states(focus)
    assert profiles["all_routes"]["late_share_pct"].isna().all()
    assert profiles["all_routes"]["late_rate_pct"].eq(0).all()
    focus["route_type"] = "Intra-state"
    assert summarize_destination_states(focus)["inter_state"].empty
    assert all(table.empty for table in summarize_destination_states(focus.iloc[:0]).values())
    with pytest.raises(ValueError, match="one row"):
        summarize_destination_states(pd.concat([focus, focus.iloc[[0]]]))


def test_destination_figure_ranks_known_states_and_renders_empty_results():
    import matplotlib.pyplot as plt
    from src.analysis.investigation_plots import plot_destination_states

    table = pd.DataFrame({
        "customer_state": ["SP", REVIEW_DESTINATION, "RJ"],
        "orders": [100, 100, 20], "late_orders": [2, 99, 3],
        "late_rate_pct": [2., 99., 15.],
    })
    before = table.copy(deep=True)
    figure = plot_destination_states(table, top_n=1)
    try:
        figure.canvas.draw()
        assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == ["RJ (N=20)"]
        assert figure.axes[0].patches[0].get_width() == 3
        assert figure.axes[1].patches[0].get_width() == 15
    finally:
        plt.close(figure)
    for empty in (table.iloc[:0], table.assign(late_orders=0)):
        figure = plot_destination_states(empty)
        figure.canvas.draw()
        plt.close(figure)
    pd.testing.assert_frame_equal(table, before)


def test_state_pairs_count_each_order_once_and_keep_unresolved_groups():
    focus = pd.DataFrame({
        "order_id": ["single", "multi", "missing", "customer_review", "second", "outside"],
        "items": [2, 2, 2, 1, 1, 1],
        "customer_state": ["RJ", "RJ", "RJ", pd.NA, "SP", "SP"],
        "location_status": ["known", "known", "known", "conflicting", "known", "known"],
        "route_type": ["Inter-state"] * 5 + ["Intra-state"],
        "is_late": [True, True, True, True, False, True],
    })
    items = pd.DataFrame({
        "order_id": ["single", "single", "multi", "multi", "missing", "missing",
                     "customer_review", "second", "outside"],
        "seller_state": ["SP", "SP", "SP", "MG", "SP", None, "SP", "RJ", "SP"],
    })
    original = [frame.copy(deep=True) for frame in (items, focus)]
    result = summarize_state_pairs(items, focus)
    coverage = result["coverage"].set_index("seller_group")
    pairs = result["pairs"].set_index(["seller_state", "customer_state"])
    assert coverage["orders"].to_dict() == {
        "Single seller state": 2, "Multiple seller states": 1, REVIEW_DESTINATION: 2,
    }
    assert coverage["late_orders"].sum() == 4
    assert len(result["orders"]) == 5
    assert pairs.loc[("SP", "RJ"), "orders"] == 1
    assert pairs.loc[("SP", "RJ"), "late_rate_pct"] == 100
    assert pairs.loc[("SP", "RJ"), "late_share_pct"] == 25
    assert result["orders"].set_index("order_id").loc["multi", "seller_group"] == "Multiple seller states"
    assert pd.isna(result["orders"].set_index("order_id").loc["multi", "seller_state"])
    shuffled = summarize_state_pairs(items.sample(frac=1, random_state=17), focus)
    pd.testing.assert_frame_equal(result["pairs"], shuffled["pairs"])
    for frame, before in zip((items, focus), original):
        pd.testing.assert_frame_equal(frame, before)


def test_state_pairs_handle_empty_and_zero_late_cohorts():
    focus = pd.DataFrame({
        "order_id": ["A"], "customer_state": ["RJ"], "location_status": ["known"],
        "route_type": ["Inter-state"], "is_late": [False], "items": [1],
    })
    items = pd.DataFrame({"order_id": ["A"], "seller_state": ["SP"]})
    result = summarize_state_pairs(items, focus)
    assert result["pairs"]["late_share_pct"].isna().all()
    assert result["coverage"]["orders"].sum() == 1
    empty = summarize_state_pairs(items, focus.iloc[:0])
    assert empty["pairs"].empty
    assert empty["coverage"]["orders"].sum() == 0
    with pytest.raises(ValueError, match="no source items"):
        summarize_state_pairs(items.iloc[:0], focus)


def test_state_pair_figure_uses_count_ranking_and_handles_empty_results():
    import matplotlib.pyplot as plt
    from src.analysis.investigation_plots import plot_state_pairs

    pairs = pd.DataFrame({
        "seller_state": ["SP", "MG"], "customer_state": ["RJ", "BA"],
        "orders": [100, 10], "late_orders": [5, 3], "late_rate_pct": [5.0, 30.0],
    })
    figure = plot_state_pairs(pairs, 8.0, top_n=1)
    try:
        figure.canvas.draw()
        assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == ["SP → RJ (N=100)"]
        assert figure.axes[0].patches[0].get_width() == 5
        assert figure.axes[1].patches[0].get_width() == 5
        assert figure.axes[1].get_xlim()[0] <= 8.0 <= figure.axes[1].get_xlim()[1]
    finally:
        plt.close(figure)
    figure = plot_state_pairs(pairs.iloc[:0], float("nan"))
    figure.canvas.draw()
    plt.close(figure)


def test_focus_time_preserves_cohorts_and_empty_calendar_groups():
    inter = pd.DataFrame({
        "order_id": ["A", "B", "C"],
        "seller_group": ["Single seller state"] * 3,
        "seller_state": ["SP", "SP", "MG"],
        "customer_state": ["RJ", "RJ", "BA"],
        "route_type": ["Inter-state"] * 3,
        "is_late": [True, False, True],
        "order_purchase_timestamp": pd.to_datetime(["2017-01-03", "2017-03-02", "2018-01-05"]),
        "approval_hour": [12, 17, 12], "approval_weekday": [1, 3, 4],
    })
    before = inter.copy(deep=True)
    result = summarize_focus_time(inter, ("SP", "RJ"))
    for key in ("monthly", "hours", "weekdays"):
        table = result[key]
        totals = table.groupby("scope")[["orders", "late_orders"]].sum()
        assert totals.loc["All inter-state orders"].tolist() == [3, 2]
        assert totals.loc["SP → RJ"].tolist() == [2, 1]
    february = result["monthly"].loc[result["monthly"]["purchase_month"].eq("2017-02")]
    assert february["orders"].eq(0).all()
    assert february["late_rate_pct"].isna().all()
    matched = result["matched_years"].set_index(["scope", "year"])
    assert matched.loc[("SP → RJ", 2017), "orders"] == 2
    assert matched.loc[("SP → RJ", 2018), "orders"] == 0
    assert pd.isna(matched.loc[("SP → RJ", 2018), "late_rate_pct"])
    assert len(result["hours"]) == 12
    assert len(result["weekdays"]) == 14
    pd.testing.assert_frame_equal(inter, before)


def test_focus_time_handles_no_leading_pair_and_empty_interstate_cohort():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.investigation_plots import plot_focus_months, plot_focus_approval_time

    inter = pd.DataFrame({
        "order_id": ["A"], "seller_group": ["Multiple seller states"],
        "seller_state": [pd.NA], "customer_state": ["RJ"], "route_type": ["Inter-state"],
        "is_late": [False], "order_purchase_timestamp": pd.to_datetime(["2017-01-03"]),
        "approval_hour": [12], "approval_weekday": [1],
    })
    result = summarize_focus_time(inter, None)
    assert result["monthly"]["scope"].unique().tolist() == ["All inter-state orders"]
    assert summarize_focus_time(inter.iloc[:0], None)["monthly"].empty
    with pytest.raises(ValueError, match="no eligible orders"):
        summarize_focus_time(inter, ("SP", "RJ"))
    empty = summarize_focus_time(inter.iloc[:0], None)
    figures = [plot_focus_months(empty["monthly"]),
               plot_focus_approval_time(empty["hours"], empty["weekdays"])]
    try:
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)


def test_focus_time_figures_render_with_missing_months():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.investigation_plots import plot_focus_months, plot_focus_approval_time

    inter = pd.DataFrame({
        "order_id": ["A", "B"], "seller_group": ["Single seller state"] * 2,
        "seller_state": ["SP"] * 2, "customer_state": ["RJ"] * 2,
        "route_type": ["Inter-state"] * 2, "is_late": [True, False],
        "order_purchase_timestamp": pd.to_datetime(["2017-01-03", "2017-03-02"]),
        "approval_hour": [12, 17], "approval_weekday": [1, 3],
    })
    result = summarize_focus_time(inter, ("SP", "RJ"))
    figures = [plot_focus_months(result["monthly"]),
               plot_focus_approval_time(result["hours"], result["weekdays"])]
    try:
        for figure in figures:
            figure.canvas.draw()
    finally:
        for figure in figures:
            plt.close(figure)


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



def test_state_pairs_reject_partial_item_input():
    focus = pd.DataFrame({"order_id": ["A"], "items": [2], "customer_state": ["RJ"],
                          "location_status": ["known"], "route_type": ["Inter-state"],
                          "is_late": [True]})
    items = pd.DataFrame({"order_id": ["A", "A"], "seller_state": ["SP", "MG"]})
    assert summarize_state_pairs(items, focus)["orders"].iloc[0]["seller_group"] == "Multiple seller states"
    with pytest.raises(ValueError, match="complete cleaned item table"):
        summarize_state_pairs(items.iloc[:1], focus)


def test_matched_years_use_purchase_date_at_july_boundary():
    frame = pd.DataFrame({
        "order_id": ["A", "B"], "seller_group": ["Single seller state"] * 2,
        "seller_state": ["SP"] * 2, "customer_state": ["RJ"] * 2,
        "route_type": ["Inter-state"] * 2, "is_late": [True, False],
        "order_purchase_timestamp": pd.to_datetime(["2018-07-31 23:00", "2018-08-01 11:00"]),
        "order_approved_at": pd.to_datetime(["2018-08-01 12:00", "2018-08-01 13:00"]),
        "approval_hour": [12, 13], "approval_weekday": [2, 2],
    })
    table = summarize_focus_time(frame, ("SP", "RJ"))["matched_years"]
    selected = table.loc[table["year"].eq(2018)]
    assert selected["orders"].tolist() == [1, 1]
    assert selected["late_orders"].tolist() == [1, 1]


def test_focus_comparisons_use_disjoint_scopes_and_weighted_period_rates():
    frame = pd.DataFrame({
        "order_id": list("ABCDE"), "seller_group": ["Single seller state"] * 5,
        "seller_state": ["SP", "SP", "SP", "MG", "MG"], "customer_state": ["RJ"] * 5,
        "route_type": ["Inter-state"] * 5, "is_late": [True, False, False, False, True],
        "order_purchase_timestamp": pd.to_datetime([
            "2017-11-01", "2017-12-01", "2017-12-02", "2017-11-01", "2018-04-01",
        ]),
    })
    before = frame.copy(deep=True)
    result = summarize_focus_comparisons(frame, ("SP", "RJ"))
    assert result["same_destination"]["orders"].tolist() == [3, 2]
    period = result["periods"].iloc[0]
    assert (period["orders"], period["late_orders"]) == (3, 1)
    assert period["late_rate_pct"] == pytest.approx(100 / 3)
    assert result["concentration"].iloc[0]["late_share_pct"] == 100
    assert result["concentration"].iloc[0]["order_share_pct"] == 100
    assert summarize_focus_comparisons(frame, None)["periods"].empty
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("leading_pair", [("SP", "RJ"), None])
@pytest.mark.parametrize("defect, message", [
    ("duplicate", "one row per non-null order_id"),
    ("missing_id", "one row per non-null order_id"),
    ("intra_state", "only inter-state orders"),
    ("missing_route", "only inter-state orders"),
    ("missing_purchase", "must be complete"),
    ("missing_outcome", "must be complete"),
    ("missing_column", "Missing focus comparison columns"),
])
def test_focus_comparisons_reject_invalid_cohorts(defect, message, leading_pair):
    frame = pd.DataFrame({
        "order_id": ["A", "B"], "seller_group": ["Single seller state"] * 2,
        "seller_state": ["SP", "MG"], "customer_state": ["RJ"] * 2,
        "route_type": pd.Series(["Inter-state"] * 2, dtype="string"),
        "is_late": pd.Series([True, False], dtype="boolean"),
        "order_purchase_timestamp": pd.to_datetime(["2017-11-01", "2017-12-01"]),
    })
    if defect == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif defect == "missing_id":
        frame.loc[1, "order_id"] = pd.NA
    elif defect == "intra_state":
        frame.loc[1, "route_type"] = "Intra-state"
    elif defect == "missing_route":
        frame.loc[1, "route_type"] = pd.NA
    elif defect == "missing_purchase":
        frame.loc[1, "order_purchase_timestamp"] = pd.NaT
    elif defect == "missing_outcome":
        frame.loc[1, "is_late"] = pd.NA
    else:
        frame = frame.drop(columns="seller_state")
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match=message):
        summarize_focus_comparisons(frame, leading_pair)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("defect", ["duplicate", "missing"])
def test_state_pairs_reject_invalid_item_keys(defect):
    focus = pd.DataFrame({
        "order_id": ["A"], "items": [2], "customer_state": ["RJ"],
        "location_status": ["known"], "route_type": ["Inter-state"], "is_late": [True],
    })
    items = pd.DataFrame({
        "order_id": ["A", "A", "outside", "outside"],
        "order_item_id": pd.Series([1, 2, pd.NA, pd.NA], dtype="Int64"),
        "seller_state": ["SP", "MG", "SP", "SP"],
    })
    assert summarize_state_pairs(items, focus)["orders"].iloc[0]["seller_group"] == "Multiple seller states"
    items.loc[1, "order_item_id"] = 1 if defect == "duplicate" else pd.NA
    before = items.copy(deep=True)
    with pytest.raises(ValueError, match="unique, non-null"):
        summarize_state_pairs(items, focus)
    pd.testing.assert_frame_equal(items, before)


@pytest.mark.parametrize("outcome", [2, -1, 0.5, "True", "1"])
def test_focus_comparisons_reject_nonbinary_outcomes(outcome):
    frame = pd.DataFrame({
        "order_id": ["A"], "seller_group": ["Single seller state"],
        "seller_state": ["SP"], "customer_state": ["RJ"], "route_type": ["Inter-state"],
        "is_late": [outcome], "order_purchase_timestamp": pd.to_datetime(["2017-11-01"]),
    })
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="only booleans or numeric 0/1"):
        summarize_focus_comparisons(frame, ("SP", "RJ"))
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("dtype", ["bool", "boolean", "int64", "Int64", "float64"])
def test_focus_comparisons_accept_binary_outcomes(dtype):
    frame = pd.DataFrame({
        "order_id": ["A", "B"], "seller_group": ["Single seller state"] * 2,
        "seller_state": ["SP"] * 2, "customer_state": ["RJ"] * 2,
        "route_type": ["Inter-state"] * 2,
        "is_late": pd.Series([1, 0], dtype=dtype),
        "order_purchase_timestamp": pd.to_datetime(["2017-11-01", "2017-11-02"]),
    })
    before = frame.copy(deep=True)
    row = summarize_focus_comparisons(frame, ("SP", "RJ"))["same_destination"].iloc[0]
    assert (row["orders"], row["late_orders"], row["late_rate_pct"]) == (2, 1, 50)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("dtype", ["Int64", "float64", "object"])
def test_state_pairs_reject_missing_reference_item_counts(dtype):
    focus = pd.DataFrame({
        "order_id": ["A", "B"], "items": pd.Series([None, None], dtype=dtype),
        "customer_state": ["RJ", "SP"], "location_status": ["known"] * 2,
        "route_type": ["Inter-state", "Intra-state"], "is_late": [True, False],
    })
    items = pd.DataFrame({"order_id": ["A"], "order_item_id": [1], "seller_state": ["SP"]})
    before = focus.copy(deep=True)
    with pytest.raises(ValueError, match="Reference item counts must be complete"):
        summarize_state_pairs(items, focus)
    pd.testing.assert_frame_equal(focus, before)
    focus.loc[0, "items"] = 1
    result = summarize_state_pairs(items, focus)
    assert result["pairs"]["orders"].sum() == 1


def test_weekday_display_preserves_numeric_codes():
    from src.analysis.presentation import format_analysis_table
    table = pd.DataFrame({"approval_weekday": [0, 4, 6]})
    html = format_analysis_table(table).to_html()
    assert all(day in html for day in ("Mon", "Fri", "Sun"))
    assert table["approval_weekday"].tolist() == [0, 4, 6]



def test_route_states_use_consistent_whitespace_normalization():
    items = pd.DataFrame({"order_id": ["A"], "customer_state": ["SP"], "seller_state": [" SP "]})
    orders = pd.DataFrame({"order_id": ["A"]})
    before = items.copy(deep=True)
    routes = classify_order_routes(items, orders)
    assert routes.iloc[0]["route_type"] == "Intra-state"
    pd.testing.assert_frame_equal(items, before)


def overnight_route_fixture():
    definitions = [
        ("inter", [" SP ", "SP"], [" RJ ", "RJ"], True),
        ("intra", ["SP", "SP"], ["SP", "SP"], False),
        ("multi", ["SP", "MG"], ["RJ", "RJ"], True),
        ("mixed", ["SP", "RJ"], ["RJ", "RJ"], False),
        ("unknown", ["SP", None], ["RJ", "RJ"], True),
        ("conflict", ["SP", "SP"], ["RJ", "MG"], False),
        ("other_intra", ["MG"], ["MG"], True),
    ]
    rows = []
    for order_id, sellers, customers, late in definitions:
        for item_id, (seller, customer) in enumerate(zip(sellers, customers), start=1):
            rows.append({"order_id": order_id, "order_item_id": item_id,
                         "seller_state": seller, "customer_state": customer})
    items = pd.DataFrame(rows)
    orders = pd.DataFrame({
        "order_id": [entry[0] for entry in definitions],
        "approval_slot": [SLOT_LABELS[0]] * len(definitions),
        "is_late": [entry[3] for entry in definitions],
        "order_purchase_timestamp": pd.to_datetime(["2018-02-01"] * len(definitions)),
    })
    routes = classify_order_routes(items, orders)
    focus, _ = build_focus_cohort(items, orders, routes, approval_slot=SLOT_LABELS[0])
    return items, focus


def test_focus_route_summary_preserves_all_route_types_and_scope_shares():
    items, focus = overnight_route_fixture()
    before = focus.copy(deep=True)
    table = summarize_focus_routes(focus).set_index("route_type")
    assert table["orders"].to_dict() == {
        "Intra-state": 2, "Inter-state": 3, "Mixed": 1, "Unknown": 1,
    }
    assert table["late_orders"].to_dict() == {
        "Intra-state": 1, "Inter-state": 2, "Mixed": 0, "Unknown": 1,
    }
    assert table["orders"].sum() == 7
    assert table["late_orders"].sum() == 4
    assert table.loc["Inter-state", "late_rate_pct"] == pytest.approx(200 / 3)
    assert table.loc["Inter-state", "order_share_pct"] == pytest.approx(300 / 7)
    assert table.loc["Unknown", "late_share_pct"] == 25
    assert table["order_share_pct"].sum() == pytest.approx(100)
    assert table["late_share_pct"].sum() == pytest.approx(100)
    pd.testing.assert_frame_equal(focus, before)


def test_expanded_pairs_cover_all_orders_without_splitting_mixed_or_multistate_orders():
    items, focus = overnight_route_fixture()
    originals = [frame.copy(deep=True) for frame in (items, focus)]
    result = summarize_state_pairs(items, focus, include_intra_state=True)
    coverage = result["coverage"].set_index("seller_group")
    pairs = result["pairs"].set_index(["seller_state", "customer_state"])
    assert coverage["orders"].to_dict() == {
        "Single seller state": 3, "Multiple seller states": 2, REVIEW_DESTINATION: 2,
    }
    assert coverage["late_orders"].to_dict() == {
        "Single seller state": 2, "Multiple seller states": 1, REVIEW_DESTINATION: 1,
    }
    assert coverage["orders"].sum() == len(focus)
    assert coverage["late_share_pct"].sum() == pytest.approx(100)
    assert pairs["orders"].sum() == 3
    assert pairs["order_share_pct"].sum() == pytest.approx(300 / 7)
    assert pairs["late_share_pct"].sum() == pytest.approx(50)
    assert pairs.loc[("SP", "RJ"), "late_share_pct"] == 25
    assert pairs.loc[("SP", "SP"), "orders"] == 1
    assert result["orders"]["order_id"].is_unique
    by_id = result["orders"].set_index("order_id")
    assert by_id.loc["mixed", "seller_group"] == "Multiple seller states"
    assert by_id.loc["unknown", "seller_group"] == REVIEW_DESTINATION
    assert by_id.loc["conflict", "seller_group"] == REVIEW_DESTINATION
    assert by_id.loc[["mixed", "unknown", "conflict"], "seller_state"].isna().all()
    shuffled = summarize_state_pairs(items.sample(frac=1, random_state=7), focus,
                                      include_intra_state=True)
    pd.testing.assert_frame_equal(result["pairs"], shuffled["pairs"])
    for frame, before in zip((items, focus), originals):
        pd.testing.assert_frame_equal(frame, before)


def test_expanded_pairs_preserve_default_interstate_scope_and_columns():
    items, focus = overnight_route_fixture()
    default = summarize_state_pairs(items, focus)
    explicit = summarize_state_pairs(items, focus, include_intra_state=False)
    for key in default:
        pd.testing.assert_frame_equal(default[key], explicit[key])
    assert set(default["orders"]["order_id"]) == {"inter", "multi", "conflict"}
    assert default["pairs"].columns.tolist() == [
        "seller_state", "customer_state", "orders", "late_orders", "late_rate_pct", "late_share_pct",
    ]
    assert default["coverage"].columns.tolist() == [
        "seller_group", "orders", "late_orders", "late_rate_pct",
    ]


def test_overnight_summaries_keep_empty_and_zero_late_denominators_explicit():
    items, focus = overnight_route_fixture()
    empty = summarize_focus_routes(focus.iloc[:0])
    assert len(empty) == 4
    assert empty["orders"].eq(0).all()
    assert empty[["late_rate_pct", "order_share_pct", "late_share_pct"]].isna().all().all()
    expanded = summarize_state_pairs(items, focus.iloc[:0], include_intra_state=True)
    assert expanded["orders"].empty and expanded["pairs"].empty
    assert len(expanded["coverage"]) == 3
    assert expanded["coverage"]["orders"].eq(0).all()
    assert expanded["coverage"][["late_rate_pct", "order_share_pct", "late_share_pct"]].isna().all().all()

    focus["is_late"] = False
    for table in [summarize_focus_routes(focus),
                  *[summarize_state_pairs(items, focus, include_intra_state=True)[key]
                    for key in ("coverage", "pairs")]]:
        assert table["late_orders"].eq(0).all()
        assert table["late_share_pct"].isna().all()
        assert table.loc[table["orders"].gt(0), "late_rate_pct"].eq(0).all()
    single = summarize_focus_routes(focus.iloc[[0]])
    assert single.loc[single["orders"].eq(0), "late_rate_pct"].isna().all()


@pytest.mark.parametrize("defect", ["partial", "duplicate", "null_key", "wrong_route", "customer_mismatch"])
def test_expanded_pairs_reject_incomplete_or_inconsistent_intra_state_items(defect):
    items, focus = overnight_route_fixture()
    if defect == "partial":
        items = items.loc[~(items["order_id"].eq("intra") & items["order_item_id"].eq(2))]
        message = "complete cleaned item table"
    elif defect in {"duplicate", "null_key"}:
        items.loc[items["order_id"].eq("intra") & items["order_item_id"].eq(2),
                  "order_item_id"] = 1 if defect == "duplicate" else pd.NA
        message = "unique, non-null"
    elif defect == "wrong_route":
        focus["route_type"] = focus["route_type"].astype("string")
        focus.loc[focus["order_id"].eq("intra"), "route_type"] = "Inter-state"
        message = "conflicts with the route classification"
    else:
        items.loc[items["order_id"].eq("intra") & items["order_item_id"].eq(2),
                  "customer_state"] = "MG"
        message = "match every source item"
    with pytest.raises(ValueError, match=message):
        summarize_state_pairs(items, focus, include_intra_state=True)


@pytest.mark.parametrize("defect,message", [
    ("duplicate", "one row per non-null"),
    ("null_id", "one row per non-null"),
    ("route", "unrecognized route"),
    ("outcome", "known and binary"),
])
def test_focus_routes_reject_invalid_order_inputs(defect, message):
    _, focus = overnight_route_fixture()
    if defect == "duplicate":
        focus = pd.concat([focus, focus.iloc[:1]], ignore_index=True)
    elif defect == "null_id":
        focus.loc[0, "order_id"] = pd.NA
    elif defect == "route":
        focus["route_type"] = focus["route_type"].astype("string")
        focus.loc[0, "route_type"] = "Other"
    else:
        focus["is_late"] = focus["is_late"].astype("object")
        focus.loc[0, "is_late"] = 2
    with pytest.raises(ValueError, match=message):
        summarize_focus_routes(focus)


@pytest.mark.parametrize("count_dtype", ["int64", "Int8"])
def test_focus_route_figure_shows_full_scope_and_undefined_empty_groups(count_dtype):
    import matplotlib.pyplot as plt
    from src.analysis.investigation_plots import plot_focus_routes

    _, focus = overnight_route_fixture()
    table = summarize_focus_routes(focus).astype({"orders": count_dtype, "late_orders": count_dtype})
    before = table.copy(deep=True)
    figure = plot_focus_routes(table, SLOT_LABELS[0])
    try:
        figure.canvas.draw()
        assert [p.get_height() for p in figure.axes[0].patches] == [1, 1, 1, 0, 1, 2, 0, 1]
        assert [p.get_height() for p in figure.axes[1].patches] == pytest.approx([50, 200/3, 0, 100])
        assert "00:00–06:00" in figure._suptitle.get_text()
        pd.testing.assert_frame_equal(table, before)
    finally:
        plt.close(figure)
    empty = plot_focus_routes(summarize_focus_routes(focus.iloc[:0]), SLOT_LABELS[0])
    try:
        empty.canvas.draw()
        assert [text.get_text() for text in empty.axes[1].texts] == ["No orders"] * 4
    finally:
        plt.close(empty)
    bad = table.copy()
    bad.loc[0, "late_rate_pct"] = 99
    with pytest.raises(ValueError, match="matching route order denominators"):
        plot_focus_routes(bad, SLOT_LABELS[0])


@pytest.mark.parametrize("slot", [SLOT_LABELS[0], SLOT_LABELS[2]])
@pytest.mark.parametrize("defect", ["duplicate_replacement", "null_key"])
def test_focus_cohort_rejects_invalid_item_keys_before_destination_assignment(slot, defect):
    items, orders, routes = focus_fixture()
    items["order_item_id"] = items.groupby("order_id").cumcount() + 1
    orders.loc[orders.order_id.eq("conflicting"), "approval_slot"] = slot
    target = items.index[items.order_id.eq("conflicting")]
    if defect == "duplicate_replacement":
        # Preserve row count while replacing the second destination with item 1.
        items.loc[target[1]] = items.loc[target[0]]
    else:
        items.loc[target[1], "order_item_id"] = None
    originals = [frame.copy(deep=True) for frame in (items, orders, routes)]
    with pytest.raises(ValueError, match="unique, non-null.*order_item_id"):
        build_focus_cohort(items, orders, routes, approval_slot=slot)
    for frame, original in zip((items, orders, routes), originals):
        pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("value", [None, pd.NA, 2, -1, "1", float("inf")])
@pytest.mark.parametrize("slot", [SLOT_LABELS[0], SLOT_LABELS[2]])
def test_focus_cohort_rejects_missing_or_nonbinary_outcomes(value, slot):
    items, orders, routes = focus_fixture()
    orders["is_late"] = orders["is_late"].astype(object)
    target = "outside" if slot == SLOT_LABELS[0] else "late"
    orders.loc[orders.order_id.eq(target), "is_late"] = value
    before = orders.copy(deep=True)
    with pytest.raises(ValueError, match="complete binary delivery outcomes"):
        build_focus_cohort(items, orders, routes, approval_slot=slot)
    pd.testing.assert_frame_equal(orders, before)


@pytest.mark.parametrize("value", [False, True, 0, 1, 0.0, 1.0])
def test_focus_cohort_accepts_binary_boolean_and_numeric_outcomes(value):
    items, orders, routes = focus_fixture()
    orders["is_late"] = value
    before = orders.copy(deep=True)
    focus, diagnostics = build_focus_cohort(items, orders, routes)
    summary = diagnostics["summary"].iloc[0]
    assert summary["orders"] == 5
    assert summary["late_orders"] == 5 * int(value)
    assert summary["on_time_orders"] == 5 * (1 - int(value))
    assert summary["late_rate_pct"] == 100 * int(value)
    pd.testing.assert_frame_equal(orders, before)
    assert focus["is_late"].dtype == orders["is_late"].dtype


def test_focus_cohort_keeps_reduced_item_contract_and_input_frames():
    items, orders, routes = focus_fixture()
    keyed = items.assign(order_item_id=items.groupby("order_id").cumcount() + 1)
    originals = [frame.copy(deep=True) for frame in (items, keyed, orders, routes)]
    reduced, reduced_details = build_focus_cohort(items, orders, routes)
    complete, complete_details = build_focus_cohort(keyed, orders, routes)
    pd.testing.assert_frame_equal(reduced, complete)
    assert complete.set_index("order_id").loc["conflicting", "location_status"] == "conflicting"
    for name in reduced_details:
        pd.testing.assert_frame_equal(reduced_details[name], complete_details[name])
    for frame, original in zip((items, keyed, orders, routes), originals):
        pd.testing.assert_frame_equal(frame, original)


def test_focus_cohort_still_rejects_partial_items_with_unique_keys():
    items, orders, routes = focus_fixture()
    items["order_item_id"] = items.groupby("order_id").cumcount() + 1
    partial = items.drop(items.index[items.order_id.eq("known")][0])
    with pytest.raises(ValueError, match="Route item counts differ"):
        build_focus_cohort(partial, orders, routes)


def test_focus_cohort_checks_only_selected_item_keys_and_outcomes():
    items, orders, routes = focus_fixture()
    expected, expected_details = build_focus_cohort(items, orders, routes)
    items["order_item_id"] = items.groupby("order_id").cumcount() + 1
    outside = items.loc[items.order_id.eq("outside")].copy()
    outside["order_item_id"] = None
    items = pd.concat([items, outside], ignore_index=True)
    orders["is_late"] = orders["is_late"].astype(object)
    orders.loc[orders.order_id.eq("outside"), "is_late"] = None
    actual, actual_details = build_focus_cohort(items, orders, routes)
    # Object dtype is retained from the caller; values and diagnostics agree.
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    for name in expected_details:
        pd.testing.assert_frame_equal(actual_details[name], expected_details[name])
    empty, details = build_focus_cohort(items, orders, routes, approval_slot=SLOT_LABELS[1])
    assert empty.empty and details["summary"].iloc[0]["orders"] == 0
    assert pd.isna(details["summary"].iloc[0]["late_rate_pct"])
