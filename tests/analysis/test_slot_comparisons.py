"""Independent selection, denominator and boundary checks for two-slot comparisons."""

import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.orders import SLOT_LABELS
from src.analysis.slot_comparisons import compare_approval_slot_groups, COMPARISON_SLOTS
from src.analysis.investigation_plots import plot_slot_group_comparison
from src.analysis.presentation import format_analysis_table


def cohort(specifications, slot, prefix):
    rows = []
    for seller, customer, size, late in specifications:
        for index in range(size):
            rows.append({
                "order_id": f"{prefix}-{len(rows)}", "approval_slot": slot,
                "is_late": index < late, "customer_state": customer,
                "seller_state": seller, "location_status": "known",
                "route_type": "Intra-state" if seller == customer else "Inter-state",
                "seller_group": "Single seller state",
            })
    return pd.DataFrame(rows, columns=[
        "order_id", "approval_slot", "is_late", "customer_state", "seller_state",
        "location_status", "route_type", "seller_group",
    ])


def test_selection_unites_both_slot_rankings_reference_and_keeps_small_groups():
    night = cohort([("AA", "RJ", 1000, 90), ("BB", "RJ", 100, 40), ("CC", "RJ", 99, 60),
                    ("DD", "RJ", 900, 80), ("EE", "RJ", 100, 20), ("SP", "RJ", 1, 0)],
                   COMPARISON_SLOTS[0], "night")
    afternoon = cohort([("AA", "RJ", 1000, 70), ("BB", "RJ", 100, 5), ("CC", "RJ", 99, 55),
                        ("DD", "RJ", 900, 85), ("EE", "RJ", 100, 30), ("SP", "RJ", 1, 0)],
                       COMPARISON_SLOTS[1], "afternoon")
    originals = [frame.copy(deep=True) for frame in (night, afternoon)]
    result = compare_approval_slot_groups(night, afternoon, group_by="state_pair", top_n=1, rate_top_n=1)
    reasons = result["selection"].set_index("group")["selection_reason"]
    assert set(reasons.index) == {"AA → RJ", "BB → RJ", "DD → RJ", "EE → RJ", "SP → RJ"}
    assert "Top late count: 00:00–06:00" in reasons["AA → RJ"]
    assert "Top late count: 12:00–18:00" in reasons["DD → RJ"]
    assert "Top late rate" in reasons["BB → RJ"]
    assert "Largest absolute gap" in reasons["BB → RJ"]
    assert "Reference state pair" in reasons["SP → RJ"]
    small = result["all_groups"].loc[result["all_groups"]["group"].eq("CC → RJ")]
    assert small["orders"].tolist() == [99, 99]
    remaining = result["plot_data"].loc[result["plot_data"]["group"].eq("Other eligible pairs")]
    assert remaining["orders"].tolist() == [99, 99]
    assert remaining["late_orders"].tolist() == [60, 55]
    full = result["all_groups"]
    assert full.loc[full["group"].eq("BB → RJ"), "difference_pp"].tolist() == [35, 35]
    for frame, original in zip((night, afternoon), originals):
        pd.testing.assert_frame_equal(frame, original)
    shuffled = compare_approval_slot_groups(night.sample(frac=1, random_state=8),
                                            afternoon.sample(frac=1, random_state=9),
                                            group_by="state_pair", top_n=1, rate_top_n=1)
    for key in result:
        pd.testing.assert_frame_equal(result[key], shuffled[key])


def test_count_leaders_can_have_less_than_minimum_rate_ranking_size():
    night = cohort([("AA", "RJ", 50, 30), ("BB", "RJ", 100, 2)], COMPARISON_SLOTS[0], "n")
    afternoon = cohort([("BB", "RJ", 100, 1)], COMPARISON_SLOTS[1], "a")
    result = compare_approval_slot_groups(night, afternoon, group_by="state_pair",
                                          top_n=1, rate_top_n=1, reference_pair=None)
    reason = result["selection"].set_index("group").loc["AA → RJ", "selection_reason"]
    assert "Top late count" in reason and "Top late rate" not in reason
    absent = result["selected"].loc[result["selected"]["group"].eq("AA → RJ")
                                   & result["selected"]["approval_slot"].eq(COMPARISON_SLOTS[1])].iloc[0]
    assert absent["orders"] == absent["late_orders"] == 0
    assert pd.isna(absent["late_rate_pct"]) and pd.isna(absent["difference_pp"])
    assert pd.isna(absent["wilson_lower_pct"]) and pd.isna(absent["wilson_upper_pct"])


def test_tie_breaks_use_counts_sizes_codes_and_unrounded_rates():
    specs = [("AA", "RJ", 100, 10), ("BB", "RJ", 100, 10), ("CC", "RJ", 100, 10)]
    night = cohort(specs, COMPARISON_SLOTS[0], "n")
    afternoon = cohort(specs, COMPARISON_SLOTS[1], "a")
    result = compare_approval_slot_groups(night, afternoon, group_by="state_pair",
                                          top_n=1, rate_top_n=1, reference_pair=None)
    assert result["selection"]["group"].tolist() == ["AA → RJ"]
    assert "Largest absolute gap" not in result["selection"].iloc[0]["selection_reason"]
    specs = [("AA", "RJ", 1001, 1), ("ZZ", "RJ", 1000, 1)]
    result = compare_approval_slot_groups(cohort(specs, COMPARISON_SLOTS[0], "n"),
                                          cohort(specs, COMPARISON_SLOTS[1], "a"),
                                          group_by="state_pair", top_n=1, rate_top_n=1, reference_pair=None)
    reasons = result["selection"].set_index("group")["selection_reason"]
    assert "Top late count" in reasons["AA → RJ"]
    assert "Top late rate" in reasons["ZZ → RJ"]
    assert "Top late rate" not in reasons["AA → RJ"]


def test_largest_absolute_gap_uses_pooled_size_then_code_and_keeps_sign():
    night = cohort([("AA", "RJ", 100, 10), ("BB", "RJ", 200, 40), ("CC", "RJ", 200, 20)],
                   COMPARISON_SLOTS[0], "n")
    afternoon = cohort([("AA", "RJ", 100, 20), ("BB", "RJ", 200, 20), ("CC", "RJ", 200, 40)],
                       COMPARISON_SLOTS[1], "a")
    result = compare_approval_slot_groups(night, afternoon, group_by="state_pair",
                                          top_n=1, rate_top_n=1, reference_pair=None)
    reasons = result["selection"].set_index("group")["selection_reason"]
    assert "Largest absolute gap" in reasons["BB → RJ"]
    assert all("Largest absolute gap" not in value for key, value in reasons.items() if key != "BB → RJ")
    differences = result["all_groups"].groupby("group")["difference_pp"].first()
    assert differences["AA → RJ"] == -10
    assert differences["BB → RJ"] == 10


@pytest.mark.parametrize("group_by", ["customer_state", "state_pair"])
def test_scope_totals_include_unresolved_and_multi_state_orders(group_by):
    night = cohort([("SP", "RJ", 2, 1), ("SP", "SP", 1, 1)], COMPARISON_SLOTS[0], "n")
    afternoon = cohort([("SP", "RJ", 2, 0)], COMPARISON_SLOTS[1], "a")
    extra = night.iloc[[0, 0]].copy()
    extra["order_id"] = ["multi", "review"]
    extra["seller_group"] = ["Multiple seller states", "Requires review"]
    extra["seller_state"] = pd.NA
    extra["customer_state"] = ["RJ", pd.NA]
    extra["location_status"] = ["known", "missing"]
    extra["route_type"] = ["Mixed", "Unknown"]
    extra["is_late"] = [True, False]
    night = pd.concat([night, extra], ignore_index=True)
    result = compare_approval_slot_groups(night, afternoon, group_by=group_by)
    assert result["scope_summary"]["orders"].tolist() == [5, 2]
    assert result["scope_summary"]["late_orders"].tolist() == [3, 0]
    for slot, frame in zip(COMPARISON_SLOTS, (night, afternoon)):
        for key in ("all_groups", "plot_data"):
            rows = result[key].loc[result[key]["approval_slot"].eq(slot)]
            assert rows["orders"].sum() == len(frame)
            assert rows["late_orders"].sum() == int(frame["is_late"].sum())
            assert rows["order_share_pct"].sum() == pytest.approx(100)
            if slot == COMPARISON_SLOTS[0]:
                assert rows["late_share_pct"].sum() == pytest.approx(100)
            else:
                assert rows["late_share_pct"].isna().all()
    if group_by == "state_pair":
        rows = result["all_groups"].set_index(["group", "approval_slot"])
        assert rows.loc[("SP → RJ", COMPARISON_SLOTS[0]), "late_share_pct"] == pytest.approx(100 / 3)
        assert rows.loc[("Multiple seller states", COMPARISON_SLOTS[0]), "orders"] == 1


def test_empty_and_zero_late_cohorts_keep_schema_and_undefined_rates():
    empty_night = cohort([], COMPARISON_SLOTS[0], "n")
    empty_afternoon = cohort([], COMPARISON_SLOTS[1], "a")
    for group_by in ("customer_state", "state_pair"):
        result = compare_approval_slot_groups(empty_night, empty_afternoon, group_by=group_by)
        assert result["selected"].empty and result["selection"].empty
        assert result["all_groups"]["orders"].eq(0).all()
        assert result["all_groups"][["late_rate_pct", "difference_pp", "wilson_upper_pct",
                                     "order_share_pct", "late_share_pct"]].isna().all().all()
    night = cohort([("MG", "SP", 100, 0)], COMPARISON_SLOTS[0], "n")
    result = compare_approval_slot_groups(night, empty_afternoon, group_by="state_pair", reference_pair=None)
    assert result["selection"].empty
    observed = result["all_groups"].loc[result["all_groups"]["orders"].gt(0)].iloc[0]
    assert observed["late_rate_pct"] == 0 and observed["wilson_upper_pct"] > 0


def test_wilson_limits_match_independent_known_binomial_values():
    night = cohort([("SP", "RJ", 100, 10)], COMPARISON_SLOTS[0], "n")
    afternoon = cohort([("SP", "RJ", 100, 100)], COMPARISON_SLOTS[1], "a")
    table = compare_approval_slot_groups(night, afternoon, group_by="customer_state")["selected"]
    assert table.iloc[0]["wilson_lower_pct"] == pytest.approx(5.5229137060)
    assert table.iloc[0]["wilson_upper_pct"] == pytest.approx(17.4365696341)
    assert table.iloc[1]["wilson_upper_pct"] == pytest.approx(100)


@pytest.mark.parametrize("defect,message", [
    ("duplicate", "one row per non-null"), ("null_id", "one row per non-null"),
    ("wrong_slot", "must belong"), ("overlap", "disjoint"), ("outcome", "binary"),
    ("blank_customer", "nonblank"), ("pair_route", "conflict"), ("missing_column", "Missing"),
])
def test_invalid_cohorts_are_rejected(defect, message):
    night = cohort([("SP", "RJ", 2, 1)], COMPARISON_SLOTS[0], "n")
    afternoon = cohort([("SP", "RJ", 2, 1)], COMPARISON_SLOTS[1], "a")
    if defect == "duplicate": night = pd.concat([night, night.iloc[:1]], ignore_index=True)
    elif defect == "null_id": night.loc[0, "order_id"] = pd.NA
    elif defect == "wrong_slot": night.loc[0, "approval_slot"] = SLOT_LABELS[1]
    elif defect == "overlap": afternoon.loc[0, "order_id"] = night.loc[0, "order_id"]
    elif defect == "outcome": night["is_late"] = "True"
    elif defect == "blank_customer": night.loc[0, "customer_state"] = " "
    elif defect == "pair_route": night.loc[0, "route_type"] = "Intra-state"
    else: night = night.drop(columns="seller_group")
    with pytest.raises(ValueError, match=message):
        compare_approval_slot_groups(night, afternoon, group_by="state_pair")


@pytest.mark.parametrize("kwargs", [
    {"top_n": 0}, {"rate_top_n": True}, {"min_orders": 0}, {"min_orders": 1.5},
    {"group_by": "route"}, {"reference_pair": ("SP", "")},
])
def test_selection_options_are_validated(kwargs):
    night = cohort([], COMPARISON_SLOTS[0], "n")
    afternoon = cohort([], COMPARISON_SLOTS[1], "a")
    options = {"group_by": "customer_state", **kwargs}
    with pytest.raises(ValueError):
        compare_approval_slot_groups(night, afternoon, **options)


def test_comparison_figures_and_display_preserve_counts_and_units():
    night = cohort([("SP", "RJ", 100, 10)], COMPARISON_SLOTS[0], "n")
    afternoon = cohort([], COMPARISON_SLOTS[1], "a")
    result = compare_approval_slot_groups(night, afternoon, group_by="state_pair")
    table = result["plot_data"]
    before = table.copy(deep=True)
    figure = plot_slot_group_comparison(table, "state_pair")
    try:
        figure.canvas.draw()
        assert figure.axes[0].patches[0].get_width() == 10
        assert any(text.get_text() == "No orders" for text in figure.axes[1].texts)
        assert "95% Wilson" in figure.axes[1].get_title()
        assert "Delivery outcomes by approval slot" in figure._suptitle.get_text()
    finally:
        plt.close(figure)
    pd.testing.assert_frame_equal(table, before)
    empty_result = compare_approval_slot_groups(
        night.iloc[:0], afternoon, group_by="state_pair", reference_pair=None,
    )
    empty_figure = plot_slot_group_comparison(empty_result["plot_data"], "state_pair")
    try:
        empty_figure.canvas.draw()
        assert all(patch.get_width() == 0 for patch in empty_figure.axes[0].patches)
    finally:
        plt.close(empty_figure)
    html = format_analysis_table(result["selected"]).to_html()
    assert "Night − afternoon (pp)" in html and "Wilson 95% lower" in html
    malformed = table.copy()
    malformed.loc[0, "late_rate_pct"] = 99
    with pytest.raises(ValueError, match="matching order counts"):
        plot_slot_group_comparison(malformed, "state_pair")
    with pytest.raises(ValueError, match="one row"):
        plot_slot_group_comparison(pd.concat([table, table.iloc[:1]]), "state_pair")

def test_pooled_remainder_uses_order_counts_instead_of_average_group_rates():
    night = cohort([("AA", "RJ", 100, 20), ("BB", "RJ", 1, 1), ("CC", "RJ", 2, 0)],
                   COMPARISON_SLOTS[0], "n")
    afternoon = cohort([("AA", "RJ", 100, 10), ("BB", "RJ", 1, 0), ("CC", "RJ", 2, 0)],
                       COMPARISON_SLOTS[1], "a")
    result = compare_approval_slot_groups(night, afternoon, group_by="state_pair",
                                          top_n=1, rate_top_n=1, reference_pair=None)
    remaining = result["plot_data"].loc[result["plot_data"]["group"].eq("Other eligible pairs")]
    assert remaining["orders"].tolist() == [3, 3]
    assert remaining["late_orders"].tolist() == [1, 0]
    assert remaining.iloc[0]["late_rate_pct"] == pytest.approx(100 / 3)
    assert remaining.iloc[0]["late_rate_pct"] != pytest.approx(50)


@pytest.mark.parametrize("group_by", ["customer_state", "state_pair"])
def test_group_state_whitespace_is_normalized_without_changing_inputs(group_by):
    night = cohort([("SP", "RJ", 5, 1)], COMPARISON_SLOTS[0], "n")
    afternoon = cohort([("SP", "RJ", 7, 2)], COMPARISON_SLOTS[1], "a")
    expected = compare_approval_slot_groups(night, afternoon, group_by=group_by)
    night["seller_state"] = " SP "
    night["customer_state"] = " RJ "
    before = night.copy(deep=True)
    actual = compare_approval_slot_groups(night, afternoon, group_by=group_by)
    for key in expected:
        pd.testing.assert_frame_equal(expected[key], actual[key])
    pd.testing.assert_frame_equal(night, before)


def dated_cohort(specifications, slot, prefix):
    pieces = []
    for month, seller, customer, size, late in specifications:
        frame = cohort([(seller, customer, size, late)], slot, f"{prefix}-{month}")
        frame["order_purchase_timestamp"] = pd.Timestamp(f"{month}-15")
        # Purchase month is independent of the recorded approval calendar date.
        frame["order_approved_at"] = pd.Timestamp("2019-01-01 01:00")
        pieces.append(frame)
    if pieces:
        return pd.concat(pieces, ignore_index=True)
    frame = cohort([], slot, prefix)
    frame["order_purchase_timestamp"] = pd.Series(dtype="datetime64[ns]")
    return frame


def test_monthly_comparison_separates_exclusion_from_weighting_and_preserves_inputs():
    from src.analysis.slot_comparisons import summarize_slot_group_months

    night = dated_cohort([
        ("2018-01", "SP", "RJ", 1, 1), ("2018-02", "SP", "RJ", 3, 0),
        ("2018-03", "SP", "RJ", 2, 1),
    ], COMPARISON_SLOTS[0], "n")
    afternoon = dated_cohort([
        ("2018-01", "SP", "RJ", 1, 0), ("2018-02", "SP", "RJ", 1, 1),
        ("2018-04", "SP", "RJ", 2, 1),
    ], COMPARISON_SLOTS[1], "a")
    originals = [frame.copy(deep=True) for frame in (night, afternoon)]
    result = summarize_slot_group_months(night, afternoon, group_by="state_pair")
    table = result["standardized"].set_index("approval_slot")
    assert table["common_months"].tolist() == [2, 2]
    assert table["included_orders"].tolist() == [4, 2]
    assert table["excluded_orders"].tolist() == [2, 2]
    assert table["included_late_orders"].tolist() == [1, 1]
    assert table["excluded_late_orders"].tolist() == [1, 1]
    assert table["raw_late_rate_pct"].tolist() == pytest.approx([100 / 3, 50])
    assert table["common_raw_late_rate_pct"].tolist() == pytest.approx([25, 50])
    assert table["standardized_late_rate_pct"].tolist() == pytest.approx([100 / 3, 200 / 3])
    assert result["weights"]["weight"].tolist() == pytest.approx([1 / 3, 2 / 3])
    assert result["weights"]["pooled_orders"].tolist() == [2, 4]
    assert result["weights"]["weight"].sum() == pytest.approx(1)
    assert len(result["monthly"]) == 8
    absent = result["monthly"].query("purchase_month == '2018-03'").iloc[1]
    assert absent["orders"] == 0 and pd.isna(absent["late_rate_pct"])
    gaps = result["differences"].iloc[0]
    assert gaps["raw_difference_pp"] == pytest.approx(-50 / 3)
    assert gaps["common_raw_difference_pp"] == -25
    assert gaps["standardized_difference_pp"] == pytest.approx(-100 / 3)
    for frame, original in zip((night, afternoon), originals):
        pd.testing.assert_frame_equal(frame, original)
    shuffled = summarize_slot_group_months(night.sample(frac=1, random_state=5),
                                           afternoon.sample(frac=1, random_state=7),
                                           group_by="state_pair")
    for key in result:
        pd.testing.assert_frame_equal(result[key], shuffled[key])


@pytest.mark.parametrize("group_by", ["customer_state", "state_pair"])
def test_monthly_groups_use_single_leaders_with_the_existing_selection_rules(group_by):
    from src.analysis.slot_comparisons import summarize_slot_group_months

    specs = [("SP", "SP", 200, 20), ("SP", "RJ", 100, 30), ("SP", "MA", 100, 15),
             ("SP", "CE", 50, 1), ("PR", "RS", 150, 5), ("MG", "BA", 101, 6)]
    night = cohort(specs, COMPARISON_SLOTS[0], "n")
    afternoon = cohort(specs, COMPARISON_SLOTS[1], "a")
    for frame in (night, afternoon):
        frame["order_purchase_timestamp"] = pd.Timestamp("2018-02-01")
    expected = compare_approval_slot_groups(night, afternoon, group_by=group_by, top_n=1, rate_top_n=1)
    result = summarize_slot_group_months(night, afternoon, group_by=group_by)
    pd.testing.assert_frame_equal(expected["selection"], result["selection"])
    assert len(result["selection"]) <= (5 if group_by == "customer_state" else 6)
    assert len(result["selection"]) < len(specs)


def test_monthly_grid_uses_the_full_cohort_range_even_for_unselected_endpoint_groups():
    from src.analysis.slot_comparisons import summarize_slot_group_months

    night = dated_cohort([("2018-01", "MG", "SP", 1, 0), ("2018-02", "SP", "RJ", 2, 1)],
                         COMPARISON_SLOTS[0], "n")
    afternoon = dated_cohort([("2018-02", "SP", "RJ", 2, 1), ("2018-03", "MG", "SP", 1, 0)],
                             COMPARISON_SLOTS[1], "a")
    result = summarize_slot_group_months(night, afternoon, group_by="state_pair")
    assert result["selection"]["group"].tolist() == ["SP → RJ"]
    assert result["monthly"]["purchase_month"].unique().tolist() == ["2018-01", "2018-02", "2018-03"]
    assert result["standardized"]["common_months"].tolist() == [1, 1]
    assert result["standardized"]["standardized_late_rate_pct"].tolist() == [50, 50]
    assert result["weights"]["weight"].tolist() == [1]


def test_monthly_comparison_handles_disjoint_months_empty_inputs_and_one_empty_slot():
    from src.analysis.slot_comparisons import summarize_slot_group_months

    night = dated_cohort([("2018-01", "SP", "RJ", 3, 1)], COMPARISON_SLOTS[0], "n")
    afternoon = dated_cohort([("2018-02", "SP", "RJ", 5, 4)], COMPARISON_SLOTS[1], "a")
    result = summarize_slot_group_months(night, afternoon, group_by="state_pair")
    table = result["standardized"]
    assert table["common_months"].eq(0).all()
    assert table["included_orders"].eq(0).all()
    assert table["excluded_orders"].tolist() == [3, 5]
    assert table["excluded_late_orders"].tolist() == [1, 4]
    assert table[["common_raw_late_rate_pct", "standardized_late_rate_pct"]].isna().all().all()
    assert result["weights"].empty
    empty_n = dated_cohort([], COMPARISON_SLOTS[0], "n")
    empty_a = dated_cohort([], COMPARISON_SLOTS[1], "a")
    empty = summarize_slot_group_months(empty_n, empty_a, group_by="state_pair")
    assert all(table.empty for table in empty.values())
    one_slot = summarize_slot_group_months(night, empty_a, group_by="state_pair")
    assert one_slot["standardized"]["raw_late_rate_pct"].iloc[0] == pytest.approx(100 / 3)
    assert pd.isna(one_slot["standardized"]["raw_late_rate_pct"].iloc[1])
    assert one_slot["standardized"]["standardized_late_rate_pct"].isna().all()


@pytest.mark.parametrize("defect", ["missing", "null", "string"])
def test_monthly_comparison_requires_valid_purchase_timestamps(defect):
    from src.analysis.slot_comparisons import summarize_slot_group_months

    night = dated_cohort([("2018-01", "SP", "RJ", 3, 1)], COMPARISON_SLOTS[0], "n")
    afternoon = dated_cohort([("2018-02", "SP", "RJ", 5, 4)], COMPARISON_SLOTS[1], "a")
    if defect == "missing":
        night = night.drop(columns="order_purchase_timestamp")
    elif defect == "null":
        night.loc[0, "order_purchase_timestamp"] = pd.NaT
    else:
        night["order_purchase_timestamp"] = "2018-01-01"
    with pytest.raises(ValueError, match="purchase timestamps"):
        summarize_slot_group_months(night, afternoon, group_by="state_pair")


def test_monthly_figures_show_zero_order_gaps_and_keep_full_rate_range():
    from src.analysis.slot_comparisons import summarize_slot_group_months
    from src.analysis.investigation_plots import plot_slot_group_months

    night = dated_cohort([("2018-01", "SP", "RJ", 1, 1), ("2018-03", "SP", "RJ", 3, 0)],
                         COMPARISON_SLOTS[0], "n")
    afternoon = dated_cohort([("2018-01", "SP", "RJ", 2, 0)], COMPARISON_SLOTS[1], "a")
    result = summarize_slot_group_months(night, afternoon, group_by="state_pair")
    table = result["monthly"]
    before = table.copy(deep=True)
    figure = plot_slot_group_months(table, group_by="state_pair", groups=["SP → RJ"])
    try:
        figure.canvas.draw()
        rates = figure.axes[0].lines[0].get_ydata()
        assert rates[0] == 100 and pd.isna(rates[1]) and rates[2] == 0
        assert figure.axes[0].get_ylim()[1] >= 100
        assert figure.axes[1].lines[0].get_ydata().tolist() == [1, 0, 3]
    finally:
        plt.close(figure)
    pd.testing.assert_frame_equal(table, before)
    with pytest.raises(ValueError, match="distinct groups present"):
        plot_slot_group_months(table, group_by="state_pair", groups=["missing"])
    bad = table.copy()
    bad.loc[0, "late_rate_pct"] = 20
    with pytest.raises(ValueError, match="matching order denominators"):
        plot_slot_group_months(bad, group_by="state_pair")
    empty = summarize_slot_group_months(night.iloc[:0], afternoon.iloc[:0], group_by="state_pair")
    empty_figure = plot_slot_group_months(empty["monthly"], group_by="state_pair")
    try:
        empty_figure.canvas.draw()
        assert "No selected groups" in empty_figure.axes[0].texts[0].get_text()
    finally:
        plt.close(empty_figure)


@pytest.mark.parametrize("slot", ["unknown", SLOT_LABELS[1], SLOT_LABELS[3], None, pd.NA])
def test_monthly_figures_reject_unexpected_slots_in_selected_groups(slot):
    from src.analysis.investigation_plots import plot_slot_group_months

    valid = pd.DataFrame({
        "group": ["SP"] * 2, "purchase_month": ["2018-01"] * 2,
        "approval_slot": list(COMPARISON_SLOTS), "orders": [10, 20],
        "late_orders": [1, 2], "late_rate_pct": [10., 10.],
    })
    extra = valid.iloc[[0]].copy()
    extra["approval_slot"] = slot
    table = pd.concat([valid, extra], ignore_index=True)
    before = table.copy(deep=True)
    with pytest.raises(ValueError, match="recognized overnight and afternoon approval slots"):
        plot_slot_group_months(table, group_by="customer_state")
    pd.testing.assert_frame_equal(table, before)


def test_monthly_figures_ignore_unselected_groups_with_other_slot_labels():
    from src.analysis.investigation_plots import plot_slot_group_months

    table = pd.DataFrame({
        "group": ["SP", "SP", "SP", "SP", "RJ"],
        "purchase_month": ["2018-01", "2018-01", "2018-02", "2018-02", "2018-01"],
        "approval_slot": [*COMPARISON_SLOTS, *COMPARISON_SLOTS, "unknown"],
        "orders": [10, 20, 0, 5, 999], "late_orders": [1, 2, 0, 0, 999],
        "late_rate_pct": [10., 10., float("nan"), 0., 100.],
    })
    before = table.copy(deep=True)
    figure = plot_slot_group_months(table, group_by="customer_state", groups=["SP"])
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 2
        assert pd.isna(figure.axes[0].lines[0].get_ydata()[1])
        assert figure.axes[1].lines[0].get_ydata().tolist() == [10, 0]
        assert figure.axes[1].lines[1].get_ydata().tolist() == [20, 5]
    finally:
        plt.close(figure)
    pd.testing.assert_frame_equal(table, before)
