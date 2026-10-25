"""Entity, condition and domain checks with independent small examples."""

import numpy as np
import pandas as pd
import pytest

from src.eda.checks import (
    build_entity_views, profile_entity_consistency, profile_keys,
    profile_order_composition, profile_text_formats, profile_timestamp_ordering,
    summarize_condition_by_group,
)


def test_keys_distinguish_item_uniqueness_from_expected_entity_repetitions(eda_full_frame):
    table = profile_keys(eda_full_frame).set_index("key")
    assert table.loc["order_items", "distinct_valid_keys"] == 3
    assert table.loc["order_items", "rows_in_repeated_keys"] == 0
    for entity in ["orders", "customers", "products", "sellers"]:
        assert table.loc[entity, "distinct_valid_keys"] == 2
        assert table.loc[entity, "repeated_key_groups"] == 1
        assert table.loc[entity, "rows_in_repeated_keys"] == 2
        assert table.loc[entity, "excess_key_rows"] == 1


def test_composite_key_invalid_counts_use_union_not_sum():
    frame = pd.DataFrame({"a": [None, "", "O", "O", None], "b": [" ", "i", "1", "1", pd.NA]}, index=[1, 1, 0, 5, 5])
    row = profile_keys(frame, {"custom": ["a", "b"]}).iloc[0]
    assert row["missing_rows"] == 2 and row["blank_rows"] == 2 and row["invalid_rows"] == 3
    assert row["valid_rows"] == 2 and row["distinct_valid_keys"] == 1
    assert row["repeated_key_groups"] == row["excess_key_rows"] == 1


@pytest.mark.parametrize("keys,error", [({}, ValueError), ({"empty": []}, ValueError), ({"": ["order_id"]}, ValueError), ({"x": "order_id"}, TypeError)])
def test_invalid_key_declarations_are_rejected(eda_full_frame, keys, error):
    with pytest.raises(error):
        profile_keys(eda_full_frame, keys)


def test_entity_views_preserve_source_grain_dtypes_and_independence(eda_full_frame):
    views = build_entity_views(eda_full_frame)
    assert {name: len(view) for name, view in views.items()} == {"order_items": 3, "orders": 2, "customers": 2, "products": 2, "sellers": 2}
    assert profile_entity_consistency(eda_full_frame)["summary"]["conflicted_entities"].eq(0).all()
    for view in views.values():
        for column in view:
            assert view[column].dtype == eda_full_frame[column].dtype
    before = eda_full_frame.copy(deep=True)
    views["products"].iloc[0, views["products"].columns.get_loc("product_category_name")] = "changed"
    pd.testing.assert_frame_equal(eda_full_frame, before)


@pytest.mark.parametrize("entity,column,position,value", [
    ("orders", "order_status", 1, "shipped"),
    ("customers", "customer_city", 1, "changed"),
    ("products", "product_weight_g", 2, 999.),
    ("sellers", "seller_state", 2, "XX"),
    ("products", "product_category_name", 2, pd.NA),
])
def test_conflicting_entity_values_are_reported_before_view_selection(eda_full_frame, entity, column, position, value):
    eda_full_frame.iloc[position, eda_full_frame.columns.get_loc(column)] = value
    result = profile_entity_consistency(eda_full_frame)
    summary = result["summary"].set_index("entity").loc[entity]
    assert summary["valid_entities"] == 2 and summary["conflicted_entities"] == 1
    assert summary["conflict_pct"] == 50 and summary["item_rows_in_conflicted_entities"] == 2
    assert result["conflicts"]["attribute"].tolist() == [column]
    with pytest.raises(ValueError, match="Cannot build reliable"):
        build_entity_views(eda_full_frame)


def test_all_missing_entity_attribute_is_incomplete_but_consistent(eda_full_frame):
    eda_full_frame.loc[eda_full_frame["product_id"].eq("P1"), "product_category_name"] = pd.NA
    assert profile_entity_consistency(eda_full_frame)["conflicts"].empty
    assert build_entity_views(eda_full_frame)["products"]["product_category_name"].isna().sum() == 1


@pytest.mark.parametrize("invalid", [pd.NA, " "])
def test_invalid_entity_keys_are_reported_and_block_views(eda_full_frame, invalid):
    eda_full_frame.iloc[0, eda_full_frame.columns.get_loc("seller_id")] = invalid
    summary = profile_entity_consistency(eda_full_frame)["summary"].set_index("entity")
    assert summary.loc["sellers", "invalid_key_rows"] == 1
    with pytest.raises(ValueError):
        build_entity_views(eda_full_frame)


def test_duplicate_item_keys_block_entity_views(eda_full_frame):
    with pytest.raises(ValueError):
        build_entity_views(pd.concat([eda_full_frame, eda_full_frame.iloc[[0]]]))


def test_empty_entity_profiles_have_undefined_denominators(eda_full_frame):
    empty = eda_full_frame.iloc[:0]
    assert profile_keys(empty)["valid_rows"].eq(0).all()
    assert profile_entity_consistency(empty)["summary"]["conflict_pct"].isna().all()
    assert all(view.empty for view in build_entity_views(empty).values())


def test_group_conditions_keep_unknowns_and_missing_categorical_keys():
    frame = pd.DataFrame({"group": pd.Categorical(["A", "A", None, "B", "B"], categories=["A", "B", "unused"]), "subgroup": ["x", None, "x", "y", "y"]}, index=[2, 2, 1, 1, 0])
    condition = pd.Series([True, pd.NA, False, pd.NA, pd.NA], dtype="boolean", index=frame.index)
    result = summarize_condition_by_group(frame, condition, ["group"])
    a, b = result.loc[result["group"].eq("A")].iloc[0], result.loc[result["group"].eq("B")].iloc[0]
    assert [a[name] for name in ["rows", "evaluated_rows", "unknown_rows", "affected_rows"]] == [2, 1, 1, 1]
    assert a["affected_pct_evaluated"] == 100 and a["affected_pct_all"] == 50
    assert b["unknown_rows"] == 2 and pd.isna(b["affected_pct_evaluated"]) and b["affected_pct_all"] == 0
    assert len(result) == 3 and result["rows"].sum() == 5
    assert result.loc[result["group"].isna(), "rows"].item() == 1
    multi = summarize_condition_by_group(frame, condition, ["group", "subgroup"])
    assert multi["rows"].sum() == 5 and multi["affected_rows"].sum() == 1
    assert summarize_condition_by_group(frame.iloc[:0], condition.iloc[:0], ["group"]).empty


def test_group_conditions_reject_misalignment_coercion_and_output_collisions():
    frame = pd.DataFrame({"group": ["A", "B"]}, index=[7, 8])
    with pytest.raises(ValueError, match="same index"):
        summarize_condition_by_group(frame, pd.Series([True, False]), ["group"])
    with pytest.raises(TypeError):
        summarize_condition_by_group(frame, pd.Series([1, 0], index=frame.index), ["group"])
    with pytest.raises(ValueError):
        summarize_condition_by_group(frame, pd.Series([True, False], index=frame.index), [])
    with pytest.raises(ValueError):
        summarize_condition_by_group(pd.DataFrame({"rows": [1]}), pd.Series([True]), ["rows"])


def test_text_checks_preserve_values_and_bound_positional_examples():
    frame = pd.DataFrame({"state": ["sp", " SP ", "SP", "", None, 5, "a  b", "bad\n", "x" + chr(0xfffd)]}, index=[1] * 9)
    before = frame.copy(deep=True)
    result = profile_text_formats(frame, ["state"], patterns={"state": "[A-Z]{2}"}, sample_size=2)
    row = result["summary"].iloc[0]
    assert row["string_rows"] == 7 and row["missing"] == row["non_string"] == row["blank"] == 1
    assert row["edge_whitespace"] == 2 and row["repeated_whitespace"] == 1
    assert row["control_character"] == row["replacement_character"] == 1
    assert row["pattern_evaluated"] == 7 and row["pattern_mismatch"] == 6
    assert row["pattern_mismatch_pct"] == pytest.approx(600 / 7)
    assert set(result["variants"]["raw_repr"]) == {repr("sp"), repr(" SP "), repr("SP")}
    assert result["examples"].groupby("check").size().le(2).all()
    for example in result["examples"].to_dict(orient="records"):
        assert example["raw_repr"] == repr(frame["state"].iloc[example["row_position"]])
    pd.testing.assert_frame_equal(frame, before)


def test_text_regex_does_not_cast_zip_numbers_or_verify_geography():
    frame = pd.DataFrame({"zip": ["00123", "1234", 123, None], "state": ["XX", "SP", None, "RJ"]})
    result = profile_text_formats(frame, ["zip", "state"], patterns={"zip": "[0-9]{5}", "state": "[A-Z]{2}"})["summary"].set_index("column")
    assert result.loc["zip", "pattern_evaluated"] == 2 and result.loc["zip", "pattern_mismatch"] == 1
    assert result.loc["zip", "non_string"] == 1 and frame["zip"].iloc[0] == "00123"
    assert result.loc["state", "pattern_mismatch"] == 0  # Shape alone does not validate XX.


@pytest.mark.parametrize("options,error", [
    ({"patterns": {"state": "["}}, ValueError), ({"patterns": {"other": ".*"}}, ValueError),
    ({"patterns": {"state": 1}}, TypeError), ({"sample_size": True}, ValueError), ({"sample_size": -1}, ValueError),
])
def test_text_check_configuration_errors_are_explicit(options, error):
    with pytest.raises(error):
        profile_text_formats(pd.DataFrame({"state": ["SP"]}), ["state"], **options)


def test_empty_text_profiles_and_disabled_examples():
    frame = pd.DataFrame({"text": pd.Series([], dtype="string")})
    result = profile_text_formats(frame, ["text"])
    assert result["summary"].iloc[0]["rows"] == 0
    assert result["examples"].empty and result["variants"].empty
    assert profile_text_formats(pd.DataFrame({"text": [" "]}), ["text"], sample_size=0)["examples"].empty


@pytest.fixture
def timeline_orders():
    return pd.DataFrame({
        "order_id": ["O1", "O2", "O3"], "order_status": ["delivered", "delivered", "processing"],
        "order_purchase_timestamp": pd.to_datetime(["2017-01-01 10:00", "2017-01-02 10:00", None]),
        "order_approved_at": pd.to_datetime(["2017-01-01 11:00", None, "2017-01-03 10:00"]),
        "order_delivered_carrier_date": pd.to_datetime(["2017-01-01 10:30", "2017-01-03 10:00", None]),
        "order_delivered_customer_date": pd.to_datetime(["2017-01-01 10:45", "2017-01-02 15:00", None]),
        "order_estimated_delivery_date": pd.to_datetime(["2017-01-01", "2017-01-01", None]),
    }, index=[4, 4, 0])


def test_timestamp_ordering_preserves_unknowns_and_calendar_date_precision(timeline_orders):
    result = profile_timestamp_ordering(timeline_orders, sample_size=1)
    table = result["summary"].set_index("check")
    assert table.loc["carrier_before_approval", "evaluated_orders"] == 1
    assert table.loc["carrier_before_approval", "unassessable_orders"] == 2
    assert table.loc["carrier_before_approval", "flagged_orders"] == 1
    assert table.loc["customer_before_carrier", "flagged_pct_evaluated"] == 50
    assert table.loc["customer_before_approval", "flagged_orders"] == 1
    assert table.loc["estimated_before_purchase", "flagged_orders"] == 1
    assert not result["conditions"]["estimated_before_purchase"].iloc[0]
    assert pd.isna(result["conditions"]["carrier_before_approval"].iloc[1])
    assert result["conditions"].dtypes.astype(str).eq("boolean").all()
    assert result["examples"].groupby("check").size().le(1).all()


def test_timestamp_ordering_empty_and_invalid_inputs(timeline_orders):
    assert profile_timestamp_ordering(timeline_orders.iloc[:0])["summary"]["flagged_pct_evaluated"].isna().all()
    with pytest.raises(ValueError):
        profile_timestamp_ordering(pd.concat([timeline_orders, timeline_orders.iloc[[0]]]))
    with pytest.raises(TypeError):
        profile_timestamp_ordering(timeline_orders.assign(order_approved_at="not parsed"))


def test_order_composition_has_explicit_order_and_seller_group_denominators(eda_full_frame):
    result = profile_order_composition(eda_full_frame)
    table = result["summary"].set_index("check")
    for check in ["Multiple items", "Multiple products", "Multiple sellers", "Multiple non-null deadlines", "Both equal and different state pairs"]:
        assert table.loc[check, "affected"] == 1 and table.loc[check, "total"] == 2
    assert table.loc["Multiple deadlines within seller", "affected"] == 0
    assert table.loc["Multiple deadlines within seller", "total"] == 3
    assert table.loc["Incomplete item numbering", "affected"] == 0
    assert result["orders"]["item_rows"].sum() == result["seller_deadlines"]["item_rows"].sum() == 3


@pytest.mark.parametrize("value", [0., -1., 1.5, np.inf])
def test_order_composition_flags_invalid_item_numbers(eda_full_frame, value):
    eda_full_frame["order_item_id"] = eda_full_frame["order_item_id"].astype(float)
    eda_full_frame.iloc[1, eda_full_frame.columns.get_loc("order_item_id")] = value
    result = profile_order_composition(eda_full_frame)
    assert result["orders"]["invalid_item_id_rows"].sum() == 1
    assert result["summary"].set_index("check").loc["Incomplete item numbering", "affected"] == 1


def test_item_numbering_gap_is_not_inferred_from_row_sort_order(eda_full_frame):
    eda_full_frame.iloc[1, eda_full_frame.columns.get_loc("order_item_id")] = 3
    result = profile_order_composition(eda_full_frame.iloc[::-1])
    assert result["summary"].set_index("check").loc["Incomplete item numbering", "affected"] == 1


def test_composition_preserves_missing_deadlines_and_uncomparable_states(eda_full_frame):
    eda_full_frame.iloc[1, eda_full_frame.columns.get_loc("shipping_limit_date")] = pd.NaT
    eda_full_frame.loc[eda_full_frame["customer_id"].eq("C1"), "customer_state"] = pd.NA
    result = profile_order_composition(eda_full_frame)
    assert result["orders"]["missing_deadline_items"].sum() == 1
    assert result["orders"]["items_with_uncomparable_states"].sum() == 2
    assert result["seller_deadlines"]["missing_deadline_items"].sum() == 1


def test_composition_empty_or_incompatible_input(eda_full_frame):
    result = profile_order_composition(eda_full_frame.iloc[:0])
    assert result["orders"].empty and result["seller_deadlines"].empty
    assert result["summary"]["affected_pct"].isna().all()
    with pytest.raises(TypeError):
        profile_order_composition(eda_full_frame.assign(shipping_limit_date="not parsed"))


@pytest.mark.parametrize("operation", [
    lambda df: profile_keys(df), lambda df: profile_entity_consistency(df),
    lambda df: build_entity_views(df), lambda df: profile_timestamp_ordering(build_entity_views(df)["orders"]),
    lambda df: profile_order_composition(df), lambda df: profile_text_formats(df, ["seller_city"]),
    lambda df: summarize_condition_by_group(df, df["price"].gt(10), ["order_status"]),
])
def test_domain_checks_never_mutate_the_input(eda_full_frame, operation):
    before = eda_full_frame.copy(deep=True)
    operation(eda_full_frame)
    pd.testing.assert_frame_equal(eda_full_frame, before)
