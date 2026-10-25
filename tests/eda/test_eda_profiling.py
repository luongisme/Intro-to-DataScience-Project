"""Hand-calculated contracts for structural/distribution/missingness profiles."""

import numpy as np
import pandas as pd
import pytest

from src.eda.profiling import (
    blank_string_mask, finite_numeric_values, get_column_roles, limit_frequencies,
    profile_columns, profile_dataset, profile_frequencies, profile_missing_patterns,
    profile_numeric, profile_paired_missingness, profile_temporal_frequencies,
    profile_timestamp_coverage, require_columns,
)
from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA


def test_column_roles_preserve_identifiers_and_source_ownership():
    table = get_column_roles().set_index("column")
    assert table.index.tolist() == FULL_MERGE_SCHEMA.columns
    assert table.loc["order_item_id", "role"] == "identifier"
    assert table.loc["customer_zip_code_prefix", "role"] == "zip_code"
    assert table.loc["product_weight_g", "role"] == "measurement"
    assert table.loc["order_approved_at", "role"] == "timestamp"
    assert table.loc["price", "source_key"] == ("order_id", "order_item_id")
    assert table.loc["product_weight_g", "source_key"] == ("product_id",)
    table.loc["order_id", "role"] = "changed"
    assert get_column_roles().set_index("column").loc["order_id", "role"] == "identifier"


@pytest.mark.parametrize("series, expected", [
    (pd.Series([None, "", " \t", "NA", "0"], dtype="string"), [False, True, True, False, False]),
    (pd.Series(pd.Categorical([None, "", "x"])), [False, True, False]),
    (pd.Series([None, np.nan, pd.NA, " ", 42, b""], dtype=object), [False, False, False, True, False, False]),
    (pd.Series([1, pd.NA, 3], dtype="Int64", index=[1, 1, 0]), [False, False, False]),
    (pd.Series([], dtype="string"), []),
])
def test_blank_strings_are_not_nulls_or_numeric_values(series, expected):
    before = series.copy(deep=True)
    mask = blank_string_mask(series)
    assert mask.tolist() == expected
    assert mask.dtype == bool
    pd.testing.assert_index_equal(mask.index, series.index)
    pd.testing.assert_series_equal(series, before)


@pytest.mark.parametrize("columns,error", [("price", TypeError), (["missing"], KeyError), (["price", "price"], ValueError)])
def test_column_selection_errors_are_explicit(eda_full_frame, columns, error):
    with pytest.raises(error):
        require_columns(eda_full_frame, columns)


def test_duplicate_column_labels_are_rejected():
    with pytest.raises(ValueError, match="unique column"):
        profile_columns(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(TypeError):
        require_columns(None)


def test_dataset_overview_distinguishes_duplicate_participants_from_copies(eda_full_frame):
    table = profile_dataset(pd.concat([eda_full_frame, eda_full_frame.iloc[[0]]])).set_index("metric")
    assert table.loc["Rows", "value"] == 4
    assert table.loc["Exact duplicate participants", "value"] == 2
    assert table.loc["Exact duplicate excess copies", "value"] == 1
    assert table.loc["Observed orders", "value"] == 2
    assert table.loc["Observed products", "value"] == 2


def test_completeness_uses_all_rows_and_preserves_literal_na():
    frame = pd.DataFrame({"custom": [" ", "", "NA", None], "price": [1., np.nan, 3., 4.]})
    table = profile_columns(frame).set_index("column")
    assert table.loc["custom", "missing"] == 1
    assert table.loc["custom", "blank"] == 2
    assert table.loc["custom", "missing_pct"] == 25
    assert table.loc["custom", "blank_pct"] == 50
    assert table.loc["custom", "distinct_non_null"] == 3
    assert table.loc["custom", "role"] == "unregistered"


def test_empty_completeness_percentages_are_undefined(eda_full_frame):
    table = profile_columns(eda_full_frame.iloc[:0])
    assert len(table) == 25
    assert table["rows"].eq(0).all()
    assert table["missing_pct"].isna().all() and table["blank_pct"].isna().all()


def test_numeric_statistics_use_finite_values_and_explicit_denominators():
    series = pd.Series([-2., 0., 1., 2., 100., np.inf, -np.inf, np.nan], name="price", index=[1, 1, 2, 2, 3, 3, 4, 4])
    row = profile_numeric(series.to_frame(), ["price"]).iloc[0]
    assert row["rows"] == 8 and row["missing"] == 1 and row["missing_pct"] == 12.5
    assert row["finite"] == 5 and row["non_finite_non_null"] == 2
    assert row["zero"] == row["negative"] == 1
    assert row["mean"] == pytest.approx(20.2)
    assert row["std"] == pytest.approx(np.sqrt(1992.2))
    assert row["p25"] == 0 and row["median"] == 1 and row["p75"] == 2
    assert row["iqr_lower"] == -3 and row["iqr_upper"] == 5
    assert row["below_iqr"] == 0 and row["above_iqr"] == 1 and row["outside_iqr_pct"] == 20
    values = finite_numeric_values(series)
    assert values.tolist() == [-2., 0., 1., 2., 100.]
    pd.testing.assert_index_equal(values.index, series.index[:5])


@pytest.mark.parametrize("values", [[], [pd.NA, pd.NA]])
def test_numeric_empty_and_missing_inputs_do_not_create_zero_statistics(values):
    frame = pd.DataFrame({"price": pd.Series(values, dtype="Float64")})
    row = profile_numeric(frame, ["price"]).iloc[0]
    assert row["finite"] == 0 and row["above_iqr"] == 0
    assert pd.isna(row["mean"]) and pd.isna(row["median"]) and pd.isna(row["outside_iqr_pct"])


def test_numeric_singleton_and_constant_samples():
    singleton = profile_numeric(pd.DataFrame({"price": [7.]}), ["price"]).iloc[0]
    constant = profile_numeric(pd.DataFrame({"price": [2., 2., 2.]}), ["price"]).iloc[0]
    assert singleton["median"] == 7 and pd.isna(singleton["std"])
    assert constant["iqr"] == 0 and constant["outside_iqr_pct"] == 0


@pytest.mark.parametrize("series,error", [
    (pd.Series([1, 2], name="order_item_id"), ValueError),
    (pd.Series([True, False]), TypeError), (pd.Series([1 + 2j]), TypeError),
    (pd.Series(["1", "2"]), TypeError),
])
def test_numeric_profiles_reject_nonmeasurement_inputs(series, error):
    with pytest.raises(error):
        finite_numeric_values(series)


def test_frequencies_preserve_values_and_exclude_unused_category_levels():
    series = pd.Series(pd.Categorical(["A", "A", "B", None], categories=["A", "B", "unused"]))
    table = profile_frequencies(series)
    assert table["count"].tolist() == [2, 1, 1]
    assert table["pct_all"].tolist() == [50, 25, 25]
    assert table["total_rows"].eq(4).all()
    assert "unused" not in table["value"].dropna().tolist()


def test_top_values_preserve_other_blank_missing_and_literal_special_names():
    full = profile_frequencies(pd.Series(["A", "A", "B", "C", "", " ", "Missing", "Other", pd.NA], dtype="string"))
    before = full.copy(deep=True)
    shown = limit_frequencies(full, top_n=1)
    assert shown.loc[shown["kind"].eq("value"), "value"].tolist() == ["A"]
    assert shown.set_index("kind")["count"].to_dict() == {"value": 2, "other": 4, "blank": 2, "missing": 1}
    assert shown["count"].sum() == 9 and shown["pct_all"].sum() == pytest.approx(100)
    assert set(full.loc[full["kind"].eq("value"), "value"]) >= {"Missing", "Other"}
    pd.testing.assert_frame_equal(full, before)


@pytest.mark.parametrize("values", [[], [pd.NA, pd.NA]])
def test_frequency_tables_account_for_empty_and_all_missing_series(values):
    table = limit_frequencies(profile_frequencies(pd.Series(values, dtype="string")), 2)
    assert table["count"].sum() == len(values)
    assert table["kind"].tolist() == ["missing"]
    assert pd.isna(table["pct_all"].iloc[0]) if not values else table["pct_all"].iloc[0] == 100


@pytest.mark.parametrize("top_n", [0, -1, True, 1.5])
def test_frequency_display_rejects_invalid_limits(top_n):
    with pytest.raises(ValueError):
        limit_frequencies(profile_frequencies(pd.Series(["A"])), top_n)


def test_frequency_display_rejects_inconsistent_or_already_collapsed_tables():
    table = profile_frequencies(pd.Series(["A", "B"]))
    with pytest.raises(ValueError):
        limit_frequencies(table.assign(count=[1, 0, 0]))
    with pytest.raises(ValueError):
        limit_frequencies(table.assign(count=[0.5, 1.5, 0.]))
    with pytest.raises(ValueError):
        limit_frequencies(limit_frequencies(table, 1))


def test_timestamp_coverage_retains_clock_precision():
    values = pd.to_datetime(["2017-01-02 01:02:03.123456789", "2017-01-02 01:02:03.123456789", "2017-02-05 00:00:00", None], format="mixed")
    row = profile_timestamp_coverage(pd.DataFrame({"time": values}), ["time"]).iloc[0]
    assert row["rows"] == 4 and row["non_null"] == 3 and row["missing_pct"] == 25
    assert row["non_midnight"] == 2 and row["subsecond"] == 2 and row["distinct_non_null"] == 2
    assert row["min"] == values[0] and row["max"] == values[2]


@pytest.mark.parametrize("frequency,expected", [
    ("month", ["2017-01", "2017-02"]), ("date", ["2017-01-02", "2017-02-05"]),
    ("weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    ("hour", list(range(24))),
])
def test_temporal_buckets_use_calendar_order_and_all_rows(frequency, expected):
    series = pd.Series(pd.to_datetime(["2017-01-02 01:00", "2017-01-02 01:00", "2017-02-05 23:00", None]))
    table = profile_temporal_frequencies(series, frequency)
    assert table.loc[table["kind"].eq("value"), "value"].tolist() == expected
    assert table["count"].sum() == 4 and table.iloc[-1]["kind"] == "missing"
    assert table.iloc[-1]["count"] == 1 and table.iloc[-1]["pct_all"] == 25
    if frequency == "hour":
        assert table.loc[table["value"].eq(1).fillna(False), "count"].item() == 2


def test_temporal_profiles_preserve_explicit_timezone_clock():
    series = pd.Series(pd.to_datetime(["2017-01-01 20:00"], utc=True)).dt.tz_convert("Asia/Saigon")
    before = series.copy(deep=True)
    assert profile_temporal_frequencies(series, "date").iloc[0]["value"] == "2017-01-02"
    pd.testing.assert_series_equal(series, before)


@pytest.mark.parametrize("frequency", ["month", "date", "weekday", "hour"])
@pytest.mark.parametrize("values", [[], [pd.NaT]])
def test_temporal_empty_or_missing_buckets_keep_totals(frequency, values):
    table = profile_temporal_frequencies(pd.Series(values, dtype="datetime64[us]"), frequency)
    assert table["count"].sum() == len(values)


def test_temporal_profiles_reject_parser_work_and_business_slots():
    with pytest.raises(TypeError):
        profile_temporal_frequencies(pd.Series(["2017-01-01"]))
    with pytest.raises(TypeError):
        profile_timestamp_coverage(pd.DataFrame({"time": ["2017-01-01"]}), ["time"])
    with pytest.raises(ValueError):
        profile_temporal_frequencies(pd.Series([], dtype="datetime64[us]"), "slot")


def test_missing_patterns_count_complete_rows_and_exact_combinations():
    frame = pd.DataFrame({"a": [None, None, 1, 1], "b": [None, 2, None, 2], "c": [5] * 4}, index=[7, 7, 2, 0])
    result = profile_missing_patterns(frame)
    assert result["row_distribution"]["rows"].tolist() == [1, 2, 1, 0]
    assert result["row_distribution"]["pct_all"].tolist() == [25, 50, 25, 0]
    assert dict(zip(result["patterns"]["missing_columns"], result["patterns"]["rows"])) == {(): 1, ("a",): 1, ("b",): 1, ("a", "b"): 1}
    assert len(profile_missing_patterns(frame, ["a"])["patterns"]) == 2
    assert profile_missing_patterns(pd.DataFrame({"text": ["", " ", None, "x"]}))["row_distribution"]["rows"].tolist() == [3, 1]


def test_missing_pairs_use_asymmetric_conditioning_denominators():
    result = profile_paired_missingness(pd.DataFrame({"a": [None, None, None, 1], "b": [None, 2, 2, 2], "c": [5] * 4}))
    pair = result["pairs"].set_index(["left", "right"]).loc[("a", "b")]
    assert [pair[name] for name in ["both_missing", "left_only_missing", "right_only_missing", "both_present"]] == [1, 2, 0, 1]
    assert result["conditional_pct"].loc["a", "b"] == pytest.approx(100 / 3)
    assert result["conditional_pct"].loc["b", "a"] == 100
    assert result["conditional_pct"].loc["a", "c"] == 0
    assert result["conditional_pct"].loc["c"].isna().all()


@pytest.mark.parametrize("mode", ["empty", "all_missing", "single_column"])
def test_missingness_edge_selections(mode):
    if mode == "empty":
        frame = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=float)})
        assert profile_missing_patterns(frame)["patterns"].empty
        assert profile_missing_patterns(frame)["row_distribution"]["pct_all"].isna().all()
        assert profile_paired_missingness(frame)["conditional_pct"].isna().all().all()
    elif mode == "all_missing":
        frame = pd.DataFrame({"a": [pd.NA, pd.NA], "b": [pd.NaT, pd.NaT]})
        assert profile_missing_patterns(frame)["patterns"].iloc[0]["missing_columns"] == ("a", "b")
        assert profile_paired_missingness(frame)["conditional_pct"].eq(100).all().all()
    else:
        result = profile_paired_missingness(pd.DataFrame({"a": [None, 1]}))
        assert result["pairs"].empty and result["conditional_pct"].shape == (1, 1)


@pytest.mark.parametrize("function", [profile_missing_patterns, profile_paired_missingness])
def test_missingness_rejects_empty_or_duplicate_selections(function):
    frame = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        function(frame, [])
    with pytest.raises(ValueError):
        function(frame, ["a", "a"])


@pytest.mark.parametrize("operation", [
    lambda df: profile_dataset(df), lambda df: profile_columns(df),
    lambda df: profile_numeric(df, ["price", "freight_value"]),
    lambda df: profile_frequencies(df["customer_city"]),
    lambda df: profile_missing_patterns(df), lambda df: profile_paired_missingness(df),
    lambda df: profile_timestamp_coverage(df, ["order_approved_at"]),
    lambda df: profile_temporal_frequencies(df["order_approved_at"]),
])
def test_profiling_never_mutates_the_input(eda_full_frame, operation):
    before = eda_full_frame.copy(deep=True)
    operation(eda_full_frame)
    pd.testing.assert_frame_equal(eda_full_frame, before)
