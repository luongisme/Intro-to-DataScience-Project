"""Unit and notebook contract tests for the cleaning phase."""

import ast
import json

import pandas as pd
import pytest

from src.cleaning import (
    clean_physical_measurements,
    flag_timeline_issues,
    format_cleaning_preview,
    normalize_city_whitespace,
)
from src.cleaning.validation import assert_input_unchanged, capture_input_baseline, validate_cleaning_input
from src.paths import PROJECT_ROOT


NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/02_data_cleaning.ipynb"


def test_input_baseline_tracks_shape_schema_values_and_row_order(eda_full_frame):
    baseline = capture_input_baseline(eda_full_frame)
    assert_input_unchanged(eda_full_frame, baseline)

    changed = eda_full_frame.copy()
    changed.iloc[0, changed.columns.get_loc("price")] += 1
    with pytest.raises(AssertionError):
        assert_input_unchanged(changed, baseline)


def test_cleaning_input_accepts_consistent_full_merge(eda_full_frame):
    profiles = validate_cleaning_input(eda_full_frame)
    assert {"key_profile", "entity_consistency", "attribute_consistency", "conflicts"} <= set(profiles)
    assert profiles["conflicts"].empty


def test_cleaning_input_rejects_conflicting_entity_attributes(eda_full_frame):
    conflicted = eda_full_frame.copy()
    conflicted.loc[conflicted.index[2], "product_weight_g"] = 999.0
    with pytest.raises(ValueError, match="conflicting attributes"):
        validate_cleaning_input(conflicted)


def test_city_normalization_changes_only_whitespace(eda_full_frame):
    source = eda_full_frame.copy()
    source.iloc[0, source.columns.get_loc("customer_city")] = "  sao   paulo "
    before = source.copy(deep=True)

    cleaned, summary = normalize_city_whitespace(source)

    assert cleaned.index.equals(source.index)
    assert cleaned.iloc[0]["customer_city"] == "sao paulo"
    assert cleaned.iloc[0]["customer_state"] == source.iloc[0]["customer_state"]
    assert source.equals(before)
    row = summary.set_index("column").loc["customer_city"]
    assert row["changed_item_rows"] == 1
    assert row["changed_entities"] == 1


def test_physical_measurement_cleaning_nulls_nonpositive_values_and_flags_rows(eda_full_frame):
    source = eda_full_frame.copy()
    source.iloc[0, source.columns.get_loc("product_weight_g")] = 0.0
    source.iloc[1, source.columns.get_loc("product_width_cm")] = -2.0
    before = source.copy(deep=True)

    cleaned, summary = clean_physical_measurements(source)

    assert pd.isna(cleaned.iloc[0]["product_weight_g"])
    assert pd.isna(cleaned.iloc[1]["product_width_cm"])
    assert cleaned["has_invalid_physical_measurement"].tolist() == [True, True, False]
    assert summary["nonpositive_item_rows"].sum() == 2
    assert source.equals(before)


def test_timeline_check_flags_reversals_without_changing_timestamps(eda_full_frame):
    source = eda_full_frame.copy()
    purchase = source.columns.get_loc("order_purchase_timestamp")
    approved = source.columns.get_loc("order_approved_at")
    source.iloc[0:2, purchase] = pd.Timestamp("2017-01-01 12:00")
    source.iloc[0:2, approved] = pd.Timestamp("2017-01-01 11:00")
    before = source.copy(deep=True)

    cleaned, summary, core_summary = flag_timeline_issues(source)

    assert cleaned["has_invalid_timeline"].tolist() == [True, True, False]
    check = summary.set_index("check").loc["approved_before_purchase"]
    assert check["affected_orders"] == 1
    assert core_summary.loc[0, "orders"] == 2
    assert source.equals(before)


def test_cleaning_preview_formats_a_copy_without_changing_data(eda_full_frame):
    source = eda_full_frame.copy(deep=True)
    before = source.copy(deep=True)

    preview = format_cleaning_preview(source, rows=2)

    assert preview.shape == (2, len(source.columns))
    assert preview.columns[0] == "Order ID"
    assert preview.loc[0, "Order Status"] == "Delivered"
    assert preview.loc[0, "Item Price (BRL)"] == "R$ 10.00"
    assert preview.loc[0, "Customer ZIP Prefix"] == "00123"
    assert all(pd.api.types.is_string_dtype(dtype) for dtype in preview.dtypes)
    assert source.equals(before)


def test_cleaning_notebook_code_parses_and_has_no_saved_cell_outputs():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    for cell in code_cells:
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        ast.parse(source)
        assert cell.get("execution_count") is None
        assert not cell.get("outputs")



