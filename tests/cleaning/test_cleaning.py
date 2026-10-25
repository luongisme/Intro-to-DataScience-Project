"""Unit and notebook contract tests for the cleaning phase."""

import ast
import json

import pandas as pd
import pytest

from src.cleaning import (
    clean_physical_measurements,
    flag_timeline_issues,
    format_cleaning_preview,
    inspect_text_fields,
    normalize_city_whitespace,
)
from src.cleaning.validation import (
    assert_input_unchanged,
    capture_input_baseline,
    validate_cleaning_input,
)
from src.paths import PROJECT_ROOT


NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/02_data_cleaning.ipynb"


@pytest.fixture
def cleaning_full_frame(eda_full_frame):
    """Use canonical Olist-shaped IDs while retaining the EDA fixture geometry."""
    frame = eda_full_frame.copy(deep=True)
    for group, column in enumerate(
        ("order_id", "product_id", "seller_id", "customer_id"), start=1,
    ):
        mapping = {
            value: f"{group:02x}{position:030x}"
            for position, value in enumerate(frame[column].unique(), start=1)
        }
        frame[column] = frame[column].map(mapping).astype("string")
    return frame


def test_input_baseline_tracks_shape_schema_values_and_row_order(cleaning_full_frame):
    baseline = capture_input_baseline(cleaning_full_frame)
    assert_input_unchanged(cleaning_full_frame, baseline)

    changed = cleaning_full_frame.copy()
    changed.iloc[0, changed.columns.get_loc("price")] += 1
    with pytest.raises(AssertionError):
        assert_input_unchanged(changed, baseline)


def test_cleaning_input_accepts_consistent_full_merge(cleaning_full_frame):
    profiles = validate_cleaning_input(cleaning_full_frame)
    assert {"key_profile", "entity_consistency", "attribute_consistency", "conflicts"} <= set(profiles)
    assert profiles["conflicts"].empty


def test_cleaning_input_rejects_conflicting_entity_attributes(cleaning_full_frame):
    conflicted = cleaning_full_frame.copy()
    conflicted.loc[conflicted.index[2], "product_weight_g"] = 999.0
    with pytest.raises(ValueError, match="conflicting attributes"):
        validate_cleaning_input(conflicted)


def test_city_normalization_changes_only_whitespace(cleaning_full_frame):
    source = cleaning_full_frame.copy()
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


def test_physical_measurement_cleaning_nulls_nonpositive_values_and_flags_rows(cleaning_full_frame):
    source = cleaning_full_frame.copy()
    source.iloc[0, source.columns.get_loc("product_weight_g")] = 0.0
    source.iloc[1, source.columns.get_loc("product_width_cm")] = -2.0
    before = source.copy(deep=True)

    cleaned, summary = clean_physical_measurements(source)

    assert pd.isna(cleaned.iloc[0]["product_weight_g"])
    assert pd.isna(cleaned.iloc[1]["product_width_cm"])
    assert cleaned["has_invalid_physical_measurement"].tolist() == [True, True, False]
    assert summary["nonpositive_item_rows"].sum() == 2
    assert source.equals(before)


def test_timeline_check_flags_reversals_without_changing_timestamps(cleaning_full_frame):
    source = cleaning_full_frame.copy()
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


def test_cleaning_preview_formats_a_copy_without_changing_data(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
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


@pytest.mark.parametrize("column,value", [
    ("order_id", " padded_id "),
    ("product_id", "G" * 32),
    ("seller_id", "a" * 31),
    ("customer_id", "a" * 33),
    ("customer_zip_code_prefix", "12x"),
    ("seller_zip_code_prefix", "1234"),
    ("customer_zip_code_prefix", "123\uff145"),
    ("customer_state", "sP"),
    ("seller_state", "ZZ"),
    ("order_status", "unknown_status"),
    ("order_status", "Delivered"),
    ("order_status", " delivered "),
    ("customer_zip_code_prefix", ""),
])
def test_text_validation_rejects_bad_formats_and_domains_without_mutation(
    cleaning_full_frame, column, value,
):
    source = cleaning_full_frame.copy(deep=True)
    source.iloc[0, source.columns.get_loc(column)] = value
    before = source.copy(deep=True)

    profile = inspect_text_fields(source)
    row = profile["summary"].set_index("column").loc[column]
    assert row["invalid_item_rows"] == 1
    issue = profile["issues"].iloc[0]
    assert issue["column"] == column
    assert issue["row_position"] == 0
    assert issue["value_repr"] == repr(value)
    with pytest.raises(ValueError, match="invalid text values"):
        validate_cleaning_input(source)
    pd.testing.assert_frame_equal(source, before, check_exact=True)


def test_text_inspection_preserves_zip_zeros_and_reports_all_duplicate_index_issues(
    cleaning_full_frame,
):
    source = cleaning_full_frame.copy(deep=True)
    source.iloc[0:2, source.columns.get_loc("seller_state")] = "ZZ"
    profile = inspect_text_fields(source)
    issues = profile["issues"]

    assert issues["row_position"].tolist() == [0, 1]
    assert issues["index_label"].tolist() == [7, 7]
    assert issues["order_item_id"].tolist() == [1, 2]
    assert source.iloc[0]["customer_zip_code_prefix"] == "00123"
    summary = profile["summary"]
    assert summary["column"].nunique() == 9
    assert (summary["missing_item_rows"] + summary["valid_item_rows"]
            + summary["invalid_item_rows"]).eq(len(source)).all()


def test_text_inspection_accepts_raw_created_status_and_preserves_optional_nulls(
    cleaning_full_frame,
):
    source = cleaning_full_frame.copy(deep=True)
    source["order_status"] = pd.Series("created", index=source.index, dtype="string")
    for column in ("customer_zip_code_prefix", "seller_state"):
        source[column] = pd.Series(pd.NA, index=source.index, dtype="string")
    before = source.copy(deep=True)

    profiles = validate_cleaning_input(source)
    summary = profiles["text_formats"].set_index("column")
    assert summary["invalid_item_rows"].sum() == 0
    assert summary.loc["customer_zip_code_prefix", "missing_item_rows"] == 3
    assert summary.loc["seller_state", "missing_item_rows"] == 3
    assert summary.loc["order_status", "valid_item_rows"] == 3
    pd.testing.assert_frame_equal(source, before, check_exact=True)


@pytest.mark.parametrize("value", [123, True])
def test_text_inspection_does_not_coerce_nonstring_zip_values(cleaning_full_frame, value):
    source = cleaning_full_frame.copy(deep=True)
    source["customer_zip_code_prefix"] = source["customer_zip_code_prefix"].astype(object)
    source.iloc[0, source.columns.get_loc("customer_zip_code_prefix")] = value
    before = source.copy(deep=True)

    summary = inspect_text_fields(source)["summary"].set_index("column")
    assert summary.loc["customer_zip_code_prefix", "invalid_item_rows"] == 1
    pd.testing.assert_frame_equal(source, before, check_exact=True)


def test_text_inspection_empty_input_has_stable_output_columns(cleaning_full_frame):
    profile = inspect_text_fields(cleaning_full_frame.iloc[:0])
    assert len(profile["summary"]) == 9
    assert profile["summary"]["item_rows"].eq(0).all()
    assert profile["issues"].empty

