"""Unit and notebook contract tests for the cleaning phase."""

import ast
import json
import io
from contextlib import redirect_stdout
from dataclasses import replace
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.cleaning import (
    CLEANED_DATASET_NAME,
    CLEANED_ORDER_ITEMS_SCHEMA,
    assert_cleaned_roundtrip,
    clean_order_items,
    save_cleaned_data,
    validate_cleaning_output,
    clean_physical_measurements,
    flag_timeline_issues,
    flag_deadline_issues,
    format_cleaning_preview,
    filter_eligible_orders,
    inspect_text_fields,
    normalize_city_whitespace,
)
from src.cleaning.validation import (
    assert_input_unchanged,
    capture_input_baseline,
    validate_cleaning_input,
)
from src.paths import PROJECT_ROOT
from src.data_loader import load_data
from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA


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


def test_deadline_review_flags_only_affected_items_and_preserves_source(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    position = source.columns.get_loc("shipping_limit_date")
    source.iloc[0, position] = pd.Timestamp("2020-01-01")
    source.iloc[2, position] = pd.Timestamp("2017-01-31")
    source["has_invalid_physical_measurement"] = [False, True, False]
    source["has_invalid_timeline"] = [False, False, True]
    before = source.copy(deep=True)

    cleaned, summary, records = flag_deadline_issues(source)

    assert cleaned["needs_deadline_review"].tolist() == [True, False, True]
    assert cleaned["needs_deadline_review"].dtype == bool
    pd.testing.assert_frame_equal(cleaned[source.columns], before, check_exact=True)
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    assert records["order_item_id"].tolist() == [1, 1]
    assert records["deadline_before_purchase"].tolist() == [False, True]
    assert records["deadline_after_purchase_year"].tolist() == [True, False]
    table = summary.set_index("check")
    assert table["affected_item_rows"].tolist() == [1, 1]
    assert table["affected_orders"].tolist() == [1, 1]
    assert table["affected_order_seller_groups"].tolist() == [1, 1]
    preview = format_cleaning_preview(cleaned)
    assert preview["Shipping Deadline Needs Review"].tolist() == ["Yes", "No", "Yes"]


def test_deadline_review_keeps_missing_comparisons_unknown(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    source.iloc[0, source.columns.get_loc("shipping_limit_date")] = pd.NaT
    source.iloc[2, source.columns.get_loc("order_purchase_timestamp")] = pd.NaT
    source.iloc[2, source.columns.get_loc("shipping_limit_date")] = pd.Timestamp("2020-01-01")

    cleaned, summary, records = flag_deadline_issues(source)

    assert cleaned["needs_deadline_review"].tolist() == [False, False, True]
    table = summary.set_index("check")
    assert table.loc["deadline_before_purchase", "comparable_item_rows"] == 1
    assert table.loc["deadline_before_purchase", "unknown_item_rows"] == 2
    assert table.loc["deadline_after_purchase_year", "unknown_item_rows"] == 1
    assert pd.isna(records.iloc[0]["deadline_before_purchase"])
    assert bool(records.iloc[0]["deadline_after_purchase_year"])
    assert str(records["deadline_before_purchase"].dtype) == "boolean"


def test_deadline_review_without_purchase_reference_does_not_invent_a_cutoff(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    source["order_purchase_timestamp"] = pd.Series(
        pd.NaT, index=source.index, dtype=source["order_purchase_timestamp"].dtype,
    )
    cleaned, summary, records = flag_deadline_issues(source)

    assert not cleaned["needs_deadline_review"].any()
    assert summary["unknown_item_rows"].tolist() == [3, 3]
    assert summary["comparable_item_rows"].tolist() == [0, 0]
    assert records.empty
    assert "Unavailable" in summary.iloc[1]["reference"]


def test_deadline_year_reference_uses_the_full_input_not_each_order_year(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    source.iloc[2, source.columns.get_loc("order_purchase_timestamp")] = pd.Timestamp("2018-01-01")
    source.iloc[0, source.columns.get_loc("shipping_limit_date")] = pd.Timestamp("2018-01-02")
    source.iloc[2, source.columns.get_loc("shipping_limit_date")] = pd.Timestamp("2018-01-03")

    cleaned, summary, records = flag_deadline_issues(source)

    assert not cleaned["needs_deadline_review"].any()
    assert records.empty
    assert summary.iloc[1]["reference"] == "Latest observed purchase year: 2018"


def test_deadline_equal_to_purchase_is_not_an_ordering_issue(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    source["shipping_limit_date"] = source["order_purchase_timestamp"].copy()
    cleaned, summary, _ = flag_deadline_issues(source)
    assert not cleaned["needs_deadline_review"].any()
    assert summary["affected_item_rows"].eq(0).all()


@pytest.mark.parametrize("column", ["shipping_limit_date", "order_purchase_timestamp"])
def test_deadline_review_rejects_string_timestamps(cleaning_full_frame, column):
    source = cleaning_full_frame.copy(deep=True)
    source[column] = source[column].astype("string")
    with pytest.raises(TypeError, match="datetime dtype"):
        flag_deadline_issues(source)


def test_deadline_review_refuses_to_overwrite_an_existing_flag(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    source["needs_deadline_review"] = False
    with pytest.raises(ValueError, match="reserved column"):
        flag_deadline_issues(source)


def test_deadline_review_rejects_duplicate_item_keys(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    source.iloc[1, source.columns.get_loc("order_item_id")] = 1
    with pytest.raises(ValueError, match="not unique"):
        flag_deadline_issues(source)


def test_core_timeline_missing_endpoint_is_unknown_without_a_false_reversal(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    source.iloc[2, source.columns.get_loc("order_delivered_customer_date")] = pd.NaT
    before = source.copy(deep=True)
    cleaned, summary, core_summary = flag_timeline_issues(source)

    assert not cleaned["has_invalid_timeline"].any()
    check = summary.set_index("check").loc["customer_before_carrier"]
    assert check["unknown_orders"] == 1
    assert check["affected_orders"] == 0
    assert core_summary.loc[0, "orders_missing_any_core_timestamp"] == 1
    pd.testing.assert_frame_equal(source, before, check_exact=True)


def test_full_cleaning_workflow_preserves_items_statuses_and_source(cleaning_full_frame):
    source = cleaning_full_frame.copy(deep=True)
    source.iloc[0:2, source.columns.get_loc("customer_city")] = "  city   a "
    source.iloc[[0, 2], source.columns.get_loc("product_weight_g")] = 0.0
    source.iloc[2, source.columns.get_loc("order_status")] = "canceled"
    before = source.copy(deep=True)

    cleaned, details = clean_order_items(source)
    again, repeated_details = clean_order_items(source)

    assert cleaned.shape == (3, 28)
    assert list(cleaned.columns) == CLEANED_ORDER_ITEMS_SCHEMA.columns
    assert cleaned["has_invalid_physical_measurement"].tolist() == [True, False, True]
    assert cleaned["order_status"].tolist() == ["delivered", "delivered", "canceled"]
    assert cleaned.iloc[1]["freight_value"] == 0.0
    assert details["text_changes"]["changed_item_rows"].sum() == 2
    assert details["output_checks"]["passed"].all()
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    pd.testing.assert_frame_equal(cleaned, again, check_exact=True)
    for name in details:
        pd.testing.assert_frame_equal(details[name], repeated_details[name], check_exact=True)
    assert len(FULL_MERGE_SCHEMA.columns) == 25
    assert all(column not in FULL_MERGE_SCHEMA.dtypes for column in (
        "has_invalid_physical_measurement", "has_invalid_timeline", "needs_deadline_review",
    ))


@pytest.mark.parametrize("column,value", [
    ("price", 999.0),
    ("order_status", "canceled"),
    ("order_delivered_customer_date", pd.Timestamp("2019-01-01")),
    ("customer_city", "a different city"),
    ("product_height_cm", pd.NA),
    ("has_invalid_physical_measurement", True),
    ("has_invalid_timeline", True),
    ("needs_deadline_review", True),
])
def test_output_validation_rejects_unapproved_cell_or_flag_changes(
    cleaning_full_frame, column, value,
):
    cleaned, _ = clean_order_items(cleaning_full_frame)
    cleaned.iloc[0, cleaned.columns.get_loc(column)] = value
    with pytest.raises(AssertionError):
        validate_cleaning_output(cleaning_full_frame, cleaned)


def test_output_validation_rejects_row_reordering(cleaning_full_frame):
    cleaned, _ = clean_order_items(cleaning_full_frame)
    with pytest.raises(AssertionError):
        validate_cleaning_output(cleaning_full_frame, cleaned.iloc[::-1])


@pytest.mark.parametrize("fallback", [False, True])
def test_processed_roundtrip_preserves_types_and_supports_csv_fallback(
    cleaning_full_frame, tmp_path, monkeypatch, fallback,
):
    cleaned, _ = clean_order_items(cleaning_full_frame)
    monkeypatch.setattr("src.data_loader.PROCESSED_DIR", tmp_path)
    if fallback:
        def unavailable_writer(*args, **kwargs):
            raise ImportError("Simulated unavailable Parquet engine")
        monkeypatch.setattr("src.pipeline.io.dataset_io.write_parquet_dataset", unavailable_writer)
    saved = save_cleaned_data(cleaned, tmp_path)
    assert saved.name == CLEANED_DATASET_NAME + (".csv" if fallback else ".parquet")
    restored = load_data(CLEANED_DATASET_NAME, stage="processed")
    assert_cleaned_roundtrip(cleaned, restored)
    assert restored.iloc[0]["customer_zip_code_prefix"] == "00123"
    assert str(restored["shipping_limit_date"].dtype) == "datetime64[us]"
    for column in ("has_invalid_physical_measurement", "has_invalid_timeline", "needs_deadline_review"):
        assert restored[column].dtype == bool
    if fallback:
        def forbidden_parquet_read(*args, **kwargs):
            pytest.fail("Explicit CSV selection must not inspect Parquet")
        monkeypatch.setattr("src.pipeline.io.dataset_io.read_parquet_dataset", forbidden_parquet_read)
        assert_cleaned_roundtrip(cleaned, load_data(
            CLEANED_DATASET_NAME, stage="processed", file_format="csv",
        ))
    assert not list(tmp_path.glob(".merge-*.tmp"))


def test_processed_loader_protects_registered_contract_and_rejects_parser_overrides():
    with pytest.raises(ValueError, match="fixed registered schema"):
        load_data(CLEANED_DATASET_NAME, stage="processed", schema=replace(
            CLEANED_ORDER_ITEMS_SCHEMA, unique_key=[],
        ))
    with pytest.raises(TypeError, match="Unsupported processed"):
        load_data(CLEANED_DATASET_NAME, stage="processed", dtype="string")


def test_processed_loader_reports_both_missing_formats(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data_loader.PROCESSED_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="either format") as error:
        load_data(CLEANED_DATASET_NAME, stage="processed")
    assert ".parquet" in str(error.value) and ".csv" in str(error.value)


def test_clean_saver_validates_before_creating_output(cleaning_full_frame, tmp_path):
    directory = tmp_path / "not_created"
    with pytest.raises(ValueError, match="column mismatch"):
        save_cleaned_data(cleaning_full_frame, directory)
    assert not directory.exists()


def test_roundtrip_comparison_detects_changed_values(cleaning_full_frame):
    cleaned, _ = clean_order_items(cleaning_full_frame)
    restored = cleaned.reset_index(drop=True)
    restored.iloc[0, restored.columns.get_loc("price")] += 1
    with pytest.raises(AssertionError):
        assert_cleaned_roundtrip(cleaned, restored)


@pytest.mark.parametrize("start_directory", ["", "notebooks"])
@pytest.mark.parametrize("fallback", [False, True])
def test_notebook_executes_complete_cleaning_and_verifies_the_saved_format(
    cleaning_full_frame, tmp_path, monkeypatch, start_directory, fallback,
):
    source = cleaning_full_frame.copy(deep=True)
    before = source.copy(deep=True)
    original_loader = load_data
    monkeypatch.setattr("src.data_loader.PROCESSED_DIR", tmp_path)
    monkeypatch.setattr("src.cleaning.workflow.PROCESSED_DIR", tmp_path)
    if fallback:
        stale, _ = clean_order_items(source)
        stale["price"] += 100.0
        save_cleaned_data(stale, tmp_path)

        def unavailable_writer(*args, **kwargs):
            raise ImportError("Simulated unavailable Parquet engine")
        monkeypatch.setattr("src.pipeline.io.dataset_io.write_parquet_dataset", unavailable_writer)

    def staged_loader(name, stage="raw", **kwargs):
        if stage == "interim":
            assert name == "merged_order_items"
            return source
        return original_loader(name, stage=stage, **kwargs)

    monkeypatch.setattr("src.data_loader.load_data", staged_loader)
    workflow_spy = Mock(wraps=clean_order_items)
    monkeypatch.setattr("src.cleaning.clean_order_items", workflow_spy)
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace, displayed = {}, []
    with (
        patch("pathlib.Path.cwd", return_value=PROJECT_ROOT / start_directory),
        patch("IPython.display.display", side_effect=displayed.append),
        redirect_stdout(io.StringIO()),
    ):
        for position, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                code = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
                exec(compile(code, f"cleaning_cell_{position}", "exec"), namespace)

    assert workflow_spy.call_count == 1
    assert namespace["cleaned_df"].shape == (3, 28)
    assert namespace["saved_path"].suffix == (".csv" if fallback else ".parquet")
    assert namespace["cleaning_details"]["output_checks"]["passed"].all()
    assert_cleaned_roundtrip(namespace["cleaned_df"], namespace["reloaded_df"])
    pd.testing.assert_frame_equal(source, before, check_exact=True)
    assert displayed
    if fallback:
        # The default remains Parquet-first; the notebook must verify the new CSV.
        stale_read = original_loader(CLEANED_DATASET_NAME, stage="processed")
        assert stale_read.iloc[0]["price"] == before.iloc[0]["price"] + 100.0
        assert namespace["reloaded_df"].iloc[0]["price"] == before.iloc[0]["price"]


def test_notebook_has_no_report_or_document_export_calls():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "to_markdown", "write_text", "write_bytes", "to_json", "to_csv",
        "to_parquet", "to_excel", "savefig", "open",
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert name not in forbidden, f"Unexpected direct export call: {name}"


@pytest.mark.parametrize("change", ["rows", "row_order", "column_order", "dtype"])
def test_source_baseline_detects_structural_changes(cleaning_full_frame, change):
    baseline = capture_input_baseline(cleaning_full_frame)
    changed = cleaning_full_frame.copy(deep=True)
    if change == "rows":
        changed = changed.iloc[:-1]
    elif change == "row_order":
        changed = changed.iloc[::-1]
    elif change == "column_order":
        changed = changed[changed.columns[::-1]]
    else:
        changed["price"] = changed["price"].astype("float32")
    with pytest.raises(AssertionError):
        assert_input_unchanged(changed, baseline)


def test_full_workflow_rejects_unapproved_transform_changes(cleaning_full_frame, monkeypatch):
    import src.cleaning.workflow as workflow

    source = cleaning_full_frame.copy(deep=True)
    before = source.copy(deep=True)
    normalize = workflow.normalize_city_whitespace

    def faulty_transform(frame):
        cleaned, summary = normalize(frame)
        cleaned["price"] += 1.0
        return cleaned, summary

    monkeypatch.setattr(workflow, "normalize_city_whitespace", faulty_transform)
    with pytest.raises(AssertionError):
        clean_order_items(source)
    pd.testing.assert_frame_equal(source, before, check_exact=True)


def test_clean_csv_restores_true_flags_nulls_and_literals_after_corrupt_parquet(
    cleaning_full_frame, tmp_path, monkeypatch,
):
    source = cleaning_full_frame.copy(deep=True)
    source.iloc[[0, 2], source.columns.get_loc("product_weight_g")] = 0.0
    source.iloc[0:2, source.columns.get_loc("order_approved_at")] = pd.Timestamp("2017-01-01 09:00")
    source.iloc[2, source.columns.get_loc("order_delivered_customer_date")] = pd.NaT
    source.iloc[1, source.columns.get_loc("shipping_limit_date")] = pd.Timestamp("2020-01-01")
    source["product_category_name"] = pd.Series(["NA", "NULL", "NA"], index=source.index, dtype="string")
    cleaned, _ = clean_order_items(source)

    def unavailable_writer(*args, **kwargs):
        raise ImportError("Simulated unavailable Parquet engine")
    monkeypatch.setattr("src.pipeline.io.dataset_io.write_parquet_dataset", unavailable_writer)
    save_cleaned_data(cleaned, tmp_path)
    (tmp_path / (CLEANED_DATASET_NAME + ".parquet")).write_bytes(b"corrupt parquet")
    monkeypatch.setattr("src.data_loader.PROCESSED_DIR", tmp_path)
    restored = load_data(CLEANED_DATASET_NAME, stage="processed")

    assert_cleaned_roundtrip(cleaned, restored)
    assert restored["has_invalid_physical_measurement"].tolist() == [True, False, True]
    assert restored["has_invalid_timeline"].tolist() == [True, True, False]
    assert restored["needs_deadline_review"].tolist() == [False, True, False]
    assert restored["product_category_name"].tolist() == ["NA", "NULL", "NA"]
    assert pd.isna(restored.iloc[2]["order_delivered_customer_date"])
    assert pd.isna(restored.iloc[0]["product_weight_g"])


def test_clean_save_reports_both_failures_and_preserves_previous_file(
    cleaning_full_frame, tmp_path, monkeypatch,
):
    cleaned, _ = clean_order_items(cleaning_full_frame)
    saved = save_cleaned_data(cleaned, tmp_path)
    previous = saved.read_bytes()

    def failed_parquet(*args, **kwargs):
        raise ImportError("Simulated missing engine")

    def failed_csv(*args, **kwargs):
        raise OSError("Simulated CSV write failure")

    monkeypatch.setattr("src.pipeline.io.dataset_io.write_parquet_dataset", failed_parquet)
    monkeypatch.setattr("src.pipeline.io.dataset_io.write_csv_dataset", failed_csv)
    with pytest.raises(RuntimeError, match="either format") as error:
        save_cleaned_data(cleaned, tmp_path)
    assert ".parquet" in str(error.value) and ".csv" in str(error.value)
    assert isinstance(error.value.__cause__, ExceptionGroup)
    assert len(error.value.__cause__.exceptions) == 2
    assert saved.read_bytes() == previous
    assert not list(tmp_path.glob(".merge-*.tmp"))


def test_order_filter_retains_valid_data_without_mutating_or_aliasing_input(cleaning_full_frame):
    inspected, _ = clean_order_items(cleaning_full_frame)
    before = inspected.copy(deep=True)
    retained, details = filter_eligible_orders(inspected)

    pd.testing.assert_frame_equal(retained, before, check_exact=True)
    assert details["excluded_orders"].empty
    assert details["reasons"]["affected_orders"].eq(0).all()
    assert details["flow"].iloc[-1]["remaining_orders"] == 2
    assert details["flow"].iloc[-1]["remaining_item_rows"] == 3
    retained.iloc[0, retained.columns.get_loc("price")] += 1
    pd.testing.assert_frame_equal(inspected, before, check_exact=True)


@pytest.mark.parametrize("column,value,reason", [
    ("order_status", "shipped", "not_delivered"),
    ("order_approved_at", pd.NaT, "missing_required_timestamp"),
    ("has_invalid_timeline", True, "has_invalid_timeline"),
    ("has_invalid_physical_measurement", True, "has_invalid_physical_measurement"),
    ("needs_deadline_review", True, "needs_deadline_review"),
])
def test_order_filter_excludes_all_items_when_one_item_fails(
    cleaning_full_frame, column, value, reason,
):
    inspected, _ = clean_order_items(cleaning_full_frame)
    inspected.iloc[0, inspected.columns.get_loc(column)] = value
    before = inspected.copy(deep=True)
    retained, details = filter_eligible_orders(inspected)

    assert retained["order_id"].tolist() == [inspected.iloc[2]["order_id"]]
    assert retained.index.tolist() == [2]
    reason_row = details["reasons"].set_index("reason").loc[reason]
    assert reason_row["affected_orders"] == 1
    assert reason_row["affected_item_rows"] == 2
    excluded = details["excluded_orders"]
    assert len(excluded) == 1 and excluded.iloc[0]["item_rows"] == 2
    assert bool(excluded.iloc[0][reason])
    pd.testing.assert_frame_equal(inspected, before, check_exact=True)


@pytest.mark.parametrize("column", [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
])
def test_order_filter_requires_all_core_and_estimated_delivery_timestamps(
    cleaning_full_frame, column,
):
    inspected, _ = clean_order_items(cleaning_full_frame)
    inspected.iloc[2, inspected.columns.get_loc(column)] = pd.NaT
    retained, details = filter_eligible_orders(inspected)
    assert len(retained) == 2
    assert details["reasons"].set_index("reason").loc[
        "missing_required_timestamp", "affected_orders",
    ] == 1


def test_order_filter_counts_overlapping_reasons_without_double_counting_removals(cleaning_full_frame):
    inspected, _ = clean_order_items(cleaning_full_frame)
    inspected.iloc[0, inspected.columns.get_loc("order_status")] = "canceled"
    inspected.iloc[0, inspected.columns.get_loc("order_approved_at")] = pd.NaT
    inspected.iloc[1, inspected.columns.get_loc("has_invalid_physical_measurement")] = True
    inspected.iloc[2, inspected.columns.get_loc("needs_deadline_review")] = True
    retained, details = filter_eligible_orders(inspected)
    reasons = details["reasons"].set_index("reason")
    flow = details["flow"].set_index("step")

    assert retained.empty
    assert reasons["affected_orders"].sum() == 4
    assert len(details["excluded_orders"]) == 2
    assert flow["removed_orders"].sum() == 2
    assert flow["removed_item_rows"].sum() == 3
    assert flow.loc["not_delivered", "removed_orders"] == 1
    assert flow.loc["missing_required_timestamp", "removed_orders"] == 0
    assert flow.loc["has_invalid_physical_measurement", "removed_orders"] == 0
    assert flow.loc["needs_deadline_review", "removed_orders"] == 1
    assert flow.iloc[-1]["remaining_orders"] == 0
    assert flow.iloc[-1]["remaining_item_rows"] == 0
    assert retained.dtypes.equals(inspected.dtypes)


def test_order_filter_preserves_optional_nulls_and_zero_freight(cleaning_full_frame):
    inspected, _ = clean_order_items(cleaning_full_frame)
    for column in ["product_category_name", "product_weight_g"]:
        inspected.iloc[0, inspected.columns.get_loc(column)] = pd.NA
    before = inspected.copy(deep=True)
    retained, details = filter_eligible_orders(inspected)
    pd.testing.assert_frame_equal(retained, before, check_exact=True)
    assert retained.iloc[1]["freight_value"] == 0.0
    assert details["excluded_orders"].empty


def test_order_filter_treats_missing_status_as_not_delivered(cleaning_full_frame):
    inspected, _ = clean_order_items(cleaning_full_frame)
    inspected.iloc[2, inspected.columns.get_loc("order_status")] = pd.NA
    retained, details = filter_eligible_orders(inspected)
    assert len(retained) == 2
    assert details["reasons"].set_index("reason").loc["not_delivered", "affected_orders"] == 1


def test_order_filter_supports_empty_input_with_stable_reports(cleaning_full_frame):
    inspected, _ = clean_order_items(cleaning_full_frame)
    empty = inspected.iloc[:0]
    retained, details = filter_eligible_orders(empty)
    pd.testing.assert_frame_equal(retained, empty, check_exact=True)
    assert len(details["reasons"]) == 5
    assert len(details["flow"]) == 6
    assert details["reasons"]["affected_orders"].eq(0).all()
    assert details["flow"]["remaining_item_rows"].eq(0).all()
    assert details["excluded_orders"].empty


@pytest.mark.parametrize("problem", ["missing_flag", "nullable_flag", "duplicate_key"])
def test_order_filter_rejects_invalid_inspection_contract(cleaning_full_frame, problem):
    inspected, _ = clean_order_items(cleaning_full_frame)
    if problem == "missing_flag":
        inspected = inspected.drop(columns="needs_deadline_review")
    elif problem == "nullable_flag":
        inspected["needs_deadline_review"] = inspected["needs_deadline_review"].astype("boolean")
        inspected.iloc[0, inspected.columns.get_loc("needs_deadline_review")] = pd.NA
    else:
        inspected.iloc[1, inspected.columns.get_loc("order_item_id")] = 1
    with pytest.raises(ValueError):
        filter_eligible_orders(inspected)

