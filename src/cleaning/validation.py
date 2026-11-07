"""Input, allowed-change and storage roundtrip checks for data cleaning."""

import pandas as pd

from src.cleaning.measurements import INVALID_PHYSICAL_FLAG, PHYSICAL_COLUMNS
from src.cleaning.schema import (
    CLEANED_ORDER_ITEMS_SCHEMA, FLAGGED_ORDER_ITEMS_SCHEMA, REQUIRED_TIMESTAMPS,
)
from src.cleaning.text import CITY_COLUMNS, inspect_text_fields
from src.cleaning.timeline import (
    CORE_TIMESTAMP_COLUMNS, DEADLINE_REVIEW_FLAG, INVALID_TIMELINE_FLAG,
)
from src.eda.checks import profile_entity_consistency, profile_keys
from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA
from src.pipeline.contracts.validator import validate_table_contract


def capture_input_baseline(df: pd.DataFrame) -> dict:
    """Capture shape, columns, dtypes and indexed row hashes without copying data."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.columns.has_duplicates:
        raise ValueError("The input must have unique column names.")
    return {
        "shape": df.shape,
        "columns": df.columns.copy(),
        "dtypes": df.dtypes.copy(),
        "row_hashes": pd.util.hash_pandas_object(df, index=True).copy(),
    }


def validate_cleaning_input(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Validate the full merge contract and repeated source entity attributes.

    Reject empty inputs, schema/key/non-null violations, invalid observed text
    formats/domains, invalid entity keys or conflicting attributes. Return key,
    text and consistency profiles for display. Optional missing text attributes
    are counted separately and preserved. Never modify input values or types.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("The full merged input must contain at least one item row.")
    validate_table_contract(df, FULL_MERGE_SCHEMA, table_name="merged_order_items")
    text_profile = inspect_text_fields(df)
    invalid_text = text_profile["summary"].loc[
        text_profile["summary"]["invalid_item_rows"].gt(0),
        ["column", "invalid_item_rows"],
    ]
    if not invalid_text.empty:
        counts = dict(invalid_text.itertuples(index=False, name=None))
        raise ValueError(
            f"Cleaning input has invalid text values (item counts): {counts}. "
            "Use src.cleaning.inspect_text_fields(df)['issues'] to inspect "
            "original values and item positions. Resolve these values before cleaning."
        )
    keys = profile_keys(df)
    consistency = profile_entity_consistency(df)
    summary = consistency["summary"]
    invalid_keys = summary.loc[summary["invalid_key_rows"].gt(0), "entity"].tolist()
    conflicts = summary.loc[summary["conflicted_entities"].gt(0), "entity"].tolist()
    if invalid_keys or conflicts:
        raise ValueError(
            "Cleaning input has invalid entity keys or conflicting attributes: "
            f"invalid_keys={invalid_keys}, conflicts={conflicts}. "
            "Inspect profile_keys(df) and profile_entity_consistency(df)."
        )
    return {
        "key_profile": keys,
        "entity_consistency": summary,
        "attribute_consistency": consistency["attributes"],
        "conflicts": consistency["conflicts"],
        "text_formats": text_profile["summary"],
    }


def assert_input_unchanged(df: pd.DataFrame, baseline: dict) -> None:
    """Raise if shape, column order, dtypes, values or row order changed."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(baseline, dict):
        raise TypeError("baseline must come from capture_input_baseline().")
    if df.shape != baseline["shape"]:
        raise AssertionError("Input shape changed during cleaning.")
    pd.testing.assert_index_equal(df.columns, baseline["columns"])
    pd.testing.assert_series_equal(df.dtypes, baseline["dtypes"])
    pd.testing.assert_series_equal(
        pd.util.hash_pandas_object(df, index=True), baseline["row_hashes"],
    )


def validate_inspected_output(
    source: pd.DataFrame,
    inspected: pd.DataFrame,
) -> pd.DataFrame:
    """Verify all preparation changes and quality flags before order filtering.

    Preserve item keys, index, row order and all original dtypes. Permit only
    city whitespace normalization and non-positive physical cells becoming
    missing. Verify all three quality flags against original observations.
    Raise on any mismatch; return a displayable table of successful checks.
    The caller must separately compare source to its pre-cleaning baseline.
    """
    validate_table_contract(source, FULL_MERGE_SCHEMA, table_name="cleaning_source")
    validate_table_contract(inspected, FLAGGED_ORDER_ITEMS_SCHEMA, table_name="inspected_order_items")
    pd.testing.assert_index_equal(source.index, inspected.index, obj="Inspection row index")
    pd.testing.assert_frame_equal(
        source[FULL_MERGE_SCHEMA.unique_key], inspected[FULL_MERGE_SCHEMA.unique_key],
        check_exact=True, obj="Preserved item keys and row order",
    )
    mutable = {*CITY_COLUMNS, *PHYSICAL_COLUMNS}
    preserved = [column for column in source.columns if column not in mutable]
    pd.testing.assert_frame_equal(
        source[preserved], inspected[preserved], check_exact=True,
        obj="Source fields outside the cleaning rules",
    )
    for column in CITY_COLUMNS:
        expected = source[column].str.strip().str.replace(r"\s+", " ", regex=True)
        pd.testing.assert_series_equal(
            expected, inspected[column], check_exact=True, obj=f"City whitespace: {column}",
        )
    invalid_measurements = source[list(PHYSICAL_COLUMNS)].le(0).fillna(False)
    for column in PHYSICAL_COLUMNS:
        expected = source[column].mask(invalid_measurements[column], pd.NA)
        pd.testing.assert_series_equal(
            expected, inspected[column], check_exact=True, obj=f"Physical cells: {column}",
        )
    expected_timeline = pd.Series(False, index=source.index, dtype=bool)
    for start, end in zip(CORE_TIMESTAMP_COLUMNS, CORE_TIMESTAMP_COLUMNS[1:]):
        expected_timeline |= source[end].lt(source[start]).fillna(False)
    deadline = source["shipping_limit_date"]
    purchase = source["order_purchase_timestamp"]
    expected_deadline = deadline.lt(purchase).fillna(False)
    latest_purchase = purchase.max()
    if pd.notna(latest_purchase):
        expected_deadline |= deadline.dt.year.gt(latest_purchase.year).fillna(False)
    expected_flags = {
        INVALID_PHYSICAL_FLAG: invalid_measurements.any(axis=1),
        INVALID_TIMELINE_FLAG: expected_timeline,
        DEADLINE_REVIEW_FLAG: expected_deadline,
    }
    for column, values in expected_flags.items():
        expected = pd.Series(values.to_numpy(dtype=bool), index=source.index, name=column)
        pd.testing.assert_series_equal(
            expected, inspected[column], check_exact=True, obj=f"Quality flag: {column}",
        )
    return pd.DataFrame({
        "check": [
            "Inspection schema and dtypes", "Item keys and row order",
            "Preserved source fields", "City whitespace normalization",
            "Physical measurement handling", "Quality flag consistency",
        ],
        "passed": True,
    })


def validate_cleaning_output(
    source: pd.DataFrame,
    cleaned: pd.DataFrame,
) -> pd.DataFrame:
    """Verify the exact retained order cohort and final 25-column values.

    Derive eligibility independently from original status, required timestamps,
    physical values and full-input temporal comparisons. Reject partial-order
    removal, missing eligible items or retained ineligible orders. Preserve
    original indexes, relative row order and every retained value except city
    whitespace normalization. Source preservation uses a separate baseline.
    """
    validate_table_contract(source, FULL_MERGE_SCHEMA, table_name="cleaning_source")
    validate_table_contract(cleaned, CLEANED_ORDER_ITEMS_SCHEMA, table_name="cleaned_order_items")
    invalid_physical = source[list(PHYSICAL_COLUMNS)].le(0).fillna(False).any(axis=1)
    invalid_timeline = pd.Series(False, index=source.index, dtype=bool)
    for start, end in zip(CORE_TIMESTAMP_COLUMNS, CORE_TIMESTAMP_COLUMNS[1:]):
        invalid_timeline |= source[end].lt(source[start]).fillna(False)
    purchase = source["order_purchase_timestamp"]
    deadline = source["shipping_limit_date"]
    deadline_review = deadline.lt(purchase).fillna(False)
    latest_purchase = purchase.max()
    if pd.notna(latest_purchase):
        deadline_review |= deadline.dt.year.gt(latest_purchase.year).fillna(False)
    failures = (
        ~source["order_status"].eq("delivered").fillna(False)
        | source[list(REQUIRED_TIMESTAMPS)].isna().any(axis=1)
        | invalid_physical | invalid_timeline | deadline_review
    )
    excluded_ids = source.loc[failures, "order_id"]
    expected_source = source.loc[~source["order_id"].isin(excluded_ids)]
    pd.testing.assert_index_equal(expected_source.index, cleaned.index, obj="Retained row index")
    pd.testing.assert_frame_equal(
        expected_source[FULL_MERGE_SCHEMA.unique_key], cleaned[FULL_MERGE_SCHEMA.unique_key],
        check_exact=True, obj="Eligible whole orders, item keys and row order",
    )
    preserved = [column for column in source.columns if column not in CITY_COLUMNS]
    pd.testing.assert_frame_equal(
        expected_source[preserved], cleaned[preserved], check_exact=True,
        obj="Retained source values",
    )
    for column in CITY_COLUMNS:
        expected = expected_source[column].str.strip().str.replace(r"\s+", " ", regex=True)
        pd.testing.assert_series_equal(
            expected, cleaned[column], check_exact=True, obj=f"Retained city: {column}",
        )
    return pd.DataFrame({
        "check": [
            "Final schema and dtypes", "Eligible whole-order selection",
            "Retained item keys and row order", "Retained source values",
            "City whitespace normalization", "Required timestamps and delivered status",
        ],
        "passed": True,
    })



def assert_cleaned_roundtrip(expected: pd.DataFrame, restored: pd.DataFrame) -> None:
    """Require exact clean values, columns, dtypes and row order after storage.

    The shared Parquet/CSV contract stores item keys and row order, not pandas
    index labels. Ignore only those in-memory labels when comparing results.
    """
    for label, frame in (("expected_clean", expected), ("restored_clean", restored)):
        validate_table_contract(frame, CLEANED_ORDER_ITEMS_SCHEMA, table_name=label)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), restored.reset_index(drop=True),
        check_exact=True, obj="Clean storage roundtrip",
    )

