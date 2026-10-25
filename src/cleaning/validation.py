"""Read-only input checks and preservation baseline for data cleaning."""

import pandas as pd

from src.cleaning.text import inspect_text_fields
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
