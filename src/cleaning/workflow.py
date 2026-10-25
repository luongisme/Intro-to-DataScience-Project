"""Reusable full cleaning workflow with explicit, schema-aware saving."""

from pathlib import Path

import pandas as pd

from src.cleaning.measurements import clean_physical_measurements
from src.cleaning.schema import CLEANED_DATASET_NAME, CLEANED_ORDER_ITEMS_SCHEMA
from src.cleaning.text import normalize_city_whitespace
from src.cleaning.timeline import flag_deadline_issues, flag_timeline_issues
from src.cleaning.validation import (
    assert_input_unchanged,
    capture_input_baseline,
    validate_cleaning_input,
    validate_cleaning_output,
)
from src.paths import PROCESSED_DIR
from src.pipeline.io.dataset_io import save_dataset


def clean_order_items(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Clean the full merged input and return data with in-memory diagnostics.

    Validate input, normalize city whitespace, invalidate physical cells, then
    add core-timeline and item-deadline review flags. Verify permitted changes
    and input preservation before returning. Start each run from the original
    25-column full merge; all statuses and items are retained.
    Diagnostics are displayable DataFrames. No loading, saving, report writing
    or display occurs in this function.
    """
    baseline = capture_input_baseline(df)
    validate_cleaning_input(df)
    cleaned, text_changes = normalize_city_whitespace(df)
    cleaned, measurements = clean_physical_measurements(cleaned)
    cleaned, timeline, core_timestamps = flag_timeline_issues(cleaned)
    cleaned, deadlines, deadline_records = flag_deadline_issues(cleaned)
    output_checks = validate_cleaning_output(df, cleaned)
    assert_input_unchanged(df, baseline)
    output_checks.loc[len(output_checks)] = ["Loaded source unchanged", True]
    return cleaned, {
        "text_changes": text_changes,
        "measurements": measurements,
        "timeline": timeline,
        "core_timestamps": core_timestamps,
        "deadlines": deadlines,
        "deadline_records": deadline_records,
        "output_checks": output_checks,
    }


def save_cleaned_data(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    """Save a clean workflow result in processed using the fixed clean schema.

    Delegate validation, atomic replacement, Parquet-first saving, CSV fallback
    and error guidance to the shared saver. Return the actual successful Path.
    Read that format explicitly for roundtrip checks if an older other-format
    file exists. A custom directory supports isolated runs without new metadata.
    """
    directory = PROCESSED_DIR if output_dir is None else output_dir
    return save_dataset(
        df, directory, CLEANED_ORDER_ITEMS_SCHEMA, dataset_name=CLEANED_DATASET_NAME,
    )
