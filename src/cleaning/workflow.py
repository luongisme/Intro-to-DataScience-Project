"""Reusable full cleaning workflow with explicit, schema-aware saving."""

from pathlib import Path

import pandas as pd

from src.cleaning.filtering import filter_eligible_orders
from src.cleaning.measurements import clean_physical_measurements
from src.cleaning.schema import CLEANED_DATASET_NAME, CLEANED_ORDER_ITEMS_SCHEMA
from src.cleaning.text import normalize_city_whitespace
from src.cleaning.timeline import flag_deadline_issues, flag_timeline_issues
from src.cleaning.validation import (
    assert_input_unchanged,
    capture_input_baseline,
    validate_cleaning_input,
    validate_cleaning_output,
    validate_inspected_output,
)
from src.paths import PROCESSED_DIR
from src.pipeline.io.dataset_io import save_dataset


def inspect_order_items(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Prepare and inspect every full-merge item before eligibility filtering.

    Return the 28-column table with three quality flags and source-grain
    diagnostic tables. Validate allowed changes and input preservation. All
    original items and statuses remain available at this inspection stage.
    """
    baseline = capture_input_baseline(df)
    validate_cleaning_input(df)
    inspected, text_changes = normalize_city_whitespace(df)
    inspected, measurements = clean_physical_measurements(inspected)
    inspected, timeline, core_timestamps = flag_timeline_issues(inspected)
    inspected, deadlines, deadline_records = flag_deadline_issues(inspected)
    inspection_checks = validate_inspected_output(df, inspected)
    assert_input_unchanged(df, baseline)
    inspection_checks.loc[len(inspection_checks)] = ["Loaded source unchanged", True]
    return inspected, {
        "text_changes": text_changes,
        "measurements": measurements,
        "timeline": timeline,
        "core_timestamps": core_timestamps,
        "deadlines": deadlines,
        "deadline_records": deadline_records,
        "inspection_checks": inspection_checks,
    }


def clean_order_items(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return eligible delivered items with the final 25-column clean contract.

    Inspect the complete original input, filter whole orders, remove quality
    flag columns, then verify eligibility and retained values against source.
    Diagnostics distinguish full-input preparation from sequential exclusion
    and the final retained dataset. No loading, display or saving occurs here.
    """
    baseline = capture_input_baseline(df)
    inspected, details = inspect_order_items(df)
    retained, exclusions = filter_eligible_orders(inspected)
    cleaned = retained.loc[:, CLEANED_ORDER_ITEMS_SCHEMA.columns].copy(deep=True)
    output_checks = validate_cleaning_output(df, cleaned)
    assert_input_unchanged(df, baseline)
    output_checks.loc[len(output_checks)] = ["Loaded source unchanged", True]
    details.update({
        "filter_reasons": exclusions["reasons"],
        "filter_flow": exclusions["flow"],
        "excluded_orders": exclusions["excluded_orders"],
        "dataset_summary": pd.DataFrame([
            ["Item rows", len(df), len(cleaned), len(df) - len(cleaned)],
            ["Orders", df["order_id"].nunique(), cleaned["order_id"].nunique(),
             df["order_id"].nunique() - cleaned["order_id"].nunique()],
            ["Columns", len(df.columns), len(cleaned.columns), 0],
        ], columns=["metric", "input", "retained", "removed"]),
        "output_checks": output_checks,
    })
    return cleaned, details



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
