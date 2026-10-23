"""Production CLI script: executes load, parse, merge, validate, and save."""

import logging
import sys
from pathlib import Path
import pandas as pd

from src.paths import INTERIM_DIR
from src.pipeline.api import merge_data
from src.pipeline.io.dataset_io import save_dataset
from src.pipeline.io.loader import load_table
from src.pipeline.merging.merger import OlistDataMerger
from src.pipeline.merging.planner import MERGE_TABLE_SPECS
from src.pipeline.merging.registry import FULL_DATASET_NAME
from src.pipeline.contracts.schema import (
    CUSTOMERS_SPEC,
    FULL_MERGE_SCHEMA,
    ORDER_ITEMS_SPEC,
    ORDERS_SPEC,
    PRODUCTS_SPEC,
    SELLERS_SPEC,
    TableLoadSpec,
)
from src.pipeline.contracts.validator import MergeValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("olist_pipeline")


def _load_and_prepare(spec: TableLoadSpec) -> pd.DataFrame:
    """Load a validated raw table through the shared typed loader.

    Keep the existing entry point for pipeline and notebook compatibility.
    Delegate column ordering, dtype normalization and source-key validation
    to load_table without maintaining a separate parsing implementation.

    Raises:
        FileNotFoundError: If the registered source file is missing.
        ValueError: If the load specification, values or source keys are invalid.
    """
    logger.info("Loading table '%s'...", spec.dataset_name)
    df = load_table(spec)
    logger.info("Loaded '%s' with shape %s", spec.dataset_name, df.shape)
    return df


def save_merged_dataset(merged_df: pd.DataFrame, output_dir: Path) -> Path:
    """Save the full merged table through the shared dataset storage workflow.

    Keep the existing positional/keyword arguments and Path return contract.
    Use FULL_MERGE_SCHEMA and FULL_DATASET_NAME so the legacy entry point
    retains the complete item-level dataset's contract and filename stem.
    Use save_dataset directly for partial combinations or custom schemas.

    Delegate validation before directory creation, atomic file replacement,
    Parquet-to-CSV fallback, notifications and error reporting to save_dataset.
    Do not normalize/clean inputs or maintain a separate writer/fallback here.

    Args:
        merged_df: Full merged table already satisfying FULL_MERGE_SCHEMA.
        output_dir: Destination directory, created after input validation.

    Returns:
        Actual saved Parquet or CSV path under the legacy full dataset name.

    Raises:
        TypeError: If input argument types are invalid.
        ValueError: If the full table contract or output path is invalid.
        OSError: If the shared output directory cannot be created/used.
        RuntimeError: If both format-specific atomic save attempts fail.
    """
    return save_dataset(
        merged_df, output_dir, FULL_MERGE_SCHEMA, dataset_name=FULL_DATASET_NAME
    )


def run_pipeline() -> pd.DataFrame:
    """Strictly merge and save the registered full Olist dataset.

    Delegate raw loading, dtype handling, preset planning, foreign-key checks
    and output validation to merge_data for all registered merge tables.
    Reuse the canonical table registry instead of repeating the five-table
    configuration or maintaining a second validation policy here.

    Save the resulting data through save_merged_dataset in INTERIM_DIR.
    Preserve Parquet-to-CSV fallback and the existing no-argument call and
    DataFrame return contract used by the notebook, tests and CLI. The merge
    result's schema/report remain in memory; this entry point returns data
    only. Call merge_data directly when schema and report are also needed.

    Do not filter rows, run legacy soft checks or write report/documents.
    Loading/merge failures propagate before saving. Save failures propagate
    without logging successful completion or returning partial results.

    Returns:
        The validated full item-level DataFrame after a successful save.

    Raises:
        Exception: Loading, merge validation or dataset saving failures
            propagate from the shared functions unchanged.
    """
    logger.info("========== STARTING OLIST MERGE PIPELINE ==========")

    result = merge_data(tables=list(MERGE_TABLE_SPECS))
    save_merged_dataset(result.data, output_dir=INTERIM_DIR)

    logger.info("========== PIPELINE COMPLETED SUCCESSFULLY ==========")
    return result.data


if __name__ == "__main__":
    run_pipeline()