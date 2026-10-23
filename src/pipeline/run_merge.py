"""Production CLI script: executes load, parse, merge, validate, and save."""

import logging
import sys
from pathlib import Path
from typing import List
import pandas as pd

from src.paths import INTERIM_DIR
from src.pipeline.dataset_io import save_dataset
from src.pipeline.loader import load_table
from src.pipeline.merger import OlistDataMerger
from src.pipeline.registry import FULL_DATASET_NAME
from src.pipeline.schema import (
    CUSTOMERS_SPEC,
    FULL_MERGE_SCHEMA,
    ORDER_ITEMS_SPEC,
    ORDERS_SPEC,
    PRODUCTS_SPEC,
    SELLERS_SPEC,
    TableLoadSpec,
)
from src.pipeline.validator import MergeValidator

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
    """Execute end-to-end data merge pipeline."""
    logger.info("========== STARTING OLIST MERGE PIPELINE ==========")

    # 1. Load and parse 5 tables
    order_items = _load_and_prepare(ORDER_ITEMS_SPEC)
    orders = _load_and_prepare(ORDERS_SPEC)
    customers = _load_and_prepare(CUSTOMERS_SPEC)
    products = _load_and_prepare(PRODUCTS_SPEC)
    sellers = _load_and_prepare(SELLERS_SPEC)

    # 2. Merge tables
    merger = OlistDataMerger()
    merged_df = merger.merge_tables(
        order_items=order_items,
        orders=orders,
        customers=customers,
        products=products,
        sellers=sellers,
    )

    # 3. Validate
    validator = MergeValidator()
    validator.validate_hard_rules(
        merged_df=merged_df,
        base_order_items=order_items,
        orders=orders,
        products=products,
        sellers=sellers,
    )
    validator.run_soft_checks(merged_df=merged_df)

    # 4. Save
    save_merged_dataset(merged_df, output_dir=INTERIM_DIR)

    logger.info("========== PIPELINE COMPLETED SUCCESSFULLY ==========")
    return merged_df


if __name__ == "__main__":
    run_pipeline()