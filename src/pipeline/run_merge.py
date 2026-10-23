"""Production CLI script: executes load, parse, merge, validate, and save."""

import logging
import sys
from pathlib import Path
from typing import List
import pandas as pd

from src.paths import INTERIM_DIR
from src.pipeline.loader import load_table
from src.pipeline.merger import OlistDataMerger
from src.pipeline.schema import (
    CUSTOMERS_SPEC,
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
    """Save DataFrame to processed folder as Parquet with CSV fallback.

    Args:
        merged_df: Merged output DataFrame.
        output_dir: Destination directory (PROCESSED_DIR).

    Returns:
        Path: Path to saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "merged_order_items.parquet"

    try:
        merged_df.to_parquet(parquet_path, index=False, engine="pyarrow")
        logger.info("Successfully saved output as Parquet to: %s", parquet_path)
        return parquet_path
    except (ImportError, ModuleNotFoundError) as err:
        csv_path = output_dir / "merged_order_items.csv"
        logger.warning(
            "pyarrow engine not available (%s). Falling back to CSV: %s",
            err,
            csv_path,
        )
        merged_df.to_csv(csv_path, index=False)
        return csv_path

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