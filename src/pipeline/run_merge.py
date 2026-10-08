"""Production CLI script: executes load, parse, merge, validate, and save."""

import logging
import sys
from pathlib import Path
import pandas as pd

from src.data_loader import load_data
from src.pipeline.schema import (
    CUSTOMERS_SPEC,
    ORDER_ITEMS_SPEC,
    ORDERS_SPEC,
    PRODUCTS_SPEC,
    SELLERS_SPEC,
    TableLoadSpec,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("olist_pipeline")


def _load_and_prepare(spec: TableLoadSpec) -> pd.DataFrame:
    """Load table via load_data with usecols/dtype and parse dates with errors='raise'.

    Args:
        spec: TableLoadSpec containing usecols, dtypes, and date columns.

    Returns:
        pd.DataFrame: Cleanly parsed DataFrame.

    Raises:
        FileNotFoundError: If file is missing.
        ValueError: If columns are missing or date parsing fails.
    """
    logger.info("Loading table '%s'...", spec.dataset_name)
    df = load_data(
        name=spec.dataset_name,
        usecols=spec.usecols,
        dtype=spec.dtypes,
    )

    missing_expected = set(spec.usecols) - set(df.columns)
    if missing_expected:
        raise ValueError(
            f"Table '{spec.dataset_name}' missing expected columns: {sorted(missing_expected)}"
        )

    # Parse timestamps with errors='raise'
    for date_col in spec.date_columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="raise")

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