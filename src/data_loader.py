"""Load raw sources or schema-aware interim and processed datasets."""

from typing import Any, Dict
from pathlib import Path

import pandas as pd

from src.paths import RAW_DIR, INTERIM_DIR, PROCESSED_DIR
from src.cleaning.schema import CLEANED_DATASET_NAME, CLEANED_ORDER_ITEMS_SCHEMA
from src.pipeline.io.dataset_io import read_dataset
from src.pipeline.merging.registry import build_dataset_registry
from src.pipeline.contracts.schema import TableSchema


# Raw datasets
DATASETS: Dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


# Canonical saved-dataset schemas; actual format selection belongs to I/O.
INTERIM_DATASETS: Dict[str, Dict[str, Any]] = build_dataset_registry()
PROCESSED_DATASETS: Dict[str, Dict[str, Any]] = {
    CLEANED_DATASET_NAME: {"schema": CLEANED_ORDER_ITEMS_SCHEMA},
}


def load_data(
    name: str,
    stage: str = "raw",
    **kwargs: Any,
) -> pd.DataFrame:
    """Load raw sources or saved datasets with a declared stage contract.

    Raw reader kwargs retain their pandas behavior. Interim/processed readers
    accept file_format='parquet' (default with CSV fallback) or 'csv' (direct),
    plus schema for custom names. Registered names have a fixed schema that an
    explicit override must equal. Custom names require schema=TableSchema(...).
    Other saved-dataset parser kwargs are rejected to preserve validated types.

    The processed name 'cleaned_order_items' uses the fixed 25-column contract
    for retained items.
    Both saved stages delegate to the shared reader for dtype restoration,
    column/key checks, format announcements and error guidance. No cleaning or
    saving occurs here. Failed direct CSV reads propagate their original error;
    failure of both default format attempts raises RuntimeError with causes.
    """

    if stage == "raw":
        registry = DATASETS
        data_dir = RAW_DIR

    elif stage in ("interim", "processed"):
        registry = INTERIM_DATASETS if stage == "interim" else PROCESSED_DATASETS
        data_dir = INTERIM_DIR if stage == "interim" else PROCESSED_DIR
        file_format = kwargs.pop("file_format", "parquet")
        schema = kwargs.pop("schema", None)
        if kwargs:
            raise TypeError(
                f"Unsupported {stage} reader options: {sorted(kwargs)}. "
                "Use file_format/schema, then select or transform the loaded "
                "DataFrame. Pandas parser options are supported for raw data only."
            )
        if schema is not None and not isinstance(schema, TableSchema):
            raise TypeError("schema must be a TableSchema or None.")
        if name in registry:
            canonical = registry[name]["schema"]
            if schema is not None and schema != canonical:
                raise ValueError(
                    f"Dataset '{name}' has a fixed registered schema. "
                    "Use its canonical contract or save custom data under a new name."
                )
            schema = canonical
        elif schema is None:
            raise ValueError(
                f"Unknown dataset '{name}' in stage '{stage}'. "
                f"Available datasets: {list(registry)}. "
                "Supply schema=TableSchema(...) to read a custom saved dataset."
            )
        return read_dataset(
            data_dir, schema, dataset_name=name, file_format=file_format
        )

    else:
        raise ValueError(
            f"Unknown stage '{stage}'. "
            "Available stages: ['raw', 'interim', 'processed']"
        )

    if name not in registry:
        raise ValueError(
            f"Unknown dataset '{name}' in stage '{stage}'. "
            f"Available datasets: {list(registry.keys())}"
        )

    path: Path = data_dir / registry[name]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: '{path}'."
        )

    # Select the correct pandas reader
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)

    elif path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)

    else:
        raise ValueError(
            f"Unsupported file format: '{path.suffix}'"
        )


def load_all():
    """Load all registered raw datasets."""
    return {
        name: load_data(name)
        for name in DATASETS
    }