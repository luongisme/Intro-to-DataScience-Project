"""Load raw source files or schema-aware registered/custom interim datasets."""

from typing import Any, Dict
from pathlib import Path

import pandas as pd

from src.paths import RAW_DIR, INTERIM_DIR
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


# Canonical interim plans and schemas; actual format selection belongs to I/O.
INTERIM_DATASETS: Dict[str, Dict[str, Any]] = build_dataset_registry()


def load_data(
    name: str,
    stage: str = "raw",
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a raw source or a schema-aware saved merge dataset.

    Preserve the raw-stage pandas reader behavior and its **kwargs forwarding.
    For interim data, reuse read_dataset for type restoration, validation and
    format selection instead of dispatching directly to pandas. The existing
    name/stage call signature and full name 'merged_order_items' are retained.

    Interim names in INTERIM_DATASETS automatically use their registered
    schema. An explicit schema must equal that fixed contract; do not weaken
    a registered dataset's validation. A custom unregistered name requires
    schema=TableSchema(...). The shared reader validates custom names/schema.

    Interim kwargs accept file_format='parquet' (default, with CSV fallback)
    or 'csv' (direct CSV), and schema for an explicit contract. Reject other
    parser kwargs rather than silently changing the validated output schema.
    Select/transform columns after loading. Do not interpret file_format or
    schema specially for raw data; raw kwargs still reach pandas unchanged.

    Args:
        name: Registered raw/interim dataset or a custom interim filename stem.
        stage: 'raw' or 'interim', including the existing positional argument.
        **kwargs: Pandas reader options for raw data; file_format/schema only
            for interim data.

    Returns:
        A DataFrame; interim columns, dtypes and constraints follow the schema.

    Raises:
        TypeError: If interim schema has the wrong type or unsupported parser
            options are supplied; raw pandas argument errors propagate.
        ValueError: If stage/name/configuration or the interim contract is invalid.
        FileNotFoundError: If a raw file or explicitly selected CSV is missing.
        RuntimeError: If both interim Parquet and CSV read attempts fail.
        Exception: Other direct CSV/raw read failures propagate unchanged.
    """

    if stage == "raw":
        registry = DATASETS
        data_dir = RAW_DIR

    elif stage == "interim":
        file_format = kwargs.pop("file_format", "parquet")
        schema = kwargs.pop("schema", None)
        if kwargs:
            raise TypeError(
                f"Unsupported interim reader options: {sorted(kwargs)}. "
                "Use file_format/schema, then select or transform the loaded "
                "DataFrame. Pandas parser options are supported for raw data only."
            )
        if schema is not None and not isinstance(schema, TableSchema):
            raise TypeError("schema must be a TableSchema or None.")
        if name in INTERIM_DATASETS:
            canonical = INTERIM_DATASETS[name]["schema"]
            if schema is not None and schema != canonical:
                raise ValueError(
                    f"Dataset '{name}' has a fixed registered schema. "
                    "Use its canonical contract or save custom data under a new name."
                )
            schema = canonical
        elif schema is None:
            raise ValueError(
                f"Unknown dataset '{name}' in stage '{stage}'. "
                f"Available datasets: {list(INTERIM_DATASETS)}. "
                "Supply schema=TableSchema(...) to read a custom interim dataset."
            )
        return read_dataset(
            INTERIM_DIR, schema, dataset_name=name, file_format=file_format
        )

    else:
        raise ValueError(
            f"Unknown stage '{stage}'. "
            "Available stages: ['raw', 'interim']"
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