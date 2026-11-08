"""Load raw sources or schema-aware interim and processed datasets."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.paths import RAW_DIR, INTERIM_DIR, PROCESSED_DIR
from src.cleaning.schema import (
    CLEANED_DATASET_NAME,
    CLEANED_ORDER_ITEMS_SCHEMA,
    LATE_DELIVERY_ORDERS_NAME,
    LATE_DELIVERY_ORDERS_SCHEMA,
    LEGACY_CLEANED_ORDER_ITEMS_EXTRA_COLUMNS,
    LEGACY_CLEANED_ORDER_ITEMS_NULLABLE_COLUMNS,
    LEGACY_CLEANED_ORDER_ITEMS_SCHEMA,
    ANOMALY_ORDERS_NAME,
    ANOMALY_ORDERS_SCHEMA,
    ORDER_ANALYSIS_NAME,
    ORDER_ANALYSIS_SCHEMA,
)
from src.pipeline.contracts.normalizer import normalize_table_types
from src.pipeline.contracts.schema import TableSchema
from src.pipeline.contracts.validator import validate_table_contract
from src.pipeline.io.dataset_io import read_dataset
from src.pipeline.merging.registry import build_dataset_registry



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

INTERIM_DATASETS: Dict[str, Dict[str, Any]] = build_dataset_registry()
PROCESSED_DATASETS = {
    CLEANED_DATASET_NAME: {"schema": CLEANED_ORDER_ITEMS_SCHEMA},
    LATE_DELIVERY_ORDERS_NAME: {"schema": LATE_DELIVERY_ORDERS_SCHEMA},
    ANOMALY_ORDERS_NAME: {"schema": ANOMALY_ORDERS_SCHEMA},
    ORDER_ANALYSIS_NAME: {"schema": ORDER_ANALYSIS_SCHEMA},
}


def _read_saved_without_schema(
    data_dir: Path,
    name: str,
    file_format: str,
) -> pd.DataFrame:
    """Read a saved file with pandas, without contract or dtype restoration."""
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or Path(name).name != name
        or "/" in name
        or "\\" in name
        or Path(name).suffix in (".csv", ".parquet")
    ):
        raise ValueError(
            "Dataset name must be a filename stem without a directory or "
            "'.csv'/'.parquet' extension."
        )
    if file_format not in ("parquet", "csv"):
        raise ValueError("file_format must be 'parquet' or 'csv'.")

    formats = ("parquet", "csv") if file_format == "parquet" else ("csv",)
    errors = []
    for fmt in formats:
        path = data_dir / f"{name}.{fmt}"
        try:
            if fmt == "parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except Exception as exc:
            errors.append((path, exc))

    details = "; ".join(
        f"{path}: {type(exc).__name__}: {exc}" for path, exc in errors
    )
    raise RuntimeError(
        f"Could not load dataset '{name}' without schema validation. "
        f"Attempts: {details}"
    ) from errors[-1][1]


def _read_legacy_cleaned_parquet(data_dir: Path) -> Optional[pd.DataFrame]:
    """Read the known 27-column legacy file and return the canonical 25 columns.

    This path is used only when the Parquet contains exactly the canonical
    columns plus the two obsolete QA flags. The selected data is then normalized
    and fully validated against the current schema.
    """
    path = data_dir / f"{CLEANED_DATASET_NAME}.parquet"
    if not path.is_file():
        return None

    try:
        import pyarrow.parquet as pq

        actual_columns = pq.read_schema(path).names
    except Exception:
        # Let the regular reader report missing engines, unreadable files, etc.
        return None

    expected_columns = list(CLEANED_ORDER_ITEMS_SCHEMA.columns)
    allowed_extra = set(LEGACY_CLEANED_ORDER_ITEMS_EXTRA_COLUMNS)
    expected_set = set(expected_columns)
    if (
        len(actual_columns) != len(expected_columns) + len(allowed_extra)
        or len(set(actual_columns)) != len(actual_columns)
        or set(actual_columns) != expected_set | allowed_extra
    ):
        return None

    result = pd.read_parquet(path, columns=expected_columns)
    result = result.loc[:, expected_columns]
    result = normalize_table_types(
        result,
        CLEANED_ORDER_ITEMS_SCHEMA,
        table_name=CLEANED_DATASET_NAME,
    )
    validate_table_contract(
        result,
        LEGACY_CLEANED_ORDER_ITEMS_SCHEMA,
        table_name=CLEANED_DATASET_NAME,
    )
    print(
        f"Loaded PARQUET from '{path}' after removing the two legacy QA columns."
    )
    return result


def load_data(
    name: str,
    stage: str = "raw",
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a raw source or a registered schema-aware saved dataset.

    Raw reader kwargs are passed to pandas. Interim/processed readers accept
    file_format='parquet' (Parquet first, CSV fallback) or 'csv', an optional
    schema for custom names, and validate=False for exploratory direct reads.

    The registered cleaned_order_items dataset is strictly validated against
    its current 25-column schema. For the known old Parquet with exactly two
    extra QA columns, those two columns are omitted and the remaining 25 are
    normalized and validated before returning.
    """
    if stage == "raw":
        registry = DATASETS
        data_dir = RAW_DIR
    elif stage in ("interim", "processed"):
        registry = INTERIM_DATASETS if stage == "interim" else PROCESSED_DATASETS
        data_dir = INTERIM_DIR if stage == "interim" else PROCESSED_DIR
        file_format = kwargs.pop("file_format", "parquet")
        schema = kwargs.pop("schema", None)
        validate = kwargs.pop("validate", True)

        if kwargs:
            raise TypeError(
                f"Unsupported {stage} reader options: {sorted(kwargs)}. "
                "Use file_format/schema/validate; pandas parser options are "
                "supported for raw data only."
            )
        if not isinstance(validate, bool):
            raise TypeError("validate must be a boolean.")
        if schema is not None and not isinstance(schema, TableSchema):
            raise TypeError("schema must be a TableSchema or None.")

        if not validate:
            if schema is not None:
                raise ValueError("schema cannot be supplied with validate=False.")
            return _read_saved_without_schema(data_dir, name, file_format)

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
                "Supply schema=TableSchema(...) or use validate=False."
            )

        if name == CLEANED_DATASET_NAME and file_format == "parquet":
            legacy = _read_legacy_cleaned_parquet(data_dir)
            if legacy is not None:
                return legacy

        return read_dataset(
            data_dir,
            schema,
            dataset_name=name,
            file_format=file_format,
        )
    else:
        raise ValueError(
            "Unknown stage '" + str(stage) + "'. "
            "Available stages: ['raw', 'interim', 'processed']"
        )

    if name not in registry:
        raise ValueError(
            f"Unknown dataset '{name}' in stage '{stage}'. "
            f"Available datasets: {list(registry)}"
        )

    path = data_dir / registry[name]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: '{path}'.")
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported file format: '{path.suffix}'")


def load_all() -> Dict[str, pd.DataFrame]:
    """Load all registered raw datasets."""
    return {name: load_data(name) for name in DATASETS}