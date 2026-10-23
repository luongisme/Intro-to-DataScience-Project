"""Typed raw-table loading for the merge pipeline."""

from typing import Optional, Sequence

import pandas as pd

from src.data_loader import load_data
from src.pipeline.normalizer import normalize_table_types
from src.pipeline.schema import TableLoadSpec, TableSchema
from src.pipeline.validator import validate_source_keys, validate_table_schema


def load_table(
    spec: TableLoadSpec,
    *,
    key_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a registered raw CSV with canonical dtypes and a valid primary key.

    Use TableLoadSpec for column selection and order. Every selected column
    must have a dtype or appear in date_columns, which use datetime64[us].
    Read cells as strings before normalization to preserve identifiers and
    prevent numeric inference from altering values before validation. CSV
    missing-value parsing follows the raw loader's existing pandas defaults.

    Reuse the shared normalizer, schema validator and source-key validator.
    Use the registered primary key unless key_columns is supplied explicitly;
    additional registered raw datasets must provide their own key declaration.
    Return a new DataFrame without filtering rows, filling missing attributes,
    checking foreign keys or saving files. Missing timestamps are retained.

    Args:
        spec: Registered raw dataset, selected columns and dtype declarations.
        key_columns: Optional explicit primary key for this load.

    Raises:
        TypeError: If spec has an unexpected type.
        FileNotFoundError: If the registered source file does not exist.
        ValueError: If configuration, columns, values or source keys are invalid.
    """
    if not isinstance(spec, TableLoadSpec):
        raise TypeError("load_table requires a TableLoadSpec.")
    if not isinstance(spec.dataset_name, str) or not spec.dataset_name.strip():
        raise ValueError("TableLoadSpec.dataset_name must be a non-empty string.")
    if isinstance(spec.usecols, str) or not isinstance(spec.usecols, Sequence):
        raise ValueError("TableLoadSpec.usecols must be a sequence of column names.")
    columns = list(spec.usecols)
    if not columns or not all(isinstance(col, str) and col.strip() for col in columns):
        raise ValueError("TableLoadSpec.usecols requires non-empty string names.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Load specification for '{spec.dataset_name}' repeats a column.")
    if isinstance(spec.date_columns, str) or not isinstance(spec.date_columns, Sequence):
        raise ValueError("TableLoadSpec.date_columns must be a sequence of names.")
    dates = list(spec.date_columns)
    if not all(isinstance(col, str) and col.strip() for col in dates):
        raise ValueError("TableLoadSpec.date_columns requires non-empty string names.")
    if len(dates) != len(set(dates)):
        raise ValueError(f"Load specification for '{spec.dataset_name}' repeats a date column.")
    if not isinstance(spec.dtypes, dict):
        raise TypeError("TableLoadSpec.dtypes must be a dictionary.")
    overlap = [col for col in dates if col in spec.dtypes]
    if overlap:
        raise ValueError(
            f"Table '{spec.dataset_name}' declares dtypes for date_columns: {overlap}. "
            "Declare timestamps only in date_columns."
        )
    dtypes = {**spec.dtypes, **dict.fromkeys(dates, "datetime64[us]")}
    missing_dtypes = [col for col in columns if col not in dtypes]
    extra_dtypes = [col for col in dtypes if col not in columns]
    if missing_dtypes or extra_dtypes:
        raise ValueError(
            f"Table '{spec.dataset_name}' has inconsistent load dtype declarations: "
            f"missing={missing_dtypes}, extra={extra_dtypes}."
        )
    schema = TableSchema(columns=columns, dtypes=dtypes)

    try:
        loaded = load_data(
            name=spec.dataset_name,
            stage="raw",
            usecols=columns,
            dtype="string",
        )
    except ValueError as err:
        if "Usecols do not match columns" in str(err):
            raise ValueError(
                f"Table '{spec.dataset_name}' missing expected columns from CSV. {err}"
            ) from err
        raise ValueError(f"Cannot load raw table '{spec.dataset_name}': {err}") from err
    missing_columns = [col for col in columns if col not in loaded.columns]
    if missing_columns:
        raise ValueError(
            f"Table '{spec.dataset_name}' missing expected columns: {missing_columns}."
        )
    if loaded.columns.has_duplicates:
        raise ValueError(f"Table '{spec.dataset_name}' has duplicate column names.")
    # pandas usecols selects columns in file order, not declaration order.
    ordered = loaded.loc[:, columns]
    normalized = normalize_table_types(ordered, schema, table_name=spec.dataset_name)
    validate_table_schema(normalized, schema, table_name=spec.dataset_name)
    validate_source_keys(normalized, spec.dataset_name, key_columns=key_columns)
    return normalized
