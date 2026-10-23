"""Typed raw-table loading for the merge pipeline."""

from typing import Optional, Sequence

import pandas as pd

from src.data_loader import load_data
from pipeline.contracts.normalizer import normalize_table_types
from pipeline.contracts.schema import TableLoadSpec, build_source_schema
from pipeline.contracts.validator import validate_source_keys, validate_table_schema


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

    Reuse the shared schema builder, normalizer and validators.
    Use the registered primary key unless key_columns is supplied explicitly;
    additional registered raw datasets must provide their own key declaration.
    Return a new DataFrame without filtering rows, filling missing attributes,
    checking foreign keys or saving files. Missing timestamps are retained.

    Args:
        spec: Registered raw dataset, selected columns and dtype declarations.
        key_columns: Optional explicit primary key for this load.

    Raises:
        TypeError: If a load declaration has an unexpected type.
        FileNotFoundError: If the registered source file does not exist.
        ValueError: If configuration, columns, values or source keys are invalid.
    """
    schema = build_source_schema(spec, key_columns=key_columns)
    columns = schema.columns

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
    validate_source_keys(normalized, spec.dataset_name, key_columns=schema.unique_key)
    return normalized
