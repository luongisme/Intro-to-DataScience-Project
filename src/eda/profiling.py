"""Read-only structural profiling for the full merged dataset."""

from collections.abc import Sequence

import pandas as pd

from src.pipeline.contracts.schema import (
    ALL_TIMESTAMP_COLUMNS,
    CUSTOMERS_SPEC,
    FULL_MERGE_SCHEMA,
    MANDATORY_NON_NULL_COLUMNS,
    ORDER_ITEMS_SPEC,
    ORDERS_SPEC,
    PRODUCTS_SPEC,
    SELLERS_SPEC,
    SOURCE_PRIMARY_KEYS,
    ZIP_COLUMNS,
)

# Source ownership follows the merge order; shared join keys belong to the
# first source contributing the column. This is metadata, not a new schema.
SOURCE_SPECS = (
    ORDER_ITEMS_SPEC, ORDERS_SPEC, CUSTOMERS_SPEC, PRODUCTS_SPEC, SELLERS_SPEC,
)


def require_columns(df: pd.DataFrame, columns: Sequence[str] = ()) -> None:
    """Require a DataFrame with unique names and the requested columns.

    Accept subsets or additional columns without enforcing merge dtypes.
    Raise TypeError for invalid argument types, ValueError for repeated names,
    or KeyError for missing columns. Never convert or modify the input.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.columns.has_duplicates:
        raise ValueError("Structural profiling requires unique column names.")
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise TypeError("columns must be a sequence of column names.")
    if not all(isinstance(column, str) for column in columns):
        raise TypeError("Requested column names must be strings.")
    if len(columns) != len(set(columns)):
        raise ValueError("Requested column names must not repeat.")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing profiling columns: {missing}.")


def blank_string_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask for empty/whitespace-only strings.

    Actual nulls, numbers and other non-string values are not blanks. Preserve
    the index, including duplicate labels, without casting the source values.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    if not (
        pd.api.types.is_string_dtype(series.dtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return pd.Series(False, index=series.index, name=series.name)
    return series.astype(object).map(
        lambda value: isinstance(value, str) and not value.strip()
    ).astype(bool)


def get_column_roles() -> pd.DataFrame:
    """Return metadata for the 25 registered full-merge columns in order.

    Columns describe role, contributing source, source-level grain and expected
    dtype. Numeric identifiers and ZIP codes are not measurements. Source grain
    describes the contributing table; it does not change the full item grain.
    Each call returns an independent table. No input or file is read.
    """
    ownership = {}
    for spec in SOURCE_SPECS:
        for column in spec.usecols:
            ownership.setdefault(column, spec.dataset_name)
    identifiers = set(MANDATORY_NON_NULL_COLUMNS) | set(FULL_MERGE_SCHEMA.unique_key)
    rows = []
    for column in FULL_MERGE_SCHEMA.columns:
        if column in identifiers:
            role = "identifier"
        elif column in ZIP_COLUMNS:
            role = "zip_code"
        elif column in ALL_TIMESTAMP_COLUMNS:
            role = "timestamp"
        elif pd.api.types.is_numeric_dtype(FULL_MERGE_SCHEMA.dtypes[column]):
            role = "measurement"
        else:
            role = "category"
        source = ownership[column]
        rows.append([column, role, source, tuple(SOURCE_PRIMARY_KEYS[source]), FULL_MERGE_SCHEMA.dtypes[column]])
    return pd.DataFrame(rows, columns=[
        "column", "role", "source_table", "source_key", "expected_dtype",
    ])


def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return item-table dimensions, completeness, duplicate and entity counts.

    Null-cell/row counts use the supplied table as-is. Exact duplicate rows
    include all columns; report both participating rows and excess copies.
    Entity counts exclude null/blank IDs and count observed IDs only, not raw
    source coverage. customer_id is not a unique-person identifier.
    """
    require_columns(df)
    rows = [
        ["Rows", len(df), "item rows"],
        ["Columns", len(df.columns), "columns"],
        ["Memory usage", int(df.memory_usage(index=True, deep=True).sum()), "bytes"],
        ["Missing cells", int(df.isna().sum().sum()), "cells"],
        ["Rows with missing values", int(df.isna().any(axis=1).sum()), "item rows"],
        ["Exact duplicate participants", int(df.duplicated(keep=False).sum()), "item rows"],
        ["Exact duplicate excess copies", int(df.duplicated().sum()), "item rows"],
    ]
    for spec in SOURCE_SPECS[1:]:
        key = SOURCE_PRIMARY_KEYS[spec.dataset_name][0]
        if key in df.columns:
            valid = df[key].notna() & ~blank_string_mask(df[key])
            rows.append([f"Observed {spec.dataset_name}", int(df.loc[valid, key].nunique()), "distinct IDs"])
    return pd.DataFrame(rows, columns=["metric", "value", "unit"])


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return one completeness row per input column, without transformations.

    All counts use input rows, including repeated entity attributes. Missing
    means isna(); blank strings are counted separately. Distinct non-null
    values include blank strings and retain case/whitespace differences.
    Percentages use all rows; an empty input has undefined percentages (NaN).
    Unregistered column names receive the role 'unregistered'.
    """
    require_columns(df)
    roles = get_column_roles().set_index("column")["role"]
    count = len(df)
    rows = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        blank = int(blank_string_mask(series).sum())
        rows.append([
            column, roles.get(column, "unregistered"), str(series.dtype), count,
            count - missing, missing, 100 * missing / count if count else float("nan"),
            blank, 100 * blank / count if count else float("nan"),
            int(series.nunique(dropna=True)),
        ])
    return pd.DataFrame(rows, columns=[
        "column", "role", "dtype", "rows", "non_null", "missing",
        "missing_pct", "blank", "blank_pct", "distinct_non_null",
    ])

