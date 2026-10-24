"""Read-only structural profiling for the full merged dataset."""

from collections.abc import Sequence

import numpy as np
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


def finite_numeric_values(series: pd.Series) -> pd.Series:
    """Return real, finite measurement values without modifying the input.

    Reject non-numeric, boolean or complex dtypes and registered columns whose
    role is not measurement (including numeric IDs). Unregistered real numeric
    columns are accepted explicitly. Nulls and infinities are excluded only
    from this temporary calculation view; profiles report them separately.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    if (
        not pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_complex_dtype(series.dtype)
    ):
        raise TypeError("Numeric profiling requires a real numeric measurement dtype.")
    roles = get_column_roles().set_index("column")["role"]
    if series.name in roles.index and roles.loc[series.name] != "measurement":
        raise ValueError(f"Column '{series.name}' is not a registered measurement.")
    return series.loc[series.notna() & np.isfinite(series).fillna(False)].copy()


def profile_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Summarize measurements using finite values at the supplied table grain.

    Return one row per selected column: completeness, finite/non-finite counts,
    zero/negative counts, sample standard deviation (ddof=1), quantiles and
    Tukey 1.5-IQR diagnostic counts. Quantiles use linear interpolation.
    Missing percentages use all rows; IQR percentages use finite values.
    Empty/all-missing data has undefined statistics and percentages, not zero
    substitutes. IQR flags are descriptive, never cleaning/removal rules.
    """
    require_columns(df, columns)
    if not columns:
        raise ValueError("Select at least one measurement column.")
    records = []
    for column in columns:
        series = df[column]
        values = finite_numeric_values(series)
        count = len(values)
        quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        q = {name: float(value) if pd.notna(value) else float("nan")
             for name, value in quantiles.items()}
        iqr = q[0.75] - q[0.25]
        lower, upper = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
        low_count, high_count = int(values.lt(lower).sum()), int(values.gt(upper).sum())
        records.append({
            "column": column, "rows": len(series),
            "missing": int(series.isna().sum()),
            "missing_pct": 100 * int(series.isna().sum()) / len(series) if len(series) else float("nan"),
            "finite": count, "non_finite_non_null": int(series.notna().sum()) - count,
            "zero": int(values.eq(0).sum()), "negative": int(values.lt(0).sum()),
            "mean": float(values.mean()) if count else float("nan"),
            "std": float(values.std(ddof=1)) if count > 1 else float("nan"),
            "min": values.min() if count else float("nan"),
            "p01": q[0.01], "p05": q[0.05], "p25": q[0.25], "median": q[0.5],
            "p75": q[0.75], "p95": q[0.95], "p99": q[0.99],
            "max": values.max() if count else float("nan"),
            "iqr": iqr, "iqr_lower": lower, "iqr_upper": upper,
            "below_iqr": low_count, "above_iqr": high_count,
            "outside_iqr_pct": 100 * (low_count + high_count) / count if count else float("nan"),
        })
    return pd.DataFrame(records)


def _frequency_table(series: pd.Series, counts: pd.Series) -> pd.DataFrame:
    """Attach explicit missing counts and the all-row percentage denominator."""
    records = [
        [value, "blank" if isinstance(value, str) and not value.strip() else "value", int(count)]
        for value, count in counts.items()
    ]
    records.append([pd.NA, "missing", int(series.isna().sum())])
    result = pd.DataFrame(records, columns=["value", "kind", "count"])
    result["total_rows"] = len(series)
    result["pct_all"] = 100 * result["count"] / len(series) if len(series) else float("nan")
    return result


def profile_frequencies(series: pd.Series) -> pd.DataFrame:
    """Return full observed-value frequencies, sorted by descending count.

    Preserve literal values, case and whitespace; identify blanks separately.
    Include one explicit missing row even when its count is zero. Percentages
    use all input rows, including missing; unused categorical levels are not
    observations. A zero-row input has undefined percentages.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    counts = series.value_counts(dropna=True, sort=False)
    counts = counts.loc[counts.gt(0)].sort_values(ascending=False, kind="stable")
    return _frequency_table(series, counts)


def limit_frequencies(table: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Return a display table with top values, Other, Blank and Missing.

    Accept an uncollapsed frequency table from this module. Rank normal values
    by count; preserve their full row denominator. Aggregate omitted normal
    values into an Other row and blank variants into one Blank row. Special
    kinds distinguish these rows from literal strings with the same names.
    Never change the full table. Counts still sum to all input rows.
    """
    require_columns(table, ["value", "kind", "count", "total_rows", "pct_all"])
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    if table.empty or not table["kind"].isin(["value", "blank", "missing"]).all():
        raise ValueError("Use an uncollapsed frequency table from this module.")
    totals = table["total_rows"].unique()
    if (
        len(totals) != 1 or not pd.api.types.is_integer_dtype(table["count"].dtype)
        or table["count"].isna().any() or table["count"].lt(0).any()
        or table["count"].sum() != totals[0]
    ):
        raise ValueError("Frequency counts must sum to one consistent total_rows value.")
    total = int(totals[0])
    normal = table.loc[table["kind"].eq("value")].sort_values("count", ascending=False, kind="stable")
    result = normal.head(top_n).copy()
    extra = []
    remaining = int(normal.iloc[top_n:]["count"].sum())
    if remaining:
        extra.append([pd.NA, "other", remaining])
    blanks = int(table.loc[table["kind"].eq("blank"), "count"].sum())
    if blanks:
        extra.append([pd.NA, "blank", blanks])
    extra.append([pd.NA, "missing", int(table.loc[table["kind"].eq("missing"), "count"].sum())])
    special = pd.DataFrame(extra, columns=["value", "kind", "count"])
    special["total_rows"] = total
    special["pct_all"] = 100 * special["count"] / total if total else float("nan")
    return pd.concat([result, special], ignore_index=True)


def profile_timestamp_coverage(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return timestamp ranges, missingness and clock precision per column.

    Require datetime dtypes; never parse strings or infer/convert timezones.
    Clock components use stored values. Non-midnight/subsecond counts use
    non-missing timestamps; missing percentages use all supplied rows.
    """
    require_columns(df, columns)
    if not columns:
        raise ValueError("Select at least one timestamp column.")
    records = []
    for column in columns:
        series = df[column]
        if not pd.api.types.is_datetime64_any_dtype(series.dtype):
            raise TypeError(f"Column '{column}' must have a datetime dtype.")
        valid = series.dropna()
        non_midnight = (
            valid.dt.hour.ne(0) | valid.dt.minute.ne(0) | valid.dt.second.ne(0)
            | valid.dt.microsecond.ne(0) | valid.dt.nanosecond.ne(0)
        )
        records.append([
            column, len(series), len(valid), int(series.isna().sum()),
            100 * int(series.isna().sum()) / len(series) if len(series) else float("nan"),
            valid.min(), valid.max(), int(valid.nunique()), int(non_midnight.sum()),
            int((valid.dt.microsecond.ne(0) | valid.dt.nanosecond.ne(0)).sum()),
        ])
    return pd.DataFrame(records, columns=[
        "column", "rows", "non_null", "missing", "missing_pct", "min", "max",
        "distinct_non_null", "non_midnight", "subsecond",
    ])


def profile_temporal_frequencies(series: pd.Series, frequency: str = "month") -> pd.DataFrame:
    """Return calendar month/date, weekday or hour frequencies in clock order.

    Accept datetime Series and month/date/weekday/hour only. Retain the stored
    clock/timezone without inferring a local timezone. Weekday/hour tables
    include all seven/24 buckets (zero where unobserved); month/date tables
    contain observed calendar buckets only. Preserve missing timestamps in a
    separate row and use all input rows as denominator. No time-slot business
    categories, delivery intervals or persistent feature columns are created.
    """
    if not isinstance(series, pd.Series) or not pd.api.types.is_datetime64_any_dtype(series.dtype):
        raise TypeError("Temporal frequencies require a datetime Series.")
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if frequency == "month":
        buckets = series.dt.strftime("%Y-%m")
        counts = buckets.value_counts(dropna=True, sort=False).sort_index()
    elif frequency == "date":
        buckets = series.dt.strftime("%Y-%m-%d")
        counts = buckets.value_counts(dropna=True, sort=False).sort_index()
    elif frequency == "weekday":
        buckets = series.dt.dayofweek
        counts = buckets.value_counts(dropna=True, sort=False).reindex(range(7), fill_value=0)
        counts.index = weekdays
    elif frequency == "hour":
        buckets = series.dt.hour
        counts = buckets.value_counts(dropna=True, sort=False).reindex(range(24), fill_value=0)
    else:
        raise ValueError("frequency must be 'month', 'date', 'weekday' or 'hour'.")
    return _frequency_table(series, counts)
