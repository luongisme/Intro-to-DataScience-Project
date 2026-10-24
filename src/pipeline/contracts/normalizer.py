"""Schema-driven dtype normalization for merge inputs and outputs."""

import pandas as pd

from src.pipeline.contracts.schema import TableSchema


def normalize_table_types(
    df: pd.DataFrame,
    schema: TableSchema,
    *,
    table_name: str = "dataset",
) -> pd.DataFrame:
    """Return a copy with schema columns converted without silent value loss.

    Preserve row order, index, column order, extra columns and existing missing
    values. Do not fill missing values or enforce key and nullability rules.
    Supported targets are pandas string, integer, floating, boolean and datetime
    dtypes. Text columns must already contain strings; numeric timestamps need
    an explicit unit and are therefore rejected here.

    Args:
        df: Input table, which is never modified.
        schema: Expected columns and canonical pandas dtypes.
        table_name: Table label included in error messages.

    Returns:
        A new DataFrame with normalized dtypes for the declared columns.

    Raises:
        TypeError: If df is not a DataFrame.
        ValueError: If columns or dtypes cannot be converted safely.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Table '{table_name}' must be a pandas DataFrame.")
    if df.columns.has_duplicates:
        raise ValueError(f"Table '{table_name}' has duplicate column names.")

    missing_columns = [col for col in schema.columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Table '{table_name}' is missing columns required for normalization: "
            f"{missing_columns}."
        )
    missing_dtypes = [col for col in schema.columns if col not in schema.dtypes]
    extra_dtypes = [col for col in schema.dtypes if col not in schema.columns]
    if missing_dtypes or extra_dtypes:
        raise ValueError(
            f"Schema for '{table_name}' has inconsistent dtype declarations: "
            f"missing={missing_dtypes}, extra={extra_dtypes}."
        )

    normalized = df.copy(deep=True)
    for column, dtype in schema.dtypes.items():
        source = df[column]
        present = source.notna()
        try:
            target = pd.api.types.pandas_dtype(dtype)
            if pd.api.types.is_datetime64_any_dtype(target):
                if any(pd.api.types.is_number(value) for value in source[present]):
                    raise ValueError("Numeric timestamps require an explicit unit.")
                parsed = pd.to_datetime(source, format="mixed", errors="raise")
                converted = parsed.astype(target)
                if not converted[present].eq(parsed[present]).all():
                    raise ValueError("Conversion would lose timestamp precision.")
            elif isinstance(target, pd.StringDtype):
                if not all(isinstance(value, str) for value in source[present]):
                    raise ValueError(
                        "Text columns must contain strings; lost leading zeros "
                        "cannot be inferred."
                    )
                converted = source.astype(target)
            elif pd.api.types.is_bool_dtype(target):
                if not all(
                    pd.api.types.is_bool_dtype(type(value))
                    for value in source[present]
                ):
                    raise ValueError("Boolean columns must contain boolean values.")
                converted = source.astype(target)
            elif (
                pd.api.types.is_integer_dtype(target)
                or pd.api.types.is_float_dtype(target)
            ):
                numeric = pd.to_numeric(source, errors="raise")
                converted = (
                    source.astype(target)
                    if pd.api.types.is_integer_dtype(target)
                    else numeric.astype(target)
                )
                if not converted[present].astype(object).eq(
                    numeric[present].astype(object)
                ).all():
                    raise ValueError("Conversion would alter numeric values.")
            else:
                raise ValueError(f"Unsupported normalization dtype: {dtype!r}.")

            if not converted.isna().equals(source.isna()):
                raise ValueError("Conversion would change which values are missing.")
            normalized[column] = converted
        except (TypeError, ValueError, OverflowError) as err:
            raise ValueError(
                f"Cannot normalize table '{table_name}', column '{column}' "
                f"to {dtype!r}: {err}"
            ) from err

    return normalized
