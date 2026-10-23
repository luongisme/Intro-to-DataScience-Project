"""Schema-aware CSV I/O using the shared reversible text-cell codec."""

from datetime import timezone
from pathlib import Path
from typing import Union

import pandas as pd

from src.pipeline.csv_codec import encode_csv_strings
from src.pipeline.schema import TableSchema
from src.pipeline.validator import validate_table_contract


def write_csv_dataset(
    df: pd.DataFrame,
    path: Union[str, Path],
    schema: TableSchema,
) -> Path:
    """Write a validated table to UTF-8 CSV and return the supplied path.

    Validate exact columns/order/dtypes, declared keys and non-null constraints
    before opening the file. Serialize all columns to text, then use the shared
    codec for missing cells and literal escape prefixes. Float values of up to
    64 bits are widened losslessly before text conversion; no decimal rounding
    is requested. Keep integer digits, boolean values and datetime precision,
    including timezone offsets. Reject unsupported dtypes before any write.

    Preserve row/column order and never mutate the input. Write a header and
    exclude the DataFrame index. Use standard CSV quoting with CRLF record
    terminators so both CR and LF inside a cell are quoted. Disable implicit
    compression regardless of the filename suffix. Do not append an extension,
    create parent directories, save schema/metadata or select another format.

    The parent directory must already exist. This low-level writer overwrites
    its exact target directly; filesystem/encoding errors propagate. Atomic
    replacement belongs to the higher-level save workflow, which must call
    this function with a temporary path. Restore values through the paired
    schema-aware reader, not pandas' default type/missing inference.

    Args:
        df: Table already normalized to its declared schema.
        path: Exact destination filename, including any desired extension.
        schema: Source column types and integrity constraints.

    Raises:
        TypeError: If arguments have invalid types.
        ValueError: If the contract, path or serialization dtype is invalid.
        OSError: If the directory is missing or the target cannot be written.
        UnicodeError: If values cannot be encoded as strict UTF-8.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("The CSV destination path must not be empty.")
    destination = Path(path)
    validate_table_contract(df, schema, table_name="csv_output")
    if not schema.columns:
        raise ValueError("CSV output requires at least one declared column.")

    serialized = df.copy(deep=True)
    for column in schema.columns:
        values = df[column]
        dtype = values.dtype
        if pd.api.types.is_float_dtype(dtype):
            if dtype.itemsize > 8:
                raise ValueError(
                    f"CSV column '{column}' has unsupported floating precision: {dtype}."
                )
            values = values.astype("Float64")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            # Format wall time and UTC offset separately to retain all nanoseconds.
            timestamps = []
            for value in values:
                if pd.isna(value):
                    timestamps.append(pd.NA)
                elif value.tzinfo is None:
                    timestamps.append(str(value))
                else:
                    offset = str(timezone(value.utcoffset())).removeprefix("UTC")
                    timestamps.append(f"{value.tz_localize(None)}{offset or '+00:00'}")
            values = pd.Series(
                timestamps, dtype="string", index=values.index, name=values.name
            )
        elif not (
            isinstance(dtype, pd.StringDtype)
            or pd.api.types.is_integer_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            raise ValueError(f"CSV column '{column}' has unsupported dtype: {dtype}.")
        if not isinstance(values.dtype, pd.StringDtype):
            values = values.astype("string")
        serialized[column] = encode_csv_strings(values)

    serialized.to_csv(
        destination,
        index=False,
        encoding="utf-8",
        errors="strict",
        lineterminator="\r\n",
        compression=None,
    )
    return destination
