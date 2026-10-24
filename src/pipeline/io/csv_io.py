"""Schema-aware CSV I/O using the shared reversible text-cell codec."""

import csv
import sys
from datetime import timezone
from pathlib import Path
from threading import RLock
from typing import Union

import pandas as pd

from src.pipeline.io.csv_codec import decode_csv_strings, encode_csv_strings
from src.pipeline.contracts.schema import TableSchema
from src.pipeline.io.csv_types import restore_csv_column
from src.pipeline.contracts.validator import validate_table_contract


_CSV_FIELD_SIZE_LOCK = RLock()


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


def read_csv_dataset(path: Union[str, Path], schema: TableSchema) -> pd.DataFrame:
    """Read this pipeline's encoded UTF-8 CSV using an explicit table contract.

    Validate the declared contract and supported dtypes before opening the
    file. Parse text with Python's strict CSV reader without type/NA inference,
    header renaming, BOM stripping, whitespace trimming or skipped records.
    Require exact header names/order and the exact field count in every data
    record. Preserve quotes, embedded CR/LF and NUL characters. Header-only
    files return typed empty frames; a file without a header is invalid.

    Reuse decode_csv_strings once per column, then restore_csv_column for
    lossless dtype restoration. Validate keys/non-null constraints after all
    columns are restored. Return a new DataFrame in file row order with a
    fresh RangeIndex; no original DataFrame index was written to the file.

    Temporarily raise the process-wide CSV field limit to support long cells,
    serializing this reader's changes with a lock and restoring the previous
    limit even when reading fails. Never write files, load schema sidecars,
    infer business rules, repair rows or try another format/path. Compression
    is not inferred from the filename. Use files produced by write_csv_dataset;
    this is not a reader for arbitrary raw CSV missing-value conventions.

    Raises:
        TypeError: If path or schema has an invalid argument type.
        ValueError: If the contract, encoding, CSV structure, cells or output
            constraints are invalid. Parse errors include file/record context.
        OSError: If the file does not exist or cannot be opened/read.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("The CSV source path must not be empty.")
    source = Path(path)
    if not isinstance(schema, TableSchema):
        raise TypeError("schema must be a TableSchema.")
    try:
        empty = pd.DataFrame({
            name: restore_csv_column(pd.Series([], dtype="string", name=name), schema.dtypes[name])
            for name in schema.columns
        })
    except (TypeError, ValueError, KeyError) as err:
        raise ValueError(f"Invalid CSV schema for '{source}': {err}") from err
    validate_table_contract(empty, schema, table_name="csv_input")
    if not schema.columns:
        raise ValueError("CSV input requires at least one declared column.")
    expected = list(schema.columns)
    columns = {name: [] for name in expected}

    # The field-size setting is shared process-wide; guard this reader's changes.
    with _CSV_FIELD_SIZE_LOCK:
        previous_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(sys.maxsize)
            with source.open("r", encoding="utf-8", errors="strict", newline="") as handle:
                reader = csv.reader(handle, strict=True)
                try:
                    header = next(reader, None)
                    if header is None:
                        raise ValueError(f"CSV '{source}' is empty: a header is required.")
                    if header != expected:
                        raise ValueError(
                            f"CSV '{source}' header differs from its schema. "
                            f"Expected: {expected}. Actual: {header}."
                        )
                    for record, row in enumerate(reader, start=2):
                        if len(row) != len(expected):
                            raise ValueError(
                                f"CSV '{source}' record {record} (physical line "
                                f"{reader.line_num}) has {len(row)} fields; "
                                f"expected {len(expected)}."
                            )
                        for name, cell in zip(expected, row):
                            columns[name].append(cell)
                except csv.Error as err:
                    raise ValueError(
                        f"Cannot parse CSV '{source}' at physical line "
                        f"{reader.line_num}: {err}"
                    ) from err
        except UnicodeError as err:
            raise ValueError(f"Cannot decode UTF-8 CSV '{source}': {err}") from err
        finally:
            csv.field_size_limit(previous_limit)

    for name in expected:
        try:
            text = pd.Series(columns[name], dtype="string", name=name)
            decoded = decode_csv_strings(text)
            columns[name] = restore_csv_column(decoded, schema.dtypes[name])
        except (TypeError, ValueError) as err:
            raise ValueError(f"Cannot read CSV '{source}', column {name!r}: {err}") from err
    result = pd.DataFrame(columns, columns=expected)
    validate_table_contract(result, schema, table_name=f"csv_input '{source}'")
    return result
