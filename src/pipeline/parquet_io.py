"""Schema-aware Parquet I/O backed by the project's PyArrow engine."""

from pathlib import Path
from typing import Union

import pandas as pd

from src.pipeline.normalizer import normalize_table_types
from src.pipeline.schema import TableSchema
from src.pipeline.validator import validate_table_contract


def write_parquet_dataset(
    df: pd.DataFrame,
    path: Union[str, Path],
    schema: TableSchema,
) -> Path:
    """Write a validated table to one Parquet file and return its exact path.

    Check exact columns/order/dtypes, declared keys and non-null constraints
    before opening the target. Convert through PyArrow with safe conversions
    before writing, preserving row/column order without the DataFrame index.
    Never mutate the input or serialize its arbitrary DataFrame attributes.

    Use Parquet 2.6 with native nanosecond timestamps and reject timestamp
    truncation. Keep Arrow's embedded schema and pandas type information for
    timezone/type restoration; no schema sidecar or run metadata is created.
    Reading with the supplied table contract belongs to the paired reader.

    The parent directory must already exist. Overwrite the exact target
    directly without appending an extension, creating directories or trying
    CSV. Atomic replacement belongs to the higher-level save workflow, which
    must supply a temporary path. Conversion and filesystem errors propagate.
    Import PyArrow only when called so CSV remains usable without that engine.

    Args:
        df: Table already normalized to its declared schema.
        path: Exact destination filename, including any desired extension.
        schema: Source column types and integrity constraints.

    Raises:
        TypeError: If arguments or Arrow conversion types are invalid.
        ValueError: If the contract, path or converted values are invalid.
        ImportError: If PyArrow is unavailable, with installation guidance.
        OSError: If the directory is missing or the target cannot be written.
        pyarrow.ArrowException: If the engine cannot represent/write the data.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("The Parquet destination path must not be empty.")
    destination = Path(path)
    validate_table_contract(df, schema, table_name="parquet_output")
    if not schema.columns:
        raise ValueError("Parquet output requires at least one declared column.")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as err:
        raise ImportError(
            "Parquet output requires a working PyArrow installation. "
            "In the project's active environment, run "
            "`python -m pip install -r requirements.txt` "
            "or `python -m pip install pyarrow`."
        ) from err

    table = pa.Table.from_pandas(df, preserve_index=False, safe=True)
    pq.write_table(
        table,
        destination,
        version="2.6",
        compression="snappy",
        use_deprecated_int96_timestamps=False,
        allow_truncated_timestamps=False,
        store_schema=True,
    )
    return destination


def read_parquet_dataset(path: Union[str, Path], schema: TableSchema) -> pd.DataFrame:
    """Read one Parquet file and restore its explicit merge table contract.

    Validate the declaration and supported merge dtypes before opening the
    file. Require exact column names/order and compatible stored types; do
    not parse text into numbers/dates or convert timestamp timezone labels.
    Restore integer/float widths and datetime units without losing values,
    then check declared keys and non-null constraints. Support the string,
    integer, floating, boolean and datetime targets of normalize_table_types.

    Ignore pandas index/attribute metadata when converting stored values.
    Keep integers in Arrow-backed columns until normalization so missing
    values cannot force a lossy float64 intermediate. Preserve row order
    and return a new RangeIndex. Native Arrow/pandas schema information may
    represent a datetime unit differently (seconds are stored as milliseconds);
    the caller's schema determines the returned dtype, subject to loss checks.

    Open only the exact file supplied, without discovering partition folders,
    loading sidecars, selecting CSV or writing anything. Filesystem errors
    propagate; malformed Parquet errors include the path. PyArrow is imported
    only when called, with installation guidance if unavailable.

    Raises:
        TypeError: If path or schema has an invalid argument type.
        ValueError: If the declaration, file structure/types, restored values
            or output constraints are invalid.
        ImportError: If PyArrow is unavailable, with installation guidance.
        OSError: If the file does not exist or cannot be opened/read.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("The Parquet source path must not be empty.")
    source = Path(path)
    if not isinstance(schema, TableSchema):
        raise TypeError("schema must be a TableSchema.")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as err:
        raise ImportError(
            "Parquet input requires a working PyArrow installation. "
            "In the project's active environment, run "
            "`python -m pip install -r requirements.txt` "
            "or `python -m pip install pyarrow`."
        ) from err

    try:
        empty = pd.DataFrame({
            name: pd.Series([], dtype=schema.dtypes[name])
            for name in schema.columns
        })
    except (TypeError, ValueError, KeyError) as err:
        raise ValueError(f"Invalid Parquet schema for '{source}': {err}") from err
    validate_table_contract(empty, schema, table_name=f"parquet_input '{source}'")
    if not schema.columns:
        raise ValueError("Parquet input requires at least one declared column.")

    try:
        expected = pa.Schema.from_pandas(empty, preserve_index=False)
    except pa.ArrowException as err:
        raise ValueError(f"Invalid Parquet schema for '{source}': {err}") from err
    for field in expected:
        dtype = field.type
        if not (
            pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_integer(dtype)
            or pa.types.is_floating(dtype)
            or pa.types.is_boolean(dtype)
            or pa.types.is_timestamp(dtype)
        ):
            raise ValueError(
                f"Parquet '{source}', column {field.name!r}, has an unsupported "
                f"restoration dtype: {schema.dtypes[field.name]!r}."
            )
    try:
        # ParquetFile reads one file, without dataset/partition discovery.
        with pq.ParquetFile(source) as parquet:
            actual = parquet.schema_arrow
            if actual.names != list(schema.columns):
                raise ValueError(
                    f"Parquet '{source}' columns differ from its schema. "
                    f"Expected: {list(schema.columns)}. Actual: {actual.names}."
                )
            for field in expected:
                stored = actual.field(field.name).type
                target = field.type
                compatible = stored.equals(target)
                if pa.types.is_timestamp(target):
                    compatible = (
                        pa.types.is_timestamp(stored) and stored.tz == target.tz
                    )
                elif pa.types.is_integer(target):
                    compatible = pa.types.is_integer(stored)
                elif pa.types.is_floating(target):
                    compatible = pa.types.is_floating(stored)
                elif pa.types.is_string(target) or pa.types.is_large_string(target):
                    compatible = (
                        pa.types.is_string(stored) or pa.types.is_large_string(stored)
                    )
                if not compatible:
                    raise ValueError(
                        f"Parquet '{source}', column {field.name!r}, has stored "
                        f"type {stored}; expected a compatible {target}."
                    )
            expected_rows = parquet.metadata.num_rows
            table = parquet.read().replace_schema_metadata(None)
            raw = table.to_pandas(
                ignore_metadata=True,
                types_mapper=lambda dtype: (
                    pd.ArrowDtype(dtype) if pa.types.is_integer(dtype) else None
                ),
            )
    except OSError:
        raise
    except pa.ArrowException as err:
        raise ValueError(f"Cannot read Parquet '{source}': {err}") from err

    result = normalize_table_types(raw, schema, table_name=f"parquet_input '{source}'")
    validate_table_contract(
        result, schema,
        table_name=f"parquet_input '{source}'",
        expected_rows=expected_rows,
    )
    return result
