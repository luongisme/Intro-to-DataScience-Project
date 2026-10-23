"""Schema-aware Parquet I/O backed by the project's PyArrow engine."""

from pathlib import Path
from typing import Union

import pandas as pd

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
