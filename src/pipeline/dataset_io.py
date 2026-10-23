"""Dataset storage orchestration using shared schema-aware atomic writers."""

import re
from pathlib import Path
from typing import Union

import pandas as pd

from src.pipeline.atomic_io import write_file_atomically
from src.pipeline.csv_io import write_csv_dataset
from src.pipeline.parquet_io import write_parquet_dataset
from src.pipeline.schema import TableSchema
from src.pipeline.validator import validate_table_contract


def save_dataset(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    schema: TableSchema,
    *,
    dataset_name: str,
) -> Path:
    """Save one validated dataset as Parquet, falling back to encoded CSV.

    Require a dataset name containing only ASCII letters, digits, underscores
    or hyphens, without an extension or directory components. Validate the
    full table contract before creating directories or attempting either
    format. Invalid columns, dtypes, keys or non-null values never trigger
    fallback. Require at least one declared column, including for empty data.

    Create the destination directory when needed. Try exactly one Parquet
    write through write_file_atomically and write_parquet_dataset. On an
    ordinary exception, print its type/message and any cleanup notes before
    attempting encoded CSV through the same atomic helper. Never catch
    KeyboardInterrupt/SystemExit or other BaseException interruptions here.

    Print the actual saved format/path and return that Path. CSV output must
    be read with the paired schema-aware reader. Preserve other existing
    formats: a CSV fallback does not delete an earlier Parquet file. Select
    file_format='csv' when reading this CSV result if both formats exist;
    the reader's default remains Parquet-first, without freshness inference.

    If both attempts fail, raise RuntimeError with both paths/reasons and an
    ExceptionGroup cause retaining both original exceptions. Missing-engine
    installation guidance comes from the Parquet writer. Shared directory
    setup errors propagate directly because changing format cannot fix them.
    No input mutation, automatic cleaning, schema sidecar, run metadata,
    persistent logging or report/notebook update is performed.

    Args:
        df: Table already normalized to its declared contract.
        output_dir: Directory receiving '<dataset_name>.parquet' or '.csv'.
        schema: Exact column types and integrity constraints for this dataset.
        dataset_name: Extension-free filename stem, including for custom data.

    Raises:
        TypeError: If input argument types are invalid.
        ValueError: If the name, path or input table contract is invalid.
        OSError: If the shared output directory cannot be created/used.
        RuntimeError: If both format-specific atomic save attempts fail.
    """
    if not isinstance(output_dir, (str, Path)):
        raise TypeError("output_dir must be a string or pathlib.Path.")
    if isinstance(output_dir, str) and not output_dir.strip():
        raise ValueError("The output directory must not be empty.")
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string.")
    if re.fullmatch(r"[A-Za-z0-9_-]+", dataset_name) is None:
        raise ValueError(
            "dataset_name must contain only ASCII letters, digits, underscores "
            "or hyphens, without a file extension or directory components."
        )
    validate_table_contract(df, schema, table_name=dataset_name)
    if not schema.columns:
        raise ValueError("Dataset output requires at least one declared column.")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    failures = []
    details = []
    for file_format, writer in (
        ("parquet", write_parquet_dataset),
        ("csv", write_csv_dataset),
    ):
        destination = directory / f"{dataset_name}.{file_format}"
        try:
            saved = write_file_atomically(
                destination, lambda temporary: writer(df, temporary, schema)
            )
        except Exception as err:
            failures.append(err)
            reason = f"{type(err).__name__}: {err}"
            for note in getattr(err, "__notes__", ()):
                reason += f"\n{note}"
            details.append(f"{file_format.upper()} '{destination}': {reason}")
            if file_format == "parquet":
                print(f"Parquet save failed at '{destination}': {reason}")
                print("Trying CSV fallback.")
            continue

        print(f"Saved {file_format.upper()} to '{saved}'.")
        if file_format == "csv":
            print("Use file_format='csv' when reading this saved output.")
        return saved

    raise RuntimeError(
        f"Could not save dataset '{dataset_name}' in either format.\n"
        + "\n".join(details)
        + "\nCheck the format-specific errors, supported dtypes, destination "
        "permissions and available disk space."
    ) from ExceptionGroup("Dataset save attempts failed", failures)
