"""Schema-aware dataset saving with fallback and format-selectable reading."""

import re
from pathlib import Path
from typing import Literal, Union

import pandas as pd

from src.pipeline.atomic_io import write_file_atomically
from src.pipeline.csv_io import read_csv_dataset, write_csv_dataset
from src.pipeline.parquet_io import read_parquet_dataset, write_parquet_dataset
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


def read_dataset(
    output_dir: Union[str, Path],
    schema: TableSchema,
    *,
    dataset_name: str,
    file_format: Literal["parquet", "csv"] = "parquet",
) -> pd.DataFrame:
    """Read a saved dataset with an explicit schema and format preference.

    Validate path/name/format arguments and the declared table contract before
    attempting files. Invalid declarations never trigger fallback. Require a
    non-empty schema with the exact columns, dtypes, keys and non-null rules
    used for saving. Input paths/names follow the same rules as save_dataset.

    With the default or explicit 'parquet', try read_parquet_dataset first.
    On an ordinary exception, report the file, error type/message and notes,
    then try read_csv_dataset exactly once. Per-file failures, including file
    contract violations, can trigger fallback; do not repair or drop records.
    If both fail, raise RuntimeError with both paths/reasons and an
    ExceptionGroup cause preserving the original exceptions.

    With 'csv', announce the explicit choice and read CSV directly, without
    checking Parquet. On failure, re-raise the original exception with a note
    naming the dataset and CSV path. Never catch Python-level interruptions
    such as KeyboardInterrupt/SystemExit to continue reading another format.

    Announce the actual successful format/path and return the reader's new
    DataFrame with restored dtypes and file row order. When both formats exist,
    always prefer Parquet in the default mode, regardless of modification
    times; select 'csv' to read a CSV saved by a recent fallback. Do not create
    directories, write files, consult sidecars or infer a schema/latest run.

    Args:
        output_dir: Directory containing the extension-free named dataset.
        schema: Exact expected column types and integrity constraints.
        dataset_name: Filename stem accepted by save_dataset.
        file_format: 'parquet' (with CSV fallback) or direct 'csv'.

    Raises:
        TypeError: If input argument types are invalid.
        ValueError: If path/name/format arguments or schema are invalid.
        RuntimeError: If both Parquet and CSV attempts fail.
        Exception: An explicit CSV read's original failure, with path context.
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
    if not isinstance(file_format, str):
        raise TypeError("file_format must be 'parquet' or 'csv'.")
    if file_format not in ("parquet", "csv"):
        raise ValueError("file_format must be 'parquet' or 'csv'.")
    if not isinstance(schema, TableSchema):
        raise TypeError("schema must be a TableSchema.")
    try:
        empty = pd.DataFrame({
            name: pd.Series([], dtype=schema.dtypes[name])
            for name in schema.columns
        })
    except (TypeError, ValueError, KeyError) as err:
        raise ValueError(f"Invalid schema for dataset '{dataset_name}': {err}") from err
    validate_table_contract(empty, schema, table_name=dataset_name)
    if not schema.columns:
        raise ValueError("Dataset input requires at least one declared column.")

    if file_format == "csv":
        print("CSV selected. Reading CSV directly.")
        attempts = (("csv", read_csv_dataset),)
    else:
        attempts = (("parquet", read_parquet_dataset), ("csv", read_csv_dataset))
    directory = Path(output_dir)
    failures = []
    details = []
    for current_format, reader in attempts:
        source = directory / f"{dataset_name}.{current_format}"
        try:
            result = reader(source, schema)
        except Exception as err:
            if file_format == "csv":
                err.add_note(f"Cannot read CSV dataset '{dataset_name}' from '{source}'.")
                raise
            failures.append(err)
            reason = f"{type(err).__name__}: {err}"
            for note in getattr(err, "__notes__", ()):
                reason += f"\n{note}"
            details.append(f"{current_format.upper()} '{source}': {reason}")
            if current_format == "parquet":
                print(f"Parquet read failed at '{source}': {reason}")
                print("Trying CSV fallback.")
            continue

        print(f"Loaded {current_format.upper()} from '{source}'.")
        return result

    raise RuntimeError(
        f"Could not read dataset '{dataset_name}' in either format.\n"
        + "\n".join(details)
        + "\nCheck that the files exist, match the supplied schema and can be read."
    ) from ExceptionGroup("Dataset read attempts failed", failures)
