"""Format-independent atomic replacement for individual dataset files."""

import os
import tempfile
from pathlib import Path
from typing import Callable, Union


def write_file_atomically(
    path: Union[str, Path],
    writer: Callable[[Path], Path],
) -> Path:
    """Publish one completed file without overwriting the target during writing.

    Create a unique temporary file in the destination's existing parent so
    replacement stays on the same filesystem. Close its initial handle before
    calling writer, allowing Windows writers to reopen it. The callback must
    write and close that exact temporary file, then return the supplied Path.
    It owns format/schema validation; this helper never inspects dataset data.

    Replace the destination with os.replace only after the callback succeeds.
    Return the originally requested destination as a Path. Do not append an
    extension, create parents, choose another format, create backups or log.
    Concurrent successful calls publish complete files; the last replacement
    wins. This does not coordinate multiple output files or provide locking.

    On failure, including Python-level interruption, remove the temporary file
    and re-raise the original exception. If cleanup also fails, preserve that
    exception and add the temporary path and cleanup error as an exception
    note. A forced process termination may leave a temporary file behind.
    Atomic visibility depends on the filesystem's os.replace semantics; this
    helper does not guarantee storage durability after a power failure.

    Args:
        path: Exact destination filename; its parent must already exist.
        writer: Callback receiving the temporary Path and returning it after
            a complete write. Existing CSV/Parquet writers can be bound to
            their DataFrame/schema with a lambda or functools.partial.

    Raises:
        TypeError: If path is not a string/Path or writer is not callable.
        ValueError: If path is empty or writer does not return/write its path.
        OSError: If the temporary file cannot be created or replacement fails.
        BaseException: The callback's original failure or interruption.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    if isinstance(path, str) and not path.strip():
        raise ValueError("The destination path must not be empty.")
    if not callable(writer):
        raise TypeError("writer must be a callable accepting a temporary Path.")
    destination = Path(path)
    target = destination.absolute()

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".merge-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)

        written = writer(temporary)
        if not isinstance(written, Path) or written != temporary:
            raise ValueError("writer must return the exact temporary Path it received.")
        if not temporary.is_file():
            raise ValueError(f"writer did not produce a file at '{temporary}'.")
        os.replace(temporary, target)
    except BaseException as err:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                err.add_note(
                    f"Could not remove temporary file '{temporary}': {cleanup_error}. "
                    "Remove it manually once it is no longer in use."
                )
        raise
    return destination
