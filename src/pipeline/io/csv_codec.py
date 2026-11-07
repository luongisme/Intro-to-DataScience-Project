"""Reversible text-cell conventions for schema-aware interim CSV files."""

import pandas as pd


CSV_MISSING_TOKEN = r"\N"
CSV_ESCAPE_PREFIX = "\\"


def encode_csv_strings(values: pd.Series) -> pd.Series:
    """Encode one pandas string column without changing the input.

    Represent missing values with CSV_MISSING_TOKEN. Prefix every non-missing
    string starting with CSV_ESCAPE_PREFIX with one additional escape prefix.
    This makes the representation unambiguous for literal missing tokens,
    existing escape sequences and repeated leading backslashes. Keep all other
    strings exactly, including empty strings, NA/NULL text and leading zeros.

    Return an independent Series with the original string dtype, index, name
    and row order. The encoded result contains no missing values. Accept pandas
    StringDtype variants, including different storage and missing conventions;
    reject object, numeric and other dtypes rather than coercing their values.

    This is a cell-value encoding, not CSV quoting or a complete file writer.
    The paired reader must disable pandas' default missing-value inference
    and decode before restoring schema dtypes. Writers must use the same token
    for missing non-text cells. Neither file I/O, schema/key validation nor
    numeric/datetime serialization belongs to this function.

    Args:
        values: A column already declared as a pandas string dtype.

    Raises:
        TypeError: If values is not a Series with a pandas string dtype.
    """
    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series.")
    if not isinstance(values.dtype, pd.StringDtype):
        raise TypeError(
            "CSV text encoding requires a pandas string dtype. "
            "Normalize the column to its declared string schema first."
        )
    encoded = values.copy(deep=True)
    escaped = encoded.str.startswith(CSV_ESCAPE_PREFIX, na=False)
    encoded = encoded.mask(escaped, CSV_ESCAPE_PREFIX + encoded)
    return encoded.fillna(CSV_MISSING_TOKEN)


def decode_csv_strings(values: pd.Series) -> pd.Series:
    """Decode one encoded CSV text column without changing the input.

    Interpret CSV_MISSING_TOKEN as missing and remove exactly one prefix from
    doubled leading CSV_ESCAPE_PREFIX strings. Determine both masks from the
    original encoded cells before replacing anything, so a literal missing
    token restored by unescaping never becomes a missing value itself.
    Leave empty strings, NA/NULL text, leading zeros and other text unchanged.

    Require a pandas StringDtype Series containing no inferred missing cells.
    Read encoded CSV text with dtype="string", keep_default_na=False and
    na_filter=False before decoding. Use skip_blank_lines=False when parsing
    to retain empty/whitespace records. Reject a single leading escape prefix
    except for the exact missing token: such a cell cannot come from the
    paired encoder. Do not guess whether malformed cells are raw text.

    Return an independent Series with its original dtype, index, name and row
    order. The dtype's missing convention determines the restored missing
    scalar. Decode exactly once after CSV parsing; this is not an idempotent
    transformation or a raw CSV decoder. Do not read/write files, convert
    numeric/datetime columns, validate schema/keys or fill missing attributes.

    Args:
        values: Encoded text cells read without missing-value inference.

    Raises:
        TypeError: If values is not a Series with a pandas string dtype.
        ValueError: If cells contain inferred missing or invalid escapes.
    """
    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series.")
    if not isinstance(values.dtype, pd.StringDtype):
        raise TypeError(
            "CSV text decoding requires a pandas string dtype. "
            "Read encoded text cells with dtype='string'."
        )
    if values.isna().any():
        raise ValueError(
            "Encoded CSV text contains unexpected missing cells. "
            "Read CSV with dtype='string', keep_default_na=False and "
            "na_filter=False before decoding."
        )
    missing = values.eq(CSV_MISSING_TOKEN)
    prefixed = values.str.startswith(CSV_ESCAPE_PREFIX, na=False)
    escaped = values.str.startswith(CSV_ESCAPE_PREFIX * 2, na=False)
    invalid = prefixed & ~missing & ~escaped
    if invalid.any():
        raise ValueError(
            f"Invalid CSV text encoding: {int(invalid.sum())} cell(s) start "
            "with an unescaped prefix. Use CSV text produced by the paired encoder."
        )
    decoded = values.copy(deep=True)
    decoded = decoded.mask(escaped, decoded.str.slice(start=len(CSV_ESCAPE_PREFIX)))
    return decoded.mask(missing, pd.NA)
