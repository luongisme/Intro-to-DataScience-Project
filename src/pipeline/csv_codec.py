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
