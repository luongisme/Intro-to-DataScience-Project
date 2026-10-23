"""Strict restoration of decoded CSV text to declared pandas column types."""

from datetime import datetime
from decimal import Decimal
import math
import re

import numpy as np
import pandas as pd


def restore_csv_column(values: pd.Series, dtype: str) -> pd.Series:
    """Restore one already decoded CSV text column without guessing values.

    Require a pandas StringDtype Series and a declared dtype string. Missing
    cells must already be restored by decode_csv_strings; do not decode again
    or treat empty/NA/NULL text as missing. Return an independent Series with
    the original index/name/order and exactly the requested pandas dtype.

    Parse integers directly with int and enforce the target's signed range.
    Parse decimal floats with float, retaining signed zero; reject overflow,
    underflow to zero and loss when converting to the declared float width.
    Wider-than-64-bit floats are unsupported, matching the CSV writer. Accept
    only True/False for booleans and ISO date/timestamp text for datetimes.
    Enforce timezone awareness and reject timestamp precision loss. A schema
    timezone restores its named zone from the recorded numeric offset.

    Do not read/write files, mutate input, validate keys, fill missing values
    or normalize arbitrary source data. Full table contracts are checked by
    the caller after all columns are restored. This function specifically
    reverses the text serialization used by the schema-aware CSV writer.

    Raises:
        TypeError: If values or dtype has an invalid argument type.
        ValueError: If the dtype is unsupported or cells cannot be restored
            without value, precision or missing-mask changes.
    """
    if not isinstance(values, pd.Series):
        raise TypeError("values must be a pandas Series.")
    if not isinstance(values.dtype, pd.StringDtype):
        raise TypeError("CSV dtype restoration requires decoded pandas string values.")
    if not isinstance(dtype, str):
        raise TypeError("The CSV dtype must be declared as a string.")
    if not dtype.strip():
        raise ValueError("The CSV dtype declaration must not be empty.")

    try:
        target = pd.api.types.pandas_dtype(dtype)
        missing = values.isna()
        if isinstance(target, pd.StringDtype):
            result = values.astype(target)
        elif pd.api.types.is_bool_dtype(target):
            parsed = []
            for cell in values:
                if pd.isna(cell):
                    parsed.append(pd.NA)
                elif cell in ("True", "False"):
                    parsed.append(cell == "True")
                else:
                    raise ValueError("Boolean cells must be exactly True or False.")
            result = pd.Series(parsed, dtype=target, index=values.index, name=values.name)
        elif pd.api.types.is_integer_dtype(target):
            bits = target.itemsize * 8
            unsigned = pd.api.types.is_unsigned_integer_dtype(target)
            lower = 0 if unsigned else -(2 ** (bits - 1))
            upper = 2 ** bits - 1 if unsigned else 2 ** (bits - 1) - 1
            parsed = []
            for cell in values:
                if pd.isna(cell):
                    parsed.append(pd.NA)
                    continue
                if re.fullmatch(r"[+-]?[0-9]+", cell) is None:
                    raise ValueError("Integer cells must contain decimal integer text.")
                number = int(cell)
                if not lower <= number <= upper:
                    raise ValueError(f"Integer {number} is outside the range of {dtype}.")
                parsed.append(number)
            result = pd.Series(parsed, dtype=target, index=values.index, name=values.name)
        elif pd.api.types.is_float_dtype(target):
            if target.itemsize > 8:
                raise ValueError("CSV floats wider than 64 bits are unsupported.")
            maximum = float(np.finfo(f"float{target.itemsize * 8}").max)
            pattern = re.compile(
                r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[+-]?inf"
            )
            parsed = []
            for cell in values:
                if pd.isna(cell):
                    parsed.append(float("nan"))
                    continue
                if pattern.fullmatch(cell) is None:
                    raise ValueError("Float cells must contain decimal text or signed inf.")
                number = float(cell)
                if math.isinf(number) and cell not in ("inf", "+inf", "-inf"):
                    raise ValueError("Float text overflows the supported numeric range.")
                if number == 0.0 and Decimal(cell) != 0:
                    raise ValueError("Float text underflows to zero.")
                if math.isfinite(number) and abs(number) > maximum:
                    raise ValueError(f"Float value is outside the range of {dtype}.")
                parsed.append(number)
            result = pd.Series(parsed, dtype=target, index=values.index, name=values.name)
            for absent, number, restored in zip(missing, parsed, result):
                if absent:
                    continue
                restored = float(restored)
                if restored != number or (
                    number == 0.0
                    and math.copysign(1.0, restored) != math.copysign(1.0, number)
                ):
                    raise ValueError(f"Float conversion would lose precision in {dtype}.")
        elif pd.api.types.is_datetime64_any_dtype(target):
            timezone = getattr(target, "tz", None)
            if timezone is None:
                timezone = getattr(getattr(target, "pyarrow_dtype", None), "tz", None)
            pattern = re.compile(
                r"(?P<year>[+-]?[0-9]{4,})-[0-9]{2}-[0-9]{2}"
                r"(?:[ T][0-9]{2}:[0-9]{2}:[0-9]{2}"
                r"(?:\.[0-9]{1,9})?"
                r"(?P<offset>Z|[+-][0-9]{2}:[0-9]{2}(?::[0-9]{2}(?:\.[0-9]{1,6})?)?)?)?"
            )
            parsed = []
            for cell in values:
                if pd.isna(cell):
                    parsed.append(pd.NaT)
                    continue
                match = pattern.fullmatch(cell)
                if match is None:
                    raise ValueError("Datetime cells must contain ISO date/timestamp text.")
                offset = match.group("offset")
                if (offset is not None) != (timezone is not None):
                    raise ValueError("Timestamp awareness does not match the declared timezone.")
                wall_time = cell[:-len(offset)] if offset is not None else cell
                stamp = pd.Timestamp(np.datetime64(wall_time))
                if stamp.year != int(match.group("year")):
                    raise ValueError("Datetime text exceeds its parsed calendar range.")
                if offset is not None:
                    # A fixed reference date isolates offset parsing from year limits.
                    zone = datetime.fromisoformat(f"2000-01-01T00:00:00{offset}").tzinfo
                    stamp = stamp.tz_localize(zone)
                parsed.append(stamp)
            result = pd.Series(parsed, dtype=target, index=values.index, name=values.name)
            for absent, original, restored in zip(missing, parsed, result):
                if not absent and original != restored:
                    raise ValueError(f"Datetime conversion would lose precision in {dtype}.")
        else:
            raise ValueError(f"Unsupported CSV dtype: {dtype}.")
        if not result.isna().equals(missing):
            raise ValueError("CSV conversion would change which values are missing.")
    except (TypeError, ValueError, ArithmeticError) as err:
        raise ValueError(
            f"Cannot restore CSV column {values.name!r} as {dtype!r}: {err}"
        ) from err
    return result.copy(deep=True)
