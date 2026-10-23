"""Derive join output contracts without inspecting row values."""

from typing import Sequence

import pandas as pd

from src.pipeline.schema import JoinSpec, TableSchema


def derive_join_schema(
    left: TableSchema,
    right: TableSchema,
    join: JoinSpec,
) -> TableSchema:
    """Describe one validated join; call successively for a longer join plan.

    Keep left column order, then append selected right columns in their source
    schema order. Always retain right join keys. Coalesce only paired keys
    with the same name; retain differently named keys. Rename only selected
    non-join-key right columns and reject collisions instead of adding suffixes.
    Paired keys must declare equal pandas dtypes; no implicit key conversion.

    Missing rows on a side can make its integer/boolean columns nullable.
    Promote those numpy dtypes to matching pandas nullable dtypes, preserving
    integer width and signedness. Other supported dtypes already allow missing.
    The executor must apply safe dtypes BEFORE joining to avoid lossy pandas
    float promotion, including intermediate conversions of integer join keys.

    Non-null constraints assume runtime checks enforce cardinality and
    require_match. Preserve an existing complete unique key only on a side
    guaranteed present and not expanded; otherwise return no unique key.
    This is conservative, especially for outer and many-to-many joins.
    Do not read data, execute joins, prove row-level validity or mutate inputs.

    Raises:
        TypeError: If schemas or join have unexpected types.
        ValueError: If declarations, join keys, projection or names are invalid.
    """
    for label, schema in (("left", left), ("right", right)):
        if not isinstance(schema, TableSchema):
            raise TypeError(f"The {label} schema must be a TableSchema.")
        for field_name in ("columns", "unique_key", "non_nullable_columns"):
            declared = getattr(schema, field_name)
            if isinstance(declared, str) or not isinstance(declared, Sequence):
                raise ValueError(f"{label}.{field_name} must be a sequence of names.")
            if not all(isinstance(name, str) and name.strip() for name in declared):
                raise ValueError(f"{label}.{field_name} requires non-empty string names.")
            if len(declared) != len(set(declared)):
                raise ValueError(f"{label}.{field_name} declares duplicate names.")
            if field_name != "columns" and any(name not in schema.columns for name in declared):
                raise ValueError(f"{label}.{field_name} references missing columns.")
        if not schema.columns or not isinstance(schema.dtypes, dict):
            raise ValueError(f"The {label} schema requires columns and a dtype mapping.")
        if set(schema.dtypes) != set(schema.columns):
            raise ValueError(f"The {label} schema has inconsistent dtype declarations.")
        for column, declared_dtype in schema.dtypes.items():
            try:
                if not isinstance(declared_dtype, str):
                    raise ValueError("Declare dtypes as strings.")
                dtype = pd.api.types.pandas_dtype(declared_dtype)
                supported = (
                    isinstance(dtype, pd.StringDtype)
                    or pd.api.types.is_numeric_dtype(dtype)
                    or pd.api.types.is_datetime64_any_dtype(dtype)
                )
                if not supported or pd.api.types.is_complex_dtype(dtype):
                    raise ValueError("Unsupported merge dtype.")
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Invalid {label} dtype for '{column}': {declared_dtype!r}."
                ) from err
    if not isinstance(join, JoinSpec):
        raise TypeError("Schema derivation requires a JoinSpec.")
    if not isinstance(join.table, str) or not join.table.strip():
        raise ValueError("JoinSpec.table must be a non-empty string.")
    if join.how not in ("left", "inner", "right", "outer"):
        raise ValueError(f"Unsupported join type: {join.how!r}.")
    if join.validate not in ("1:1", "1:m", "m:1", "m:m"):
        raise ValueError(f"Unsupported join cardinality: {join.validate!r}.")
    if not isinstance(join.require_match, bool):
        raise TypeError("JoinSpec.require_match must be a boolean.")
    if join.on is not None:
        if join.left_on is not None or join.right_on is not None:
            raise ValueError("Specify either on or left_on/right_on, not both.")
        declarations = (join.on, join.on)
    else:
        if join.left_on is None or join.right_on is None:
            raise ValueError("Specify on or both left_on and right_on.")
        declarations = (join.left_on, join.right_on)
    key_groups = []
    for label, schema, declared in zip(("left", "right"), (left, right), declarations):
        if isinstance(declared, str):
            keys = [declared]
        elif isinstance(declared, Sequence):
            keys = list(declared)
        else:
            raise ValueError(f"The {label} join keys must be column names.")
        if not keys or not all(isinstance(key, str) and key.strip() for key in keys):
            raise ValueError(f"The {label} join keys require non-empty string names.")
        if len(keys) != len(set(keys)):
            raise ValueError(f"The {label} join keys declare duplicate names.")
        missing = [key for key in keys if key not in schema.columns]
        if missing:
            raise ValueError(f"The {label} schema is missing join keys: {missing}.")
        key_groups.append(keys)
    left_keys, right_keys = key_groups
    if len(left_keys) != len(right_keys):
        raise ValueError("left_on and right_on must contain the same number of keys.")
    for left_key, right_key in zip(left_keys, right_keys):
        if not pd.api.types.is_dtype_equal(
            pd.api.types.pandas_dtype(left.dtypes[left_key]),
            pd.api.types.pandas_dtype(right.dtypes[right_key]),
        ):
            raise ValueError(
                f"Join key dtypes differ for '{left_key}' and '{right_key}'. "
                "Normalize paired keys to the same dtype before joining."
            )

    selected = list(right.columns) if join.columns is None else join.columns
    if isinstance(selected, str) or not isinstance(selected, Sequence):
        raise ValueError("JoinSpec.columns must be a sequence of column names.")
    if not all(isinstance(name, str) and name.strip() for name in selected):
        raise ValueError("JoinSpec.columns requires non-empty string names.")
    if len(selected) != len(set(selected)):
        raise ValueError("JoinSpec.columns declares duplicate names.")
    missing = [name for name in selected if name not in right.columns]
    if missing:
        raise ValueError(f"Selected right columns do not exist: {missing}.")
    included = set(selected) | set(right_keys)
    selected = [name for name in right.columns if name in included]
    if not isinstance(join.rename, dict):
        raise TypeError("JoinSpec.rename must be a dictionary.")
    for original, renamed in join.rename.items():
        if original not in selected or original in right_keys:
            raise ValueError("Rename only selected non-join-key right columns.")
        if not isinstance(renamed, str) or not renamed.strip():
            raise ValueError("Renamed columns require non-empty string names.")

    shared = {a for a, b in zip(left_keys, right_keys) if a == b}
    columns = list(left.columns)
    dtypes = dict(left.dtypes)
    for original in selected:
        if original in shared:
            continue
        name = join.rename.get(original, original)
        if name in columns:
            raise ValueError(
                f"Join column collision: '{name}'. Rename or select columns explicitly."
            )
        columns.append(name)
        dtypes[name] = right.dtypes[original]

    left_present = join.how in ("left", "inner")
    right_present = join.how in ("right", "inner") or join.require_match
    left_non_null = set(left.non_nullable_columns) | set(left.unique_key)
    right_non_null = set(right.non_nullable_columns) | set(right.unique_key)
    if join.require_match:
        left_non_null.update(left_keys)
        if join.how in ("left", "inner"):
            right_non_null.update(right_keys)
    non_null = {name for name in left.columns if left_present and name in left_non_null}
    non_null.update(
        join.rename.get(name, name)
        for name in selected
        if name not in shared and right_present and name in right_non_null
    )
    for name in shared:
        if join.how == "left":
            complete = name in left_non_null
        elif join.how == "right":
            complete = name in right_non_null
        elif join.how == "inner":
            complete = name in left_non_null or name in right_non_null
        else:
            complete = name in right_non_null and (name in left_non_null or join.require_match)
        if complete:
            non_null.add(name)
        else:
            non_null.discard(name)

    nullable = {name for name in left.columns if not left_present and name not in shared}
    nullable.update(
        join.rename.get(name, name)
        for name in selected
        if not right_present and name not in shared
    )
    for name in nullable:
        dtype = pd.api.types.pandas_dtype(dtypes[name])
        if isinstance(dtype, pd.api.extensions.ExtensionDtype):
            continue
        if pd.api.types.is_integer_dtype(dtype):
            prefix = "UInt" if pd.api.types.is_unsigned_integer_dtype(dtype) else "Int"
            dtypes[name] = f"{prefix}{dtype.itemsize * 8}"
        elif pd.api.types.is_bool_dtype(dtype):
            dtypes[name] = "boolean"

    unique_key = []
    if left.unique_key and left_present and join.validate in ("1:1", "m:1"):
        unique_key = list(left.unique_key)
    elif right.unique_key and right_present and join.validate in ("1:1", "1:m"):
        if all(name in included for name in right.unique_key):
            unique_key = [join.rename.get(name, name) for name in right.unique_key]
    if not all(name in non_null for name in unique_key):
        unique_key = []
    return TableSchema(
        columns=columns,
        dtypes=dtypes,
        unique_key=unique_key,
        non_nullable_columns=[name for name in columns if name in non_null],
    )
