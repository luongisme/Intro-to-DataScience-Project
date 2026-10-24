"""Read-only key and entity-consistency checks for full-merge EDA."""

from collections.abc import Mapping, Sequence

import pandas as pd

from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA, SOURCE_PRIMARY_KEYS
from src.pipeline.contracts.validator import validate_source_keys

from .profiling import SOURCE_SPECS, blank_string_mask, require_columns


def profile_keys(
    df: pd.DataFrame,
    keys: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Summarize missing, blank and repeated keys on the supplied item table.

    Default to the five existing source keys. Composite keys are checked as
    tuples. Null/blank components make a row invalid; missing and blank counts
    may overlap, while invalid_rows counts their union. Repetition counts use
    valid rows only. Excess copies exclude the first row per repeated key.
    Repeated order/product/seller/customer IDs are expected at item grain;
    these counts do not by themselves indicate duplicate source records.

    Return one row per named key without raising on dirty key values. Invalid
    declarations raise TypeError/ValueError/KeyError; empty input is allowed.
    """
    require_columns(df)
    selected = SOURCE_PRIMARY_KEYS if keys is None else keys
    if not isinstance(selected, Mapping) or not selected:
        raise ValueError("keys must be a non-empty mapping of names to column sequences.")
    rows = []
    for name, columns in selected.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each key profile requires a non-empty name.")
        require_columns(df, columns)
        if not columns:
            raise ValueError(f"Key '{name}' requires at least one column.")
        columns = list(columns)
        key_data = df.loc[:, columns]
        missing = key_data.isna().any(axis=1)
        blank = pd.concat([blank_string_mask(key_data[column]) for column in columns], axis=1).any(axis=1)
        invalid = missing | blank
        valid_keys = key_data.loc[~invalid]
        repeated = valid_keys.duplicated(keep=False)
        distinct = len(valid_keys.drop_duplicates())
        rows.append([
            name, tuple(columns), len(df), int(missing.sum()), int(blank.sum()),
            int(invalid.sum()), len(valid_keys), distinct,
            len(valid_keys.loc[repeated].drop_duplicates()), int(repeated.sum()),
            len(valid_keys) - distinct,
        ])
    return pd.DataFrame(rows, columns=[
        "key", "columns", "rows", "missing_rows", "blank_rows", "invalid_rows",
        "valid_rows", "distinct_valid_keys", "repeated_key_groups",
        "rows_in_repeated_keys", "excess_key_rows",
    ])


def profile_entity_consistency(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Check order/customer/product/seller attributes within each valid ID.

    Reuse each source's selected columns and key. A value-versus-null mismatch
    is a conflict; an attribute missing on every row of one ID is consistent,
    though still incomplete. Case/whitespace differences remain distinct.
    Null/blank ID rows are excluded from groups and reported explicitly.

    Return summary, attributes and conflicts tables. Conflicts lists every
    conflicting entity/attribute with its ID and distinct-value count (including
    missing), rather than choosing a representative value. Percentages use
    valid distinct entities; empty denominators yield NaN. Input is unchanged.
    """
    require_columns(df)
    summaries, attributes, conflicts = [], [], []
    for spec in SOURCE_SPECS[1:]:
        require_columns(df, spec.usecols)
        key = SOURCE_PRIMARY_KEYS[spec.dataset_name][0]
        attribute_columns = [column for column in spec.usecols if column != key]
        valid_mask = df[key].notna() & ~blank_string_mask(df[key])
        valid = df.loc[valid_mask, spec.usecols]
        groups = valid.groupby(key, sort=False, observed=True, dropna=False)
        distinct = groups[attribute_columns].nunique(dropna=False)
        sizes = groups.size()
        any_conflict = distinct.gt(1).any(axis=1)
        total = len(distinct)
        summaries.append([
            spec.dataset_name, key, len(df), int((~valid_mask).sum()), len(valid),
            total, int(any_conflict.sum()),
            100 * int(any_conflict.sum()) / total if total else float("nan"),
            int(sizes.loc[any_conflict].sum()),
        ])
        for column in attribute_columns:
            affected = distinct[column].gt(1)
            attributes.append([
                spec.dataset_name, column, total, int(affected.sum()),
                100 * int(affected.sum()) / total if total else float("nan"),
                int(sizes.loc[affected].sum()),
            ])
            for entity_id, count in distinct.loc[affected, column].items():
                conflicts.append([
                    spec.dataset_name, key, entity_id, column, int(count),
                    int(sizes.loc[entity_id]),
                ])
    return {
        "summary": pd.DataFrame(summaries, columns=[
            "entity", "key", "item_rows", "invalid_key_rows", "valid_key_rows",
            "valid_entities", "conflicted_entities", "conflict_pct",
            "item_rows_in_conflicted_entities",
        ]),
        "attributes": pd.DataFrame(attributes, columns=[
            "entity", "attribute", "valid_entities", "conflicted_entities",
            "conflict_pct", "item_rows_in_conflicted_entities",
        ]),
        "conflicts": pd.DataFrame(conflicts, columns=[
            "entity", "key_column", "key_value", "attribute",
            "distinct_values_including_missing", "item_rows",
        ]),
    }


def build_entity_views(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return independent in-memory source-grain views of the observed full input.

    Require a unique, complete order-item key via the existing merge validator,
    then recheck all entity attributes. Refuse null/blank entity IDs or any
    conflict instead of silently keeping a first value. Raise ValueError with
    the affected entity names; inspect profile_entity_consistency for details.

    Return five tables using the original source column selections, row order
    of first occurrence and original dtypes. Deduplication is only in temporary
    views after consistency checks; the original item table remains untouched.
    Views contain observed entities only, not absent raw-source entities.
    """
    require_columns(df, FULL_MERGE_SCHEMA.columns)
    validate_source_keys(
        df, "order_items", key_columns=SOURCE_PRIMARY_KEYS["order_items"],
    )
    consistency = profile_entity_consistency(df)["summary"]
    blocked = consistency.loc[
        consistency["invalid_key_rows"].gt(0) | consistency["conflicted_entities"].gt(0),
        "entity",
    ].tolist()
    if blocked:
        raise ValueError(
            f"Cannot build reliable entity views for: {blocked}. "
            "Inspect profile_entity_consistency(df) before selecting entity values."
        )
    return {
        spec.dataset_name: df.loc[:, spec.usecols].drop_duplicates(
            subset=SOURCE_PRIMARY_KEYS[spec.dataset_name],
        ).copy(deep=True)
        for spec in SOURCE_SPECS
    }
