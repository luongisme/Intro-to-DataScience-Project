"""Read-only key and entity-consistency checks for full-merge EDA."""

from collections.abc import Mapping, Sequence

import re

import numpy as np
import pandas as pd

from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA, ORDERS_SPEC, SOURCE_PRIMARY_KEYS
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


def summarize_condition_by_group(
    df: pd.DataFrame,
    condition: pd.Series,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Count a temporary boolean condition by one or more observed groups.

    Require a bool/nullable-boolean Series with the same index and row order
    as df. Missing condition values remain unknown, not False. Return group
    rows, evaluated/unknown/affected counts, affected_pct_evaluated (evaluated
    denominator) and affected_pct_all (confirmed True divided by all group
    rows). An all-unknown group has an undefined evaluated percentage.

    Retain null group keys, literal blanks and observed categorical levels;
    do not create combinations that never occur. Group percentages describe
    the supplied row grain, not distinct entities inferred from column names.
    Empty input returns the same output columns with no observed groups.
    No loading, transformation of input columns, display or saving occurs.
    """
    require_columns(df, group_columns)
    if not group_columns:
        raise ValueError("Select at least one grouping column.")
    metrics = [
        "rows", "evaluated_rows", "unknown_rows", "affected_rows",
        "affected_pct_evaluated", "affected_pct_all",
    ]
    if set(group_columns).intersection(metrics):
        raise ValueError(f"Grouping names must not overlap output metrics: {metrics}.")
    if not isinstance(condition, pd.Series) or not pd.api.types.is_bool_dtype(condition.dtype):
        raise TypeError("condition must be a bool or nullable-boolean pandas Series.")
    if not condition.index.equals(df.index):
        raise ValueError("condition must have the same index and row order as df.")
    counts = pd.DataFrame({
        "rows": 1,
        "evaluated_rows": condition.notna().to_numpy(dtype="int64"),
        "affected_rows": condition.fillna(False).to_numpy(dtype="int64"),
    }, index=df.index)
    result = counts.groupby(
        [df[column] for column in group_columns],
        dropna=False, sort=False, observed=True,
    ).sum().reset_index()
    result["unknown_rows"] = result["rows"] - result["evaluated_rows"]
    result["affected_pct_evaluated"] = (
        100 * result["affected_rows"] / result["evaluated_rows"].where(result["evaluated_rows"].ne(0))
    )
    result["affected_pct_all"] = (
        100 * result["affected_rows"] / result["rows"].where(result["rows"].ne(0))
    )
    return result.loc[:, [*group_columns, *metrics]]


def profile_text_formats(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    patterns: Mapping[str, str] | None = None,
    sample_size: int = 5,
) -> dict[str, pd.DataFrame]:
    """Inspect original text values, optional regex shapes and candidate variants.

    Return summary, variants and examples tables. Never cast values to text or
    normalize the input. Regex full matches use original strings, including
    blanks; missing/non-string values are outside the regex denominator and
    counted separately. Checks overlap, so their counts must not be added.

    Candidate variant groups share strip().casefold() comparison keys, used
    only in this report; these keys do not establish semantic/place identity.
    Blank strings are excluded from variant groups. Keep all variant forms in
    RAM; examples are capped per column/check and include positional row
    references so duplicate input index labels remain unambiguous.
    """
    require_columns(df, columns)
    if not columns:
        raise ValueError("Select at least one text column.")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        raise ValueError("sample_size must be a non-negative integer.")
    if patterns is None:
        patterns = {}
    if not isinstance(patterns, Mapping):
        raise TypeError("patterns must map selected column names to regex strings.")
    if set(patterns).difference(columns):
        raise ValueError("Regex patterns must refer only to selected columns.")
    compiled = {}
    for column, pattern in patterns.items():
        if not isinstance(pattern, str):
            raise TypeError(f"Regex for '{column}' must be a string.")
        try:
            compiled[column] = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"Invalid regex for '{column}': {error}") from error
    summaries, variants, examples = [], [], []
    for column in columns:
        series = df[column]
        is_string = series.astype(object).map(lambda value: isinstance(value, str))
        text = series.loc[is_string]
        masks = {
            "non_string": series.notna() & ~is_string,
            "blank": blank_string_mask(series),
            "edge_whitespace": series.astype(object).map(lambda value: isinstance(value, str) and value != value.strip()),
            "repeated_whitespace": series.astype(object).map(lambda value: isinstance(value, str) and re.search(r"\s{2,}", value) is not None),
            "control_character": series.astype(object).map(lambda value: isinstance(value, str) and re.search(r"[\x00-\x1f\x7f]", value) is not None),
            "replacement_character": series.astype(object).map(lambda value: isinstance(value, str) and "\ufffd" in value),
        }
        if column in compiled:
            masks["pattern_mismatch"] = series.astype(object).map(
                lambda value: isinstance(value, str) and compiled[column].fullmatch(value) is None
            )
        forms = pd.DataFrame([
            [value, value.strip().casefold(), int(count)]
            for value, count in text.value_counts(sort=False).items() if value.strip()
        ], columns=["raw_value", "comparison_key", "rows"])
        form_counts = forms.groupby("comparison_key", sort=False)["raw_value"].nunique()
        candidate_keys = form_counts.loc[form_counts.gt(1)]
        for value, key, count in forms.loc[forms["comparison_key"].isin(candidate_keys.index)].itertuples(index=False, name=None):
            variants.append([column, key, repr(value), count, int(candidate_keys.loc[key])])
        lengths = text.map(len)
        mismatch = int(masks["pattern_mismatch"].sum()) if column in compiled else 0
        evaluated = len(text) if column in compiled else 0
        summaries.append([
            column, len(series), int(series.isna().sum()), len(text),
            *[int(masks[name].sum()) for name in (
                "non_string", "blank", "edge_whitespace", "repeated_whitespace",
                "control_character", "replacement_character",
            )],
            lengths.min() if len(lengths) else float("nan"),
            lengths.max() if len(lengths) else float("nan"),
            patterns.get(column), evaluated, mismatch,
            100 * mismatch / evaluated if evaluated else float("nan"), len(candidate_keys),
        ])
        for check, mask in masks.items():
            for position in np.flatnonzero(mask.to_numpy(dtype=bool))[:sample_size]:
                examples.append([column, check, int(position), df.index[position], repr(series.iloc[position])])
    return {
        "summary": pd.DataFrame(summaries, columns=[
            "column", "rows", "missing", "string_rows", "non_string", "blank",
            "edge_whitespace", "repeated_whitespace", "control_character",
            "replacement_character", "min_length", "max_length", "pattern",
            "pattern_evaluated", "pattern_mismatch", "pattern_mismatch_pct", "candidate_variant_groups",
        ]),
        "variants": pd.DataFrame(variants, columns=[
            "column", "comparison_key", "raw_repr", "rows", "distinct_forms",
        ]),
        "examples": pd.DataFrame(examples, columns=[
            "column", "check", "row_position", "index_label", "raw_repr",
        ]),
    }


def profile_timestamp_ordering(
    orders: pd.DataFrame,
    *,
    sample_size: int = 5,
) -> dict[str, pd.DataFrame]:
    """Check raw timestamp ordering once per order, without calculating features.

    Require a complete unique order_id and datetime columns from ORDERS_SPEC.
    Return summary, conditions and examples. Physical timestamp comparisons
    use the stored clock; estimated_before_purchase compares calendar dates
    because estimates can be recorded at midnight. Equality is not a violation.

    Missing either timestamp makes the corresponding condition unknown (NA),
    rather than False. Percentages use comparable orders; no rows are removed.
    Conditions retain the original index/order for grouped diagnostics. Examples
    are bounded per check, but all diagnostic conditions remain in memory.
    No durations, lateness labels, business-day counts or time slots are created.
    """
    require_columns(orders, ["order_id", "order_status", *ORDERS_SPEC.date_columns])
    validate_source_keys(orders, "orders")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
        raise ValueError("sample_size must be a non-negative integer.")
    for column in ORDERS_SPEC.date_columns:
        if not pd.api.types.is_datetime64_any_dtype(orders[column].dtype):
            raise TypeError(f"Column '{column}' must have a datetime dtype.")
    purchase = "order_purchase_timestamp"
    approved = "order_approved_at"
    carrier = "order_delivered_carrier_date"
    customer = "order_delivered_customer_date"
    pairs = [
        ("approved_before_purchase", purchase, approved, "timestamp"),
        ("carrier_before_approval", approved, carrier, "timestamp"),
        ("customer_before_carrier", carrier, customer, "timestamp"),
        ("carrier_before_purchase", purchase, carrier, "timestamp"),
        ("customer_before_approval", approved, customer, "timestamp"),
        ("customer_before_purchase", purchase, customer, "timestamp"),
        ("estimated_before_purchase", purchase, "order_estimated_delivery_date", "calendar_date"),
    ]
    conditions = pd.DataFrame(index=orders.index)
    summaries, examples = [], []
    for check, start, end, unit in pairs:
        left, right = orders[start], orders[end]
        comparable = left.notna() & right.notna()
        if unit == "calendar_date":
            left, right = left.dt.normalize(), right.dt.normalize()
        condition = right.lt(left).astype("boolean").where(comparable, pd.NA)
        conditions[check] = condition
        evaluated, affected = int(comparable.sum()), int(condition.sum())
        summaries.append([
            check, start, end, unit, len(orders), evaluated,
            len(orders) - evaluated, affected,
            100 * affected / evaluated if evaluated else float("nan"),
        ])
        for position in np.flatnonzero(condition.fillna(False).to_numpy(dtype=bool))[:sample_size]:
            examples.append([
                check, int(position), orders["order_id"].iloc[position],
                orders["order_status"].iloc[position], start, end, unit,
                orders[start].iloc[position], orders[end].iloc[position],
            ])
    return {
        "summary": pd.DataFrame(summaries, columns=[
            "check", "start_column", "end_column", "comparison_unit", "orders",
            "evaluated_orders", "unassessable_orders", "flagged_orders", "flagged_pct_evaluated",
        ]),
        "conditions": conditions,
        "examples": pd.DataFrame(examples, columns=[
            "check", "row_position", "order_id", "order_status", "start_column",
            "end_column", "comparison_unit", "start_value", "end_value",
        ]),
    }


def profile_order_composition(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Describe item/seller/deadline/location multiplicity without order features.

    Reuse validated entity views to require complete unique item keys and
    consistent entity attributes. Return order-level diagnostic counts, observed
    order-seller deadline groups and a summary with explicit denominators.
    Missing deadlines/states remain separate from distinct non-null values.

    Item numbering is consecutive only for positive integer values 1..N with
    no gaps. State equality compares original, present, non-blank strings;
    it does not infer distance, validate locations or create a route-type label.
    Min/max deadlines describe a range, not a chosen order-level deadline rule.
    All results are temporary summaries; the supplied full table is unchanged.
    """
    views = build_entity_views(df)
    if not pd.api.types.is_datetime64_any_dtype(df["shipping_limit_date"].dtype):
        raise TypeError("shipping_limit_date must have a datetime dtype.")
    ids = df["order_item_id"]
    if (
        not pd.api.types.is_numeric_dtype(ids.dtype)
        or pd.api.types.is_bool_dtype(ids.dtype)
        or pd.api.types.is_complex_dtype(ids.dtype)
    ):
        raise TypeError("order_item_id must have a real numeric dtype.")
    finite = np.isfinite(ids).fillna(False)
    positive_integer = finite & ids.gt(0) & ids.where(finite).mod(1).eq(0)
    customer_present = df["customer_state"].notna() & ~blank_string_mask(df["customer_state"])
    seller_present = df["seller_state"].notna() & ~blank_string_mask(df["seller_state"])
    state_comparable = customer_present & seller_present
    records = df.loc[:, [
        "order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date",
        "customer_state", "seller_state",
    ]].copy()
    records["_invalid_item_number"] = ~positive_integer
    records["_missing_deadline"] = df["shipping_limit_date"].isna()
    records["_missing_customer_state"] = ~customer_present
    records["_missing_seller_state"] = ~seller_present
    records["_equal_states"] = state_comparable & df["customer_state"].eq(df["seller_state"]).fillna(False)
    records["_different_states"] = state_comparable & df["customer_state"].ne(df["seller_state"]).fillna(False)
    records["_uncomparable_states"] = ~state_comparable
    orders = records.groupby("order_id", sort=False, observed=True, dropna=False).agg(
        item_rows=("order_item_id", "size"), min_item_id=("order_item_id", "min"),
        max_item_id=("order_item_id", "max"), distinct_item_ids=("order_item_id", "nunique"),
        invalid_item_id_rows=("_invalid_item_number", "sum"), sellers=("seller_id", "nunique"),
        products=("product_id", "nunique"), deadlines=("shipping_limit_date", "nunique"),
        missing_deadline_items=("_missing_deadline", "sum"),
        seller_states_non_null=("seller_state", "nunique"), customer_states_non_null=("customer_state", "nunique"),
        missing_seller_state_items=("_missing_seller_state", "sum"),
        missing_customer_state_items=("_missing_customer_state", "sum"),
        items_with_equal_states=("_equal_states", "sum"), items_with_different_states=("_different_states", "sum"),
        items_with_uncomparable_states=("_uncomparable_states", "sum"),
    ).reset_index()
    orders["consecutive_item_numbers"] = (
        orders["invalid_item_id_rows"].eq(0) & orders["min_item_id"].eq(1)
        & orders["max_item_id"].eq(orders["item_rows"])
        & orders["distinct_item_ids"].eq(orders["item_rows"])
    )
    deadlines = records.groupby(["order_id", "seller_id"], sort=False, observed=True, dropna=False).agg(
        item_rows=("order_item_id", "size"), distinct_deadlines=("shipping_limit_date", "nunique"),
        missing_deadline_items=("_missing_deadline", "sum"),
        min_deadline=("shipping_limit_date", "min"), max_deadline=("shipping_limit_date", "max"),
    ).reset_index()
    metrics = [
        ("Multiple items", "orders", len(orders), orders["item_rows"].gt(1).sum()),
        ("Multiple products", "orders", len(orders), orders["products"].gt(1).sum()),
        ("Multiple sellers", "orders", len(orders), orders["sellers"].gt(1).sum()),
        ("Multiple non-null deadlines", "orders", len(orders), orders["deadlines"].gt(1).sum()),
        ("Missing deadline items", "orders", len(orders), orders["missing_deadline_items"].gt(0).sum()),
        ("Multiple non-null seller states", "orders", len(orders), orders["seller_states_non_null"].gt(1).sum()),
        ("Both equal and different state pairs", "orders", len(orders), (orders["items_with_equal_states"].gt(0) & orders["items_with_different_states"].gt(0)).sum()),
        ("Uncomparable state pairs", "orders", len(orders), orders["items_with_uncomparable_states"].gt(0).sum()),
        ("Incomplete item numbering", "orders", len(orders), (~orders["consecutive_item_numbers"]).sum()),
        ("Multiple deadlines within seller", "order-seller groups", len(deadlines), deadlines["distinct_deadlines"].gt(1).sum()),
    ]
    summary = pd.DataFrame([
        [check, unit, total, int(affected), 100 * int(affected) / total if total else float("nan")]
        for check, unit, total, affected in metrics
    ], columns=["check", "unit", "total", "affected", "affected_pct"])
    if len(orders) != len(views["orders"]):
        raise ValueError("Order counts changed during diagnostic grouping.")
    return {"summary": summary, "orders": orders, "seller_deadlines": deadlines}
