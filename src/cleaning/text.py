"""Source text checks and targeted city whitespace cleanup."""

import pandas as pd

from src.eda.profiling import require_columns
from src.pipeline.contracts.schema import ZIP_COLUMNS


CITY_COLUMNS = {
    "customer_city": "customer_id",
    "seller_city": "seller_id",
}

STRING_ID_COLUMNS = ("order_id", "product_id", "seller_id", "customer_id")
TEXT_PATTERNS = {
    **dict.fromkeys(STRING_ID_COLUMNS, r"[0-9a-f]{32}"),
    **dict.fromkeys(ZIP_COLUMNS, r"[0-9]{5}"),
}
# Fixed source domains, checked against the raw customers and orders tables.
# Keep 'created', which exists in raw orders even when absent from the item merge.
STATE_CODES = frozenset({
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
    "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
    "RS", "SC", "SE", "SP", "TO",
})
ORDER_STATUSES = frozenset({
    "approved", "canceled", "created", "delivered", "invoiced", "processing",
    "shipped", "unavailable",
})
TEXT_DOMAINS = {
    "customer_state": STATE_CODES,
    "seller_state": STATE_CODES,
    "order_status": ORDER_STATUSES,
}


def inspect_text_fields(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Profile source ID/ZIP formats and state/status domains without mutation.

    Require four string IDs to match 32 lowercase hexadecimal characters and
    ZIP prefixes to match five ASCII digits. Check exact state/status values
    against fixed source domains, not values inferred from the current input.
    Do not trim, change case, coerce non-string values or infer geography.

    Return an item-grain summary for all nine columns and all observed invalid
    cells in issues. Missing values are counted separately from invalid observed
    values; input validation enforces mandatory non-null IDs. Optional missing
    ZIP/state/status values remain available for subsequent completeness checks.
    Each issue includes zero-based positions, index labels, the item key and
    repr of the original value so duplicate indexes and whitespace are unambiguous.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    columns = [*TEXT_PATTERNS, *TEXT_DOMAINS]
    require_columns(df, [*columns, "order_item_id"])
    if df.columns.has_duplicates:
        raise ValueError("Text inspection requires unique column names.")

    summary_rows, issue_rows = [], []
    for column in columns:
        original = df[column]
        missing = original.isna()
        is_string = original.map(lambda value: isinstance(value, str)).astype(bool)
        strings = original.where(is_string, pd.NA).astype("string")
        if column in TEXT_PATTERNS:
            rule = TEXT_PATTERNS[column]
            valid = strings.str.fullmatch(rule, na=False) & is_string
        else:
            domain = TEXT_DOMAINS[column]
            rule = f"Exact member of {len(domain)} source " + (
                "statuses" if column == "order_status" else "state codes"
            )
            valid = strings.isin(domain) & is_string
        invalid = ~missing & ~valid
        summary_rows.append([
            column, rule, len(df), int(missing.sum()), int(valid.sum()),
            int(invalid.sum()),
        ])
        for position in invalid.to_numpy(dtype=bool).nonzero()[0]:
            row = df.iloc[position]
            issue_rows.append([
                column, rule, int(position), df.index[position], row["order_id"],
                row["order_item_id"], repr(original.iloc[position]),
            ])

    return {
        "summary": pd.DataFrame(summary_rows, columns=[
            "column", "rule", "item_rows", "missing_item_rows",
            "valid_item_rows", "invalid_item_rows",
        ]),
        "issues": pd.DataFrame(issue_rows, columns=[
            "column", "rule", "row_position", "index_label", "order_id",
            "order_item_id", "value_repr",
        ]),
    }


def normalize_city_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trim city edges and collapse repeated whitespace in a clean copy.

    Keep original case, spelling, nulls, row order, index, columns and dtypes.
    Return an independent item-level DataFrame and one change summary row per
    city column. Counts of edge/repeated whitespace may overlap.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    require_columns(df, [*CITY_COLUMNS, *CITY_COLUMNS.values()])

    for column in CITY_COLUMNS:
        if not pd.api.types.is_string_dtype(df[column].dtype):
            raise TypeError(f"Column '{column}' must have a string dtype.")

    cleaned = df.copy(deep=True)
    summary_rows = []
    for column, entity_key in CITY_COLUMNS.items():
        original = df[column]
        normalized = original.str.strip().str.replace(r"\s+", " ", regex=True)
        changed = original.ne(normalized).fillna(False)
        edge_whitespace = original.ne(original.str.strip()).fillna(False)
        repeated_whitespace = original.str.contains(r"\s{2,}", regex=True, na=False)
        cleaned[column] = normalized
        summary_rows.append([
            column,
            len(df),
            int(original.isna().sum()),
            int(edge_whitespace.sum()),
            int(repeated_whitespace.sum()),
            int(changed.sum()),
            int(df.loc[changed, entity_key].nunique()),
        ])

    summary = pd.DataFrame(summary_rows, columns=[
        "column", "item_rows", "missing_item_rows", "edge_whitespace_item_rows",
        "repeated_whitespace_item_rows", "changed_item_rows", "changed_entities",
    ])
    return cleaned, summary
