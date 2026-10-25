"""Targeted whitespace cleanup for customer and seller city fields."""

import pandas as pd

from src.eda.profiling import require_columns


CITY_COLUMNS = {
    "customer_city": "customer_id",
    "seller_city": "seller_id",
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
