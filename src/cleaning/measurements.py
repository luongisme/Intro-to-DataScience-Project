"""Handle non-positive physical product measurements."""

import pandas as pd

from src.eda.profiling import require_columns


PHYSICAL_COLUMNS = (
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
)
INVALID_PHYSICAL_FLAG = "has_invalid_physical_measurement"


def clean_physical_measurements(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Set observed physical measurements <= 0 to missing in a clean copy.

    Preserve pre-existing missing values, all other fields, dtypes, index and
    row order. Add a boolean item-level flag for rows with at least one changed
    product measurement. Return per-measurement missing/change counts and
    affected product/order counts. Do not treat statistical outliers as invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    require_columns(df, ["order_id", "product_id", *PHYSICAL_COLUMNS])
    if INVALID_PHYSICAL_FLAG in df.columns:
        raise ValueError(f"Input already contains the reserved column '{INVALID_PHYSICAL_FLAG}'.")
    for column in PHYSICAL_COLUMNS:
        dtype = df[column].dtype
        if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            raise TypeError(f"Column '{column}' must contain numeric measurements.")

    cleaned = df.copy(deep=True)
    affected_items = pd.Series(False, index=df.index, dtype=bool)
    summary_rows = []
    for column in PHYSICAL_COLUMNS:
        original = df[column]
        nonpositive = original.le(0).fillna(False)
        cleaned[column] = original.mask(nonpositive, pd.NA)
        affected_items = affected_items | nonpositive
        summary_rows.append([
            column,
            len(df),
            int(original.isna().sum()),
            int(nonpositive.sum()),
            int(df.loc[nonpositive, "product_id"].nunique()),
            int(df.loc[nonpositive, "order_id"].nunique()),
            int(cleaned[column].isna().sum()),
        ])

    cleaned[INVALID_PHYSICAL_FLAG] = affected_items.to_numpy(dtype=bool)
    summary = pd.DataFrame(summary_rows, columns=[
        "measurement", "item_rows", "missing_item_rows_before",
        "nonpositive_item_rows", "affected_products", "affected_orders",
        "missing_item_rows_after",
    ])
    return cleaned, summary
