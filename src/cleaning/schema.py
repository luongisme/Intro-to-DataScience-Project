"""Canonical contract for the cleaned item-level dataset."""

from src.cleaning.measurements import INVALID_PHYSICAL_FLAG
from src.cleaning.timeline import DEADLINE_REVIEW_FLAG, INVALID_TIMELINE_FLAG
from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA, TableSchema


CLEANED_DATASET_NAME = "cleaned_order_items"
QUALITY_FLAG_COLUMNS = (
    INVALID_PHYSICAL_FLAG, INVALID_TIMELINE_FLAG, DEADLINE_REVIEW_FLAG,
)
CLEANED_ORDER_ITEMS_SCHEMA = TableSchema(
    columns=[*FULL_MERGE_SCHEMA.columns, *QUALITY_FLAG_COLUMNS],
    dtypes={**FULL_MERGE_SCHEMA.dtypes, **dict.fromkeys(QUALITY_FLAG_COLUMNS, "bool")},
    unique_key=FULL_MERGE_SCHEMA.unique_key.copy(),
    non_nullable_columns=[*FULL_MERGE_SCHEMA.non_nullable_columns, *QUALITY_FLAG_COLUMNS],
)
