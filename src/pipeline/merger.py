"""Merge logic for combining Olist tables into a single item-level DataFrame."""

import logging
from typing import List, Set
import pandas as pd

from src.pipeline.schema import EXPECTED_FINAL_COLUMNS

logger = logging.getLogger(__name__)


class OlistDataMerger:
    """Executes sequential many-to-one left joins starting from order_items."""

    def __init__(self) -> None:
        self.expected_columns: List[str] = EXPECTED_FINAL_COLUMNS

    @staticmethod
    def _safe_left_join(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        on: str,
        table_name: str,
    ) -> pd.DataFrame:
        """Perform a validated many-to-one left join without column collision.

        Args:
            left_df: Base DataFrame.
            right_df: Dimension table to join.
            on: Common key column.
            table_name: Name of the right table for error logging.

        Returns:
            pd.DataFrame: Merged DataFrame.

        Raises:
            ValueError: If non-key column names collide or if right table contains duplicate keys.
        """
        # Guard against suffix collisions (_x, _y)
        overlapping_cols: Set[str] = (set(left_df.columns) & set(right_df.columns)) - {on}
        if overlapping_cols:
            raise ValueError(
                f"Column name collision detected when joining '{table_name}' on key '{on}'. "
                f"Colliding columns: {sorted(overlapping_cols)}. "
                "Adding suffixes (_x, _y) is strictly forbidden."
            )

        merged = left_df.merge(right_df, on=on, how="left", validate="m:1")
        return merged