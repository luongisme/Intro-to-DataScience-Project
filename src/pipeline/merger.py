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

    def merge_tables(
        self,
        order_items: pd.DataFrame,
        orders: pd.DataFrame,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        sellers: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all 5 Olist tables strictly in left-join order.

        Join sequence:
        1. order_items LEFT JOIN orders on order_id
        2. + LEFT JOIN customers on customer_id
        3. + LEFT JOIN products on product_id
        4. + LEFT JOIN sellers on seller_id

        Args:
            order_items: Base item table.
            orders: Orders table.
            customers: Customers table.
            products: Products table.
            sellers: Sellers table.

        Returns:
            pd.DataFrame: New merged DataFrame matching EXPECTED_FINAL_COLUMNS.
        """
        # Immutability guarantee: clone input base table
        current_df = order_items.copy(deep=True)
        initial_rows = len(current_df)
        logger.info("Starting merge pipeline. Base table rows: %d", initial_rows)

        # 1. Join orders on order_id
        current_df = self._safe_left_join(
            left_df=current_df,
            right_df=orders,
            on="order_id",
            table_name="orders",
        )
        logger.info("Joined 'orders'. Current shape: %s", current_df.shape)

        # 2. Join customers on customer_id
        current_df = self._safe_left_join(
            left_df=current_df,
            right_df=customers,
            on="customer_id",
            table_name="customers",
        )
        logger.info("Joined 'customers'. Current shape: %s", current_df.shape)

        # 3. Join products on product_id
        current_df = self._safe_left_join(
            left_df=current_df,
            right_df=products,
            on="product_id",
            table_name="products",
        )
        logger.info("Joined 'products'. Current shape: %s", current_df.shape)

        # 4. Join sellers on seller_id
        current_df = self._safe_left_join(
            left_df=current_df,
            right_df=sellers,
            on="seller_id",
            table_name="sellers",
        )
        logger.info("Joined 'sellers'. Current shape: %s", current_df.shape)

        # Ensure column ordering matches specification exactly
        result = current_df[self.expected_columns].copy(deep=True)
        logger.info("Merge completed successfully. Output shape: %s", result.shape)
        return result