"""Data Contract and Schema specifications for the Olist Merge Pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

@dataclass(frozen=True)
class TableSchema:
    """Describe a DataFrame independently of its storage format.

    Attributes:
        columns: Expected column names in their required order.
        dtypes: Canonical pandas dtype for every column, including timestamps.
        unique_key: Columns forming one composite unique key, if specified.
        non_nullable_columns: Columns that must not contain missing values.

    This descriptor stores expectations; it does not validate or modify data.
    """

    columns: List[str]
    dtypes: Dict[str, str]
    unique_key: List[str] = field(default_factory=list)
    non_nullable_columns: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class JoinSpec:
    """Describe one join against a named right-hand table.

    Attributes:
        table: Registered name of the right-hand table.
        on: Shared join key or ordered list of shared join keys.
        left_on: Left-table key or ordered keys when their names differ.
        right_on: Right-table key or ordered keys matching left_on.
        how: Join type; left joins preserve the default pipeline behavior.
        validate: Expected key cardinality; m:m requires an explicit choice.
        columns: Right-table columns to include, or None for all columns.
            Join keys are retained even when not listed here.
        rename: New names for selected non-key right-table columns.
        require_match: Whether every left-side key must exist on the right.

    Specify either on or both left_on and right_on using original key names.
    This descriptor stores configuration; join validation and execution are
    separate operations.
    """

    table: str
    on: Optional[Union[str, List[str]]] = None
    left_on: Optional[Union[str, List[str]]] = None
    right_on: Optional[Union[str, List[str]]] = None
    how: Literal["left", "inner", "right", "outer"] = "left"
    validate: Literal["1:1", "1:m", "m:1", "m:m"] = "m:1"
    columns: Optional[List[str]] = None
    rename: Dict[str, str] = field(default_factory=dict)
    require_match: bool = True


@dataclass(frozen=True, eq=False)
class MergeResult:
    """Carry merged data with its schema and merge report.

    Attributes:
        data: Merged DataFrame, retained by reference without copying.
        schema: Expected structure and integrity constraints for the data.
        report: Merge steps and validation results collected by the pipeline.

    This container does not validate data, write files, or build a report.
    DataFrame contents remain mutable. Compare their values explicitly;
    result instances use identity equality to avoid ambiguous frame comparisons.
    """

    data: pd.DataFrame
    schema: TableSchema
    report: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TableLoadSpec:
    """Specification for loading a raw Olist table."""

    dataset_name: str
    usecols: List[str]
    dtypes: Dict[str, str] = field(default_factory=dict)
    date_columns: List[str] = field(default_factory=list)


# 1. Order Items (Base Table)
ORDER_ITEMS_SPEC = TableLoadSpec(
    dataset_name="order_items",
    usecols=[
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value",
    ],
    dtypes={
        "order_id": "string",
        "order_item_id": "int64",
        "product_id": "string",
        "seller_id": "string",
        "price": "float64",
        "freight_value": "float64",
    },
    date_columns=["shipping_limit_date"],
)

# 2. Orders Table
ORDERS_SPEC = TableLoadSpec(
    dataset_name="orders",
    usecols=[
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    dtypes={
        "order_id": "string",
        "customer_id": "string",
        "order_status": "string",
    },
    date_columns=[
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
)

# 3. Customers Table
CUSTOMERS_SPEC = TableLoadSpec(
    dataset_name="customers",
    usecols=[
        "customer_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
    dtypes={
        "customer_id": "string",
        "customer_zip_code_prefix": "string",
        "customer_city": "string",
        "customer_state": "string",
    },
    date_columns=[],
)

# 4. Products Table
PRODUCTS_SPEC = TableLoadSpec(
    dataset_name="products",
    usecols=[
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ],
    dtypes={
        "product_id": "string",
        "product_category_name": "string",
        "product_weight_g": "float64",
        "product_length_cm": "float64",
        "product_height_cm": "float64",
        "product_width_cm": "float64",
    },
    date_columns=[],
)

# 5. Sellers Table
SELLERS_SPEC = TableLoadSpec(
    dataset_name="sellers",
    usecols=[
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state",
    ],
    dtypes={
        "seller_id": "string",
        "seller_zip_code_prefix": "string",
        "seller_city": "string",
        "seller_state": "string",
    },
    date_columns=[],
)

# Exact 25 columns required in final output in order
EXPECTED_FINAL_COLUMNS: List[str] = [
    "order_id",
    "order_item_id",
    "product_id",
    "seller_id",
    "shipping_limit_date",
    "price",
    "freight_value",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state",
    "product_category_name",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state",
]

# Validation contract constants
MANDATORY_NON_NULL_COLUMNS: List[str] = [
    "order_id",
    "order_item_id",
    "product_id",
    "seller_id",
    "customer_id",
]

ALL_TIMESTAMP_COLUMNS: List[str] = [
    "shipping_limit_date",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

ZIP_COLUMNS: List[str] = [
    "customer_zip_code_prefix",
    "seller_zip_code_prefix",
]

BENCHMARK_ROW_COUNT: int = 112650