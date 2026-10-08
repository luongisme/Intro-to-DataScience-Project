"""Data Contract and Schema specifications for the Olist Merge Pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List


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