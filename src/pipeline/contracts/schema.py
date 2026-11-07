"""Data Contract and Schema specifications for the Olist Merge Pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

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

# Each list defines one primary key for a raw source table.
# Multiple columns form a composite key; they are not individually unique.
SOURCE_PRIMARY_KEYS: Dict[str, List[str]] = {
    ORDER_ITEMS_SPEC.dataset_name: ["order_id", "order_item_id"],
    ORDERS_SPEC.dataset_name: ["order_id"],
    CUSTOMERS_SPEC.dataset_name: ["customer_id"],
    PRODUCTS_SPEC.dataset_name: ["product_id"],
    SELLERS_SPEC.dataset_name: ["seller_id"],
}


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

# Reuse source declarations for the complete item-level output contract.
# Timestamp precision is explicit so both storage formats can restore it.
FULL_MERGE_SCHEMA = TableSchema(
    columns=EXPECTED_FINAL_COLUMNS.copy(),
    dtypes={
        **ORDER_ITEMS_SPEC.dtypes,
        **ORDERS_SPEC.dtypes,
        **CUSTOMERS_SPEC.dtypes,
        **PRODUCTS_SPEC.dtypes,
        **SELLERS_SPEC.dtypes,
        **dict.fromkeys(ALL_TIMESTAMP_COLUMNS, "datetime64[us]"),
    },
    unique_key=SOURCE_PRIMARY_KEYS[ORDER_ITEMS_SPEC.dataset_name].copy(),
    non_nullable_columns=MANDATORY_NON_NULL_COLUMNS.copy(),
)


def build_source_schema(
    spec: TableLoadSpec,
    *,
    key_columns: Optional[Sequence[str]] = None,
) -> TableSchema:
    """Create an independent source contract from raw loading declarations.

    Preserve declared column order and dtypes. Columns in date_columns use
    datetime64[us] and must not also appear in dtypes. Every selected column
    needs exactly one dtype declaration. Use SOURCE_PRIMARY_KEYS unless an
    explicit key is supplied; other source names require key_columns.

    Declare primary-key components as non-null. Other attributes, including
    foreign keys, gain constraints through separate join validation rather
    than assumptions about their names. The returned schema has independent
    lists and mappings. Do not load data, inspect values or validate that
    actual rows satisfy this contract.

    Args:
        spec: Source name, selected columns and raw dtype declarations.
        key_columns: Optional primary-key override for this source contract.

    Raises:
        TypeError: If spec or its dtype mapping has an unexpected type.
        ValueError: If columns, dtype declarations or key declarations are invalid.
    """
    if not isinstance(spec, TableLoadSpec):
        raise TypeError("build_source_schema requires a TableLoadSpec.")
    if not isinstance(spec.dataset_name, str) or not spec.dataset_name.strip():
        raise ValueError("TableLoadSpec.dataset_name must be a non-empty string.")
    if isinstance(spec.usecols, str) or not isinstance(spec.usecols, Sequence):
        raise ValueError("TableLoadSpec.usecols must be a sequence of column names.")
    columns = list(spec.usecols)
    if not columns or not all(isinstance(col, str) and col.strip() for col in columns):
        raise ValueError("TableLoadSpec.usecols requires non-empty string names.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Load specification for '{spec.dataset_name}' repeats a column.")
    if isinstance(spec.date_columns, str) or not isinstance(spec.date_columns, Sequence):
        raise ValueError("TableLoadSpec.date_columns must be a sequence of names.")
    dates = list(spec.date_columns)
    if not all(isinstance(col, str) and col.strip() for col in dates):
        raise ValueError("TableLoadSpec.date_columns requires non-empty string names.")
    if len(dates) != len(set(dates)):
        raise ValueError(f"Load specification for '{spec.dataset_name}' repeats a date column.")
    if not isinstance(spec.dtypes, dict):
        raise TypeError("TableLoadSpec.dtypes must be a dictionary.")
    overlap = [col for col in dates if col in spec.dtypes]
    if overlap:
        raise ValueError(
            f"Table '{spec.dataset_name}' declares dtypes for date_columns: {overlap}. "
            "Declare timestamps only in date_columns."
        )
    dtypes = {**spec.dtypes, **dict.fromkeys(dates, "datetime64[us]")}
    missing_dtypes = [col for col in columns if col not in dtypes]
    extra_dtypes = [col for col in dtypes if col not in columns]
    if missing_dtypes or extra_dtypes:
        raise ValueError(
            f"Table '{spec.dataset_name}' has inconsistent load dtype declarations: "
            f"missing={missing_dtypes}, extra={extra_dtypes}."
        )
    for column, dtype in dtypes.items():
        if not isinstance(dtype, str) or not dtype.strip():
            raise ValueError(
                f"Table '{spec.dataset_name}' requires a dtype string for '{column}'."
            )
        try:
            pd.api.types.pandas_dtype(dtype)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Table '{spec.dataset_name}' declares an invalid dtype for "
                f"'{column}': {dtype!r}."
            ) from err

    declared_keys = key_columns
    if declared_keys is None:
        if spec.dataset_name not in SOURCE_PRIMARY_KEYS:
            raise ValueError(
                f"No primary key is registered for table '{spec.dataset_name}'. "
                "Supply key_columns explicitly for a custom source contract."
            )
        declared_keys = SOURCE_PRIMARY_KEYS[spec.dataset_name]
    if isinstance(declared_keys, str) or not isinstance(declared_keys, Sequence):
        raise ValueError("Primary key must be a non-empty sequence of column names.")
    keys = list(declared_keys)
    if not keys or not all(isinstance(key, str) and key.strip() for key in keys):
        raise ValueError("Primary key requires non-empty string column names.")
    if len(keys) != len(set(keys)):
        raise ValueError(f"Primary key for '{spec.dataset_name}' repeats a column name.")
    missing_keys = [key for key in keys if key not in columns]
    if missing_keys:
        raise ValueError(
            f"Source schema for '{spec.dataset_name}' is missing primary key columns: "
            f"{missing_keys}."
        )
    return TableSchema(
        columns=columns,
        dtypes=dtypes,
        unique_key=keys,
        non_nullable_columns=keys.copy(),
    )
