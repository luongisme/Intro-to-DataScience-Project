"""Contracts for inspected and persisted order-item and delivery data."""

from src.cleaning.measurements import INVALID_PHYSICAL_FLAG
from src.cleaning.timeline import (
    CORE_TIMESTAMP_COLUMNS,
    DEADLINE_REVIEW_FLAG,
    INVALID_TIMELINE_FLAG,
)
from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA, TableSchema


CLEANED_DATASET_NAME = "cleaned_order_items"
REQUIRED_TIMESTAMPS = (*CORE_TIMESTAMP_COLUMNS, "order_estimated_delivery_date")

# These two columns occur in an older processed Parquet file. They are not part
# of the current cleaning output, so the loader strips them on read instead of
# changing the canonical 25-column cleaning contract.
LEGACY_CLEANED_ORDER_ITEMS_EXTRA_COLUMNS = (
    "missing_category",
    "missing_product_features",
)

QUALITY_FLAG_COLUMNS = (
    INVALID_PHYSICAL_FLAG,
    INVALID_TIMELINE_FLAG,
    DEADLINE_REVIEW_FLAG,
)

FLAGGED_ORDER_ITEMS_SCHEMA = TableSchema(
    columns=[*FULL_MERGE_SCHEMA.columns, *QUALITY_FLAG_COLUMNS],
    dtypes={
        **FULL_MERGE_SCHEMA.dtypes,
        **dict.fromkeys(QUALITY_FLAG_COLUMNS, "bool"),
    },
    unique_key=FULL_MERGE_SCHEMA.unique_key.copy(),
    non_nullable_columns=[
        *FULL_MERGE_SCHEMA.non_nullable_columns,
        *QUALITY_FLAG_COLUMNS,
    ],
)

# Persist only eligible item rows and original data columns.
CLEANED_ORDER_ITEMS_SCHEMA = TableSchema(
    columns=FULL_MERGE_SCHEMA.columns.copy(),
    dtypes=FULL_MERGE_SCHEMA.dtypes.copy(),
    unique_key=FULL_MERGE_SCHEMA.unique_key.copy(),
    non_nullable_columns=[
        *FULL_MERGE_SCHEMA.non_nullable_columns,
        "order_status",
        *REQUIRED_TIMESTAMPS,
    ],
)
LEGACY_CLEANED_ORDER_ITEMS_NULLABLE_COLUMNS = (
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
)

LEGACY_CLEANED_ORDER_ITEMS_SCHEMA = TableSchema(
    columns=CLEANED_ORDER_ITEMS_SCHEMA.columns.copy(),
    dtypes=CLEANED_ORDER_ITEMS_SCHEMA.dtypes.copy(),
    unique_key=CLEANED_ORDER_ITEMS_SCHEMA.unique_key.copy(),
    non_nullable_columns=[
        column
        for column in CLEANED_ORDER_ITEMS_SCHEMA.non_nullable_columns
        if column not in LEGACY_CLEANED_ORDER_ITEMS_NULLABLE_COLUMNS
    ],

)


# Persisted delivery features: one row per order (not one row per order item).
# Column order and types are based on the existing late_delivery_orders.parquet.
LATE_DELIVERY_ORDERS_NAME = "late_delivery_orders"
LATE_DELIVERY_ORDERS_SCHEMA = TableSchema(
    columns=[
        "order_id",
        "order_status",
        "customer_id",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "customer_city",
        "customer_state",
        "order_value",
        "freight_value",
        "item_count",
        "carrier_before_approval",
        "delivery_days",
        "delay_days",
        "is_late",
    ],
    dtypes={
        "order_id": "string",
        "order_status": "string",
        "customer_id": "string",
        "order_purchase_timestamp": "datetime64[us]",
        "order_approved_at": "datetime64[us]",
        "order_delivered_carrier_date": "datetime64[us]",
        "order_delivered_customer_date": "datetime64[us]",
        "order_estimated_delivery_date": "datetime64[us]",
        "customer_city": "string",
        "customer_state": "string",
        "order_value": "float64",
        "freight_value": "float64",
        "item_count": "int64",
        "carrier_before_approval": "bool",
        "delivery_days": "float64",
        "delay_days": "float64",
        "is_late": "bool",
    },
    unique_key=["order_id"],
    non_nullable_columns=["order_id"],
)

ANOMALY_ORDERS_NAME = "anomaly_orders"

ANOMALY_NEG_FLAG_COLUMNS = [
    "neg_approved_before_purchase",
    "neg_carrier_before_approval",
    "neg_customer_before_carrier",
    "neg_carrier_before_purchase",
    "neg_customer_before_approval",
    "neg_customer_before_purchase",
]

ANOMALY_NULLABLE_COLUMNS = [
    "order_approved_at",
    "order_delivered_carrier_date",
    *ANOMALY_NEG_FLAG_COLUMNS,
]

ANOMALY_ORDERS_SCHEMA = TableSchema(
    columns=[
        *LATE_DELIVERY_ORDERS_SCHEMA.columns,
        "drop_reason",
        *ANOMALY_NEG_FLAG_COLUMNS,
        "flag_carrier_before_purchase",
    ],
    dtypes={
        **LATE_DELIVERY_ORDERS_SCHEMA.dtypes,
        "drop_reason": "string",
        **dict.fromkeys(ANOMALY_NEG_FLAG_COLUMNS, "boolean"),
        "flag_carrier_before_purchase": "int64",
    },
    unique_key=["order_id"],
    non_nullable_columns=[
        column
        for column in [
            *LATE_DELIVERY_ORDERS_SCHEMA.columns,
            "drop_reason",
            *ANOMALY_NEG_FLAG_COLUMNS,
            "flag_carrier_before_purchase",
        ]
        if column not in ANOMALY_NULLABLE_COLUMNS
    ],
)


ORDER_ANALYSIS_NAME = "order_analysis"

ORDER_ANALYSIS_EXTRA_DTYPES = {
    "seller_id": "string",
    "seller_state": "string",
    "n_sellers": "int64",
    "product_category": "string",
    "total_weight_g": "float64",
    "shipping_limit_date": "datetime64[us]",
    "approve_hour": "int32",
    "approve_slot": "string",
    "hours_type": "string",
    "weekday": "string",
    "day_type": "string",
    "month": "string",
    "season": "string",
    "is_peak": "int64",
    "purchase_hour": "int32",
    "purchase_slot": "string",
    "carrier_hour": "int32",
    "carrier_slot": "string",
    "carrier_ts_regime": "string",
    "approval_days": "float64",
    "processing_lead_time": "float64",
    "carrier_transit_time": "float64",
    "lead_days": "float64",
    "estimated_lead_days": "float64",
    "seller_delay": "int64",
    "same_state": "int64",
    "route_type": "string",
    "route": "string",
    "is_multi_seller": "int64",
    "is_bulky": "int64",
    "is_remote": "int64",
    "slow_approval": "int64",
}

ORDER_ANALYSIS_DTYPES = {
    **LATE_DELIVERY_ORDERS_SCHEMA.dtypes,
    "delay_days": "int64",
    "is_late": "int64",
    **ORDER_ANALYSIS_EXTRA_DTYPES,
}

ORDER_ANALYSIS_COLUMNS = [
    *LATE_DELIVERY_ORDERS_SCHEMA.columns,
    *ORDER_ANALYSIS_EXTRA_DTYPES,
]

ORDER_ANALYSIS_NULLABLE_COLUMNS = [
    "product_category",
    "total_weight_g",
]

ORDER_ANALYSIS_SCHEMA = TableSchema(
    columns=ORDER_ANALYSIS_COLUMNS,
    dtypes=ORDER_ANALYSIS_DTYPES,
    unique_key=["order_id"],
    non_nullable_columns=[
        column
        for column in ORDER_ANALYSIS_COLUMNS
        if column not in ORDER_ANALYSIS_NULLABLE_COLUMNS
    ],
)