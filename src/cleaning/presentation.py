"""Presentation-only formatting for cleaning tables and previews."""

from numbers import Real

import pandas as pd


COLUMN_LABELS = {
    "order_id": "Order ID",
    "order_item_id": "Order Item Number",
    "product_id": "Product ID",
    "seller_id": "Seller ID",
    "customer_id": "Customer ID",
    "order_status": "Order Status",
    "order_purchase_timestamp": "Purchase Timestamp",
    "order_approved_at": "Payment Approved At",
    "order_delivered_carrier_date": "Carrier Handoff At",
    "order_delivered_customer_date": "Customer Delivered At",
    "order_estimated_delivery_date": "Estimated Delivery Date",
    "shipping_limit_date": "Shipping Deadline",
    "customer_zip_code_prefix": "Customer ZIP Prefix",
    "seller_zip_code_prefix": "Seller ZIP Prefix",
    "customer_city": "Customer City",
    "seller_city": "Seller City",
    "customer_state": "Customer State",
    "seller_state": "Seller State",
    "product_category_name": "Product Category",
    "price": "Item Price (BRL)",
    "freight_value": "Freight Value (BRL)",
    "product_weight_g": "Product Weight (g)",
    "product_length_cm": "Product Length (cm)",
    "product_height_cm": "Product Height (cm)",
    "product_width_cm": "Product Width (cm)",
    "has_invalid_physical_measurement": "Invalid Physical Measurement",
    "has_invalid_timeline": "Invalid Core Timestamp Order",
    "metric": "Metric",
    "reference": "Reference",
    "observed": "Observed",
    "matches reference": "Matches Reference",
    "column": "Column",
    "measurement": "Measurement",
    "item_rows": "Item Rows",
    "rows": "Rows",
    "missing_item_rows": "Missing Item Rows",
    "missing_item_rows_before": "Missing Item Rows Before",
    "missing_item_rows_after": "Missing Item Rows After",
    "edge_whitespace_item_rows": "Rows With Edge Whitespace",
    "repeated_whitespace_item_rows": "Rows With Repeated Whitespace",
    "changed_item_rows": "Changed Item Rows",
    "changed_entities": "Changed Entities",
    "nonpositive_item_rows": "Nonpositive Item Rows",
    "affected_products": "Affected Products",
    "affected_orders": "Affected Orders",
    "check": "Timestamp Check",
    "orders": "Orders",
    "comparable_orders": "Comparable Orders",
    "unknown_orders": "Unassessable Orders",
    "same_day_reversals": "Same-Day Reversals",
    "earlier_date_reversals": "Earlier-Date Reversals",
    "orders_missing_any_core_timestamp": "Orders Missing a Core Timestamp",
    "orders_with_all_core_timestamps": "Orders With All Core Timestamps",
    "missing_rows": "Missing Rows",
    "blank_rows": "Blank Rows",
    "invalid_rows": "Invalid-Key Rows",
    "valid_rows": "Valid-Key Rows",
    "distinct_valid_keys": "Distinct Valid Keys",
    "repeated_key_groups": "Repeated-Key Groups",
    "rows_in_repeated_keys": "Rows in Repeated-Key Groups",
    "excess_key_rows": "Excess Key Rows",
    "valid_entities": "Valid Entities",
    "conflicted_entities": "Conflicted Entities",
    "conflict_pct": "Conflicted Entities (%)",
    "item_rows_in_conflicted_entities": "Items in Conflicted Entities",
    "missing_pct": "Missing (%)",
    "blank_pct": "Blank (%)",
    "pct_all": "All Rows (%)",
    "affected_pct_evaluated": "Affected Comparable Rows (%)",
    "affected_pct_all": "Affected All Rows (%)",
    "flagged_pct_evaluated": "Flagged Comparable Orders (%)",
    "outside_iqr_pct": "Outside IQR (%)",
}

DISPLAY_ORDER = (
    "order_id", "order_item_id", "product_id", "seller_id", "customer_id",
    "order_status", "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date", "shipping_limit_date",
    "product_category_name", "product_weight_g", "product_length_cm",
    "product_height_cm", "product_width_cm", "price", "freight_value",
    "customer_zip_code_prefix", "customer_city", "customer_state",
    "seller_zip_code_prefix", "seller_city", "seller_state",
    "has_invalid_physical_measurement", "has_invalid_timeline",
)
CURRENCY_COLUMNS = {"price", "freight_value"}
DATETIME_COLUMNS = {
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date", "shipping_limit_date",
}
PERCENT_COLUMNS = {
    "missing_pct", "blank_pct", "pct_all", "affected_pct_evaluated",
    "affected_pct_all", "flagged_pct_evaluated", "outside_iqr_pct", "conflict_pct",
}


def _format_number(value: Real) -> str:
    number = float(value)
    if number.is_integer():
        return f"{number:,.0f}"
    return f"{number:,.15g}"


def _format_city(value: str) -> str:
    small_words = {"da", "das", "de", "do", "dos", "e"}
    words = value.split()
    return " ".join(
        word.lower() if position and word.casefold() in small_words else word.capitalize()
        for position, word in enumerate(words)
    )


def _format_value(value, column: str) -> str:
    """Format one scalar for display without changing its source value."""
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return "Missing"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, pd.Timestamp):
        if column == "order_estimated_delivery_date" and value == value.normalize():
            return value.strftime("%Y-%m-%d")
        if value.microsecond or value.nanosecond:
            return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, Real):
        if column in CURRENCY_COLUMNS:
            return f"R$ {float(value):,.2f}"
        if column in PERCENT_COLUMNS:
            return f"{float(value):,.2f}%"
        return _format_number(value)
    if isinstance(value, str):
        if column == "order_status":
            return value.title()
        if column == "product_category_name":
            return value.replace("_", " ").capitalize()
        if column in {"customer_city", "seller_city"}:
            return _format_city(value)
        if column == "check":
            return value.replace("_", " ").title()
        return value
    if isinstance(value, (tuple, list)):
        return ", ".join(str(part) for part in value)
    return str(value)


def format_cleaning_preview(
    df: pd.DataFrame,
    *,
    rows: int = 10,
) -> pd.DataFrame:
    """Return a readable, bounded display copy with labeled columns and values.

    Keep input values and dtypes untouched. The returned preview is display-only:
    its cell values are strings, nulls appear as 'Missing', monetary values show
    BRL to two decimals, timestamps use explicit precision, and known category
    labels/flags use readable text. City casing is improved for display without
    inventing diacritics. Preserve ZIP prefixes and IDs as displayed.
    Reorder full-table columns by identifier, status/time, product, amount,
    geography and quality-flag groups; keep diagnostic table column order.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.columns.has_duplicates:
        raise ValueError("Display formatting requires unique column names.")
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise TypeError("rows must be a positive integer.")
    if rows < 1:
        raise ValueError("rows must be a positive integer.")

    columns = list(df.columns)
    known = [column for column in DISPLAY_ORDER if column in columns]
    ordered = known + [column for column in columns if column not in DISPLAY_ORDER]
    preview = df.loc[:, ordered].head(rows).copy(deep=True)
    formatted = pd.DataFrame(index=range(len(preview)))
    for column in ordered:
        formatted[COLUMN_LABELS.get(column, column.replace("_", " ").title())] = (
            preview[column].map(lambda value: _format_value(value, column)).astype("string").array
        )
    return formatted
