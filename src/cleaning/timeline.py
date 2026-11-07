"""Flag core timestamp reversals and item deadlines needing review."""

import pandas as pd

from src.eda.checks import build_entity_views, profile_timestamp_ordering
from src.eda.profiling import require_columns
from src.pipeline.contracts.validator import validate_source_keys


CORE_TIMESTAMP_COLUMNS = (
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
)
ADJACENT_TIMESTAMP_CHECKS = (
    "approved_before_purchase",
    "carrier_before_approval",
    "customer_before_carrier",
)
INVALID_TIMELINE_FLAG = "has_invalid_timeline"
DEADLINE_REVIEW_FLAG = "needs_deadline_review"


def flag_timeline_issues(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag observed adjacent core timestamp reversals without changing dates.

    Build validated order-grain views, evaluate the three adjacent timestamp
    comparisons and retain missing comparisons as unknown in the summary.
    A missing core timestamp is reported separately and does not by itself set
    the chronology flag. Return a clean item-level copy, the per-check order
    summary and the core-timestamp completeness summary.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    require_columns(df, ["order_id", *CORE_TIMESTAMP_COLUMNS])
    if INVALID_TIMELINE_FLAG in df.columns:
        raise ValueError(f"Input already contains the reserved column '{INVALID_TIMELINE_FLAG}'.")

    orders = build_entity_views(df)["orders"]
    profile = profile_timestamp_ordering(orders)
    conditions = profile["conditions"].loc[:, list(ADJACENT_TIMESTAMP_CHECKS)]
    order_summary = profile["summary"].set_index("check")
    precision_rows = []
    for check in ADJACENT_TIMESTAMP_CHECKS:
        row = order_summary.loc[check]
        flagged = conditions[check].fillna(False)
        start = orders[row["start_column"]]
        end = orders[row["end_column"]]
        same_day = flagged & start.dt.normalize().eq(end.dt.normalize()).fillna(False)
        earlier_date = flagged & end.dt.normalize().lt(start.dt.normalize()).fillna(False)
        precision_rows.append([
            check,
            int(row["orders"]),
            int(row["evaluated_orders"]),
            int(row["unassessable_orders"]),
            int(row["flagged_orders"]),
            int(same_day.sum()),
            int(earlier_date.sum()),
        ])

    timeline_summary = pd.DataFrame(precision_rows, columns=[
        "check", "orders", "comparable_orders", "unknown_orders",
        "affected_orders", "same_day_reversals", "earlier_date_reversals",
    ])
    missing_core = orders.loc[:, list(CORE_TIMESTAMP_COLUMNS)].isna().any(axis=1)
    core_summary = pd.DataFrame([{
        "orders": len(orders),
        "orders_missing_any_core_timestamp": int(missing_core.sum()),
        "orders_with_all_core_timestamps": int((~missing_core).sum()),
    }])

    affected_order_ids = orders.loc[conditions.fillna(False).any(axis=1), "order_id"]
    item_flag = df["order_id"].isin(affected_order_ids).to_numpy(dtype=bool)
    cleaned = df.copy(deep=True)
    cleaned[INVALID_TIMELINE_FLAG] = item_flag
    return cleaned, timeline_summary, core_summary


def flag_deadline_issues(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag item deadlines needing review while preserving original timestamps.

    Check shipping_limit_date < order_purchase_timestamp at full precision and
    deadline year > the latest observed purchase year in the supplied table.
    Run on the full cleaning input so the latter reference remains comparable.
    A later year is a review signal, not an SLA limit or proof of an invalid
    deadline. Never replace dates, remove items or select an order-level deadline.

    Return a clean copy with a bool needs_deadline_review flag, a two-check
    item-grain summary and all flagged original item records with nullable
    per-check conditions. Missing endpoints are unknown. If all purchases are
    missing, the year comparison is unknown for every item. The combined flag
    means at least one observed issue; False does not imply complete timestamps.
    Count affected orders/order-seller groups separately for each check.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    record_columns = [
        "order_id", "order_item_id", "seller_id", "order_status",
        "order_purchase_timestamp", "shipping_limit_date",
    ]
    require_columns(df, record_columns)
    if DEADLINE_REVIEW_FLAG in df.columns:
        raise ValueError(f"Input already contains the reserved column '{DEADLINE_REVIEW_FLAG}'.")
    validate_source_keys(df, "order_items")
    for column in ("order_purchase_timestamp", "shipping_limit_date"):
        if not pd.api.types.is_datetime64_any_dtype(df[column].dtype):
            raise TypeError(f"Column '{column}' must have a datetime dtype.")

    purchase = df["order_purchase_timestamp"]
    deadline = df["shipping_limit_date"]
    comparable = purchase.notna() & deadline.notna()
    before_purchase = deadline.lt(purchase).astype("boolean").where(comparable, pd.NA)
    latest_purchase = purchase.max()
    if pd.notna(latest_purchase):
        reference = f"Latest observed purchase year: {latest_purchase.year}"
        after_purchase_year = deadline.dt.year.gt(latest_purchase.year).astype("boolean").where(
            deadline.notna(), pd.NA,
        )
    else:
        reference = "Unavailable: no observed purchase timestamp"
        after_purchase_year = pd.Series(pd.NA, index=df.index, dtype="boolean")
    conditions = pd.DataFrame({
        "deadline_before_purchase": before_purchase,
        "deadline_after_purchase_year": after_purchase_year,
    }, index=df.index)

    summary_rows = []
    for check in conditions:
        condition = conditions[check]
        affected = df.loc[condition.fillna(False)]
        summary_rows.append([
            check,
            "Item purchase timestamp" if check == "deadline_before_purchase" else reference,
            len(df), int(condition.notna().sum()), int(condition.isna().sum()),
            len(affected), int(affected["order_id"].nunique()),
            len(affected[["order_id", "seller_id"]].drop_duplicates()),
        ])
    summary = pd.DataFrame(summary_rows, columns=[
        "check", "reference", "item_rows", "comparable_item_rows",
        "unknown_item_rows", "affected_item_rows", "affected_orders",
        "affected_order_seller_groups",
    ])
    flagged = conditions.fillna(False).any(axis=1)
    cleaned = df.copy(deep=True)
    cleaned[DEADLINE_REVIEW_FLAG] = flagged.to_numpy(dtype=bool)
    records = df.loc[flagged, record_columns].copy(deep=True)
    for check in conditions:
        records[check] = conditions.loc[flagged, check].array
    return cleaned, summary, records

