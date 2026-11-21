"""Derive approval-time and delivery outcomes at order grain."""

import pandas as pd


ORDER_FIELDS = (
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
)
SLOT_LABELS = ("00:00–06:00", "06:00–12:00", "12:00–18:00", "18:00–24:00")


def build_order_analysis_view(items: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Derive order metrics while retaining conflicting orders for review.

    The processed loader provides the item-key and dtype contract. A missing or
    conflicting order-level value quarantines every item of that order. Valid
    orders remain analyzable, and the original item table is unchanged.
    """
    required = ("order_id", *ORDER_FIELDS)
    missing_columns = set(required).difference(items.columns)
    if missing_columns:
        raise ValueError(f"Missing analysis columns: {sorted(missing_columns)}")

    groups = items.groupby("order_id", sort=False, dropna=False, observed=True)
    distinct = groups[list(ORDER_FIELDS)].nunique(dropna=False)
    missing = items.loc[:, list(ORDER_FIELDS)].isna().groupby(
        items["order_id"], sort=False, dropna=False, observed=True
    ).any()
    reasons = []
    for field in ORDER_FIELDS:
        for order_id in distinct.index[distinct[field].gt(1)]:
            reasons.append((order_id, field, "conflicting_values"))
        for order_id in missing.index[missing[field]]:
            reasons.append((order_id, field, "missing_value"))
    review_reasons = pd.DataFrame(reasons, columns=["order_id", "field", "issue"])
    reviewed_ids = pd.Index(review_reasons["order_id"].unique())
    review_items = items.loc[items["order_id"].isin(reviewed_ids)].copy()
    valid_items = items.loc[~items["order_id"].isin(reviewed_ids)]
    orders = valid_items.loc[:, list(required)].drop_duplicates("order_id").copy()
    orders.reset_index(drop=True, inplace=True)

    approved = orders["order_approved_at"]
    delivery_date = orders["order_delivered_customer_date"].dt.normalize()
    estimated_date = orders["order_estimated_delivery_date"].dt.normalize()
    days_difference = (delivery_date - estimated_date).dt.days
    orders["approval_hour"] = approved.dt.hour
    orders["approval_weekday"] = approved.dt.dayofweek
    orders["approval_slot"] = pd.Categorical.from_codes(
        orders["approval_hour"] // 6, categories=SLOT_LABELS, ordered=True
    )
    orders["is_late"] = days_difference.gt(0)
    orders["delay_days"] = days_difference.where(orders["is_late"]).astype("Int64")

    summary = pd.DataFrame(
        [("Input orders", len(distinct)),
         ("Analysis orders", len(orders)),
         ("Orders requiring review", len(reviewed_ids)),
         ("Late analysis orders", int(orders["is_late"].sum()))],
        columns=["measure", "orders"],
    )
    return orders, {
        "review_reasons": review_reasons,
        "review_items": review_items,
        "cohort_summary": summary,
    }
