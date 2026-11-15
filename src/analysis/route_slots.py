"""Compare approval slots within a complete seller-to-customer state pair."""

import numpy as np
import pandas as pd

from .orders import SLOT_LABELS
from .statistics import _wilson_interval


def build_state_pair_orders(items: pd.DataFrame, orders: pd.DataFrame,
                            routes: pd.DataFrame,
                            state_pair: tuple[str, str] | None, *,
                            include_intra_state: bool = False) -> pd.DataFrame:
    """Select orders whose every item belongs to the same known state pair.

    Item counts must agree with the existing order-route classification. This
    retains multi-item orders while excluding multi-state seller groups.
    Intra-state pairs require include_intra_state=True; the default is inter-state.
    """
    if not isinstance(include_intra_state, bool):
        raise ValueError("include_intra_state must be a boolean.")
    item_fields = {"order_id", "order_item_id", "seller_state", "customer_state"}
    order_fields = {"order_id", "approval_slot", "is_late", "order_purchase_timestamp"}
    route_fields = {"order_id", "route_type", "items"}
    for name, frame, fields in (("items", items, item_fields),
                                ("orders", orders, order_fields),
                                ("routes", routes, route_fields)):
        missing = fields.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
    if (orders["order_id"].isna().any() or not orders["order_id"].is_unique
            or routes["order_id"].isna().any() or not routes["order_id"].is_unique):
        raise ValueError("Orders and routes require unique, non-null order IDs.")
    if set(orders["order_id"]) != set(routes["order_id"]):
        raise ValueError("Route order IDs must match analysis order IDs exactly.")
    if (orders["approval_slot"].isna().any()
            or not orders["approval_slot"].isin(SLOT_LABELS).all()):
        raise ValueError("Orders require a recognized approval slot.")
    if (orders["is_late"].isna().any()
            or not orders["is_late"].isin([False, True]).all()):
        raise ValueError("Orders require a binary delivery outcome.")
    purchase = orders["order_purchase_timestamp"]
    if purchase.isna().any() or not pd.api.types.is_datetime64_any_dtype(purchase):
        raise ValueError("Orders require complete purchase timestamps.")
    if state_pair is None:
        return orders.iloc[:0].copy()
    if (not isinstance(state_pair, tuple) or len(state_pair) != 2
            or any(not isinstance(state, str) or not state.strip() for state in state_pair)):
        raise ValueError("State pair must contain two nonblank state codes.")
    seller, customer = (state.strip() for state in state_pair)
    if seller == customer and not include_intra_state:
        raise ValueError("The selected state pair must be inter-state.")

    selected = items.loc[items["order_id"].isin(orders["order_id"]),
                         list(item_fields)].copy()
    keys = selected[["order_id", "order_item_id"]]
    if keys.isna().any().any() or keys.duplicated().any():
        raise ValueError("Selected items require unique, non-null item keys.")
    for field in ("seller_state", "customer_state"):
        selected[field] = selected[field].astype("string").str.strip().replace("", pd.NA)
    selected["matches_pair"] = (selected["seller_state"].eq(seller).fillna(False)
                                & selected["customer_state"].eq(customer).fillna(False))
    grouped = selected.groupby("order_id", sort=False, observed=True).agg(
        item_count=("order_item_id", "size"), all_items_match=("matches_pair", "all"),
    )
    counts = routes.set_index("order_id")["items"]
    actual = grouped["item_count"].reindex(orders["order_id"])
    expected = counts.reindex(orders["order_id"])
    if actual.isna().any() or expected.isna().any() or not actual.eq(expected).all():
        raise ValueError("Item counts differ from the route classification. Supply complete items.")
    chosen = grouped.index[grouped["all_items_match"]]
    pair = orders.loc[orders["order_id"].isin(chosen)].copy()
    expected_route = "Intra-state" if seller == customer else "Inter-state"
    if not routes.set_index("order_id").loc[pair["order_id"], "route_type"].eq(expected_route).all():
        raise ValueError("The selected pair conflicts with the route classification.")
    return pair


def _purchase_month_profile(
    orders: pd.DataFrame, slots: tuple[str, ...], months: pd.PeriodIndex | None = None,
) -> pd.DataFrame:
    """Count validated orders on a complete purchase-month × approval-slot grid."""
    purchase = orders["order_purchase_timestamp"]
    if months is None:
        months = (pd.period_range(purchase.min().to_period("M"), purchase.max().to_period("M"), freq="M")
                  if not orders.empty else pd.PeriodIndex([], freq="M"))
    if not len(months):
        return pd.DataFrame(columns=["purchase_month", "approval_slot", "orders",
                                     "late_orders", "late_rate_pct"])
    grid = pd.MultiIndex.from_product([months, slots], names=["purchase_month", "approval_slot"])
    monthly = orders.assign(purchase_month=purchase.dt.to_period("M")).groupby(
        ["purchase_month", "approval_slot"], observed=True,
    ).agg(orders=("order_id", "size"), late_orders=("is_late", "sum"))
    monthly = monthly.reindex(grid, fill_value=0).reset_index()
    monthly[["orders", "late_orders"]] = monthly[["orders", "late_orders"]].astype("int64")
    monthly["late_rate_pct"] = 100 * monthly["late_orders"] / monthly["orders"].replace(0, np.nan)
    monthly["purchase_month"] = monthly["purchase_month"].astype(str)
    return monthly


def _standardize_month_profile(
    monthly: pd.DataFrame, slots: tuple[str, ...], *, excluded_months: tuple[str, ...] = (),
) -> dict[str, pd.DataFrame]:
    """Use pooled order weights in months represented in every requested slot.

    Counts and raw rates retain the full monthly profile. Excluded months only
    restrict the common-month comparison; they never remove source orders.
    """
    sizes = monthly.pivot(index="purchase_month", columns="approval_slot", values="orders").reindex(columns=slots)
    common = sizes.index[sizes.gt(0).all(axis=1)].difference(list(excluded_months), sort=False)
    eligible = monthly.loc[monthly["purchase_month"].isin(common)]
    pooled = eligible.groupby("purchase_month", observed=True)["orders"].sum().astype("int64")
    weights = pooled / pooled.sum() if not pooled.empty else pooled.astype(float)
    totals = monthly.groupby("approval_slot", observed=True)[["orders", "late_orders"]].sum().reindex(slots, fill_value=0)
    records = []
    for slot in slots:
        cells = eligible.loc[eligible["approval_slot"].eq(slot)]
        used = int(cells["orders"].sum())
        used_late = int(cells["late_orders"].sum())
        total = int(totals.loc[slot, "orders"])
        total_late = int(totals.loc[slot, "late_orders"])
        rate = (cells.set_index("purchase_month")["late_rate_pct"] * weights).sum()
        records.append({
            "approval_slot": slot, "common_months": len(common),
            "included_orders": used, "excluded_orders": total - used,
            "included_late_orders": used_late, "excluded_late_orders": total_late - used_late,
            "raw_late_rate_pct": 100 * total_late / total if total else np.nan,
            "common_raw_late_rate_pct": 100 * used_late / used if used else np.nan,
            "standardized_late_rate_pct": float(rate) if len(common) else np.nan,
        })
    weight_table = pd.DataFrame({"pooled_orders": pooled, "weight": weights}).rename_axis("purchase_month").reset_index()
    return {"summary": pd.DataFrame(records), "weights": weight_table}


def summarize_state_pair_slots(pair_orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return raw and month-standardized late rates for the four approval slots.

    Common-month weights follow pooled orders across all four slots in months
    where every slot is observed. The interior-month view excludes the first
    and last observed purchase months as a sensitivity check. Raw-rate Wilson
    intervals describe binomial sampling precision if orders are independent;
    they do not adjust for shared sellers, dates or route conditions.
    """
    required = {"order_id", "approval_slot", "is_late", "order_purchase_timestamp"}
    missing = required.difference(pair_orders.columns)
    if missing:
        raise ValueError(f"Missing pair columns: {sorted(missing)}")
    if pair_orders["order_id"].isna().any() or not pair_orders["order_id"].is_unique:
        raise ValueError("Pair analysis requires one row per non-null order ID.")
    if (pair_orders["approval_slot"].isna().any()
            or not pair_orders["approval_slot"].isin(SLOT_LABELS).all()):
        raise ValueError("Pair orders require a recognized approval slot.")
    if (pair_orders["is_late"].isna().any()
            or not pair_orders["is_late"].isin([False, True]).all()):
        raise ValueError("Pair orders require a binary delivery outcome.")
    purchase = pair_orders["order_purchase_timestamp"]
    if purchase.isna().any() or not pd.api.types.is_datetime64_any_dtype(purchase):
        raise ValueError("Pair orders require complete purchase timestamps.")

    slots = pd.Index(SLOT_LABELS, name="approval_slot")
    raw = pair_orders.groupby("approval_slot", observed=True).agg(
        orders=("order_id", "size"), late_orders=("is_late", "sum"),
    ).reindex(slots, fill_value=0).reset_index()
    raw[["orders", "late_orders"]] = raw[["orders", "late_orders"]].astype("int64")
    raw["late_rate_pct"] = 100 * raw["late_orders"] / raw["orders"].replace(0, np.nan)
    lower, upper = _wilson_interval(raw["late_orders"], raw["orders"])
    raw["late_rate_95_lower_pct"] = lower
    raw["late_rate_95_upper_pct"] = upper

    monthly = _purchase_month_profile(pair_orders, SLOT_LABELS)
    months = monthly["purchase_month"].drop_duplicates()
    endpoints = (str(months.iloc[0]), str(months.iloc[-1])) if len(months) else ()
    results = []
    columns = ["scope", "approval_slot", "common_months", "included_orders", "excluded_orders",
               "raw_late_rate_pct", "standardized_late_rate_pct"]
    for scope, excluded in (("Common months", ()), ("Interior common months", endpoints)):
        table = _standardize_month_profile(monthly, SLOT_LABELS, excluded_months=excluded)["summary"]
        table["scope"] = scope
        results.extend(table[columns].to_dict("records"))
    standardized = pd.DataFrame(results)
    reference = standardized.loc[standardized["approval_slot"].eq(SLOT_LABELS[2]),
                                 ["scope", "standardized_late_rate_pct"]].set_index("scope")
    standardized["difference_pp_vs_12_18"] = (
        standardized["standardized_late_rate_pct"]
        - standardized["scope"].map(reference["standardized_late_rate_pct"])
    )
    return {"raw": raw, "monthly": monthly, "standardized": standardized}
