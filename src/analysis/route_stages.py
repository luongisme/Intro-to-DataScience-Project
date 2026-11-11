"""Compare recorded delivery stages within matched route and purchase periods."""

import numpy as np
import pandas as pd

from .orders import SLOT_LABELS


PURCHASE_PERIODS = (
    ("Nov–Dec 2017", "2017-11", "2017-12"),
    ("Feb–Mar 2018", "2018-02", "2018-03"),
    ("Apr–Jul 2018", "2018-04", "2018-07"),
)
STAGES = ("Processing", "Carrier transit")


def compare_route_delivery_stages(items: pd.DataFrame, orders: pd.DataFrame,
                                  routes: pd.DataFrame, pair_orders: pd.DataFrame,
                                  stage_profiles: dict[str, pd.DataFrame],
                                  state_pair: tuple[str, str]) -> dict[str, pd.DataFrame]:
    """Compare a selected pair with other inter-state orders to its destination.

    Purchase periods and approval slots define like-for-like descriptive cells.
    Stage durations come from investigate_delivery_stages, including its
    invalid-interval status and eligible-day calendar convention.
    """
    needed = {"order_id", "order_purchase_timestamp", "approval_slot", "is_late"}
    if needed.difference(orders.columns) or {"order_id", "customer_state", "seller_state"}.difference(items.columns):
        raise ValueError("Orders or items lack route-stage comparison columns.")
    if {"order_id", "route_type", "items"}.difference(routes.columns):
        raise ValueError("Routes require order ID, route type and item counts.")
    if not isinstance(state_pair, tuple) or len(state_pair) != 2 or state_pair[0] == state_pair[1]:
        raise ValueError("State pair must identify an inter-state route.")
    if orders["order_id"].isna().any() or not orders["order_id"].is_unique:
        raise ValueError("Order input requires unique, non-null order IDs.")
    if pair_orders["order_id"].isna().any() or not pair_orders["order_id"].is_unique:
        raise ValueError("Selected pair requires unique, non-null order IDs.")
    if not set(pair_orders["order_id"]).issubset(set(orders["order_id"])):
        raise ValueError("Selected pair must be a subset of analysis orders.")
    if routes["order_id"].isna().any() or not routes["order_id"].is_unique:
        raise ValueError("Routes require unique, non-null order IDs.")
    if set(routes["order_id"]) != set(orders["order_id"]):
        raise ValueError("Route order IDs must match analysis orders.")
    if orders["order_purchase_timestamp"].isna().any() or not pd.api.types.is_datetime64_any_dtype(orders["order_purchase_timestamp"]):
        raise ValueError("Purchase timestamps must be complete datetimes.")
    if orders["approval_slot"].isna().any() or not orders["approval_slot"].isin(SLOT_LABELS).all():
        raise ValueError("Approval slots must be recognized and complete.")
    if orders["is_late"].isna().any() or not orders["is_late"].isin([False, True]).all():
        raise ValueError("Delivery outcomes must be binary and complete.")
    if not {"records", "windows"}.issubset(stage_profiles):
        raise ValueError("Stage profiles require records and windows.")

    selected = items.loc[items["order_id"].isin(orders["order_id"]),
                         ["order_id", "customer_state", "seller_state"]].copy()
    state = selected["customer_state"].astype("string").str.strip()
    seller = selected["seller_state"].astype("string").str.strip()
    selected["customer_state"] = state.replace("", pd.NA)
    selected["known_rj"] = state.eq(state_pair[1]).fillna(False)
    selected["matches_pair"] = (selected["known_rj"]
                                & seller.eq(state_pair[0]).fillna(False))
    selected["known"] = state.notna() & state.ne("").fillna(False)
    destination = selected.groupby("order_id", sort=False, observed=True).agg(
        item_count=("known_rj", "size"), all_destination=("known_rj", "all"),
        complete_destination=("known", "all"),
        distinct_destination=("customer_state", "nunique"),
        all_pair_items=("matches_pair", "all"),
    )
    destination["complete_destination"] &= destination["distinct_destination"].eq(1)
    route_view = routes.set_index("order_id")
    expected = route_view["items"].reindex(orders["order_id"])
    actual = destination["item_count"].reindex(orders["order_id"])
    if expected.isna().any() or actual.isna().any() or not expected.eq(actual).all():
        raise ValueError("Item counts differ from route input. Supply complete cleaned items.")
    if not set(pair_orders["order_id"]).issubset(set(destination.index[destination["all_pair_items"]])):
        raise ValueError("Selected pair must have the stated seller and customer states on every item.")
    pair_ids = set(pair_orders["order_id"])
    view = orders[["order_id", "order_purchase_timestamp", "approval_slot", "is_late"]].reset_index(drop=True).copy()
    view = view.join(route_view["route_type"].reindex(view["order_id"]).reset_index(drop=True))
    view = view.join(destination[["all_destination", "complete_destination"]].reindex(view["order_id"]).reset_index(drop=True))
    view["purchase_month"] = view["order_purchase_timestamp"].dt.strftime("%Y-%m")
    view["period"] = pd.NA
    for label, start, end in PURCHASE_PERIODS:
        view.loc[view["purchase_month"].between(start, end), "period"] = label
    view = view.loc[view["period"].notna()].copy()
    view["group"] = np.select(
        [view["order_id"].isin(pair_ids),
         view["route_type"].eq("Inter-state") & view["all_destination"]],
        [f"{state_pair[0]} → {state_pair[1]}", f"Other inter-state to {state_pair[1]}"],
        default="Outside comparison",
    )
    if not view.loc[view["order_id"].isin(pair_ids), "route_type"].eq("Inter-state").all():
        raise ValueError("Selected pair conflicts with inter-state route classification.")
    groups = (f"{state_pair[0]} → {state_pair[1]}", f"Other inter-state to {state_pair[1]}")
    coverage = []
    for period, _, _ in PURCHASE_PERIODS:
        rows = view.loc[view["period"].eq(period)]
        inter = rows.loc[rows["route_type"].eq("Inter-state")]
        compared = rows.loc[rows["group"].isin(groups)]
        coverage.append({
            "period": period, "inter_state_orders": len(inter),
            "inter_state_unknown_destination": int((~inter["complete_destination"]).sum()),
            "inter_state_to_destination": int(inter["all_destination"].sum()),
            "selected_pair": int(compared["group"].eq(groups[0]).sum()),
            "other_to_destination": int(compared["group"].eq(groups[1]).sum()),
        })
    cohort = view.loc[view["group"].isin(groups)].copy()
    base_index = pd.MultiIndex.from_product(
        [[entry[0] for entry in PURCHASE_PERIODS], groups, SLOT_LABELS],
        names=["period", "group", "approval_slot"],
    )
    outcome = cohort.groupby(list(base_index.names), observed=True).agg(
        orders=("order_id", "size"), late_orders=("is_late", "sum"),
    ).reindex(base_index, fill_value=0).reset_index()
    outcome[["orders", "late_orders"]] = outcome[["orders", "late_orders"]].astype(int)
    outcome["late_rate_pct"] = 100 * outcome["late_orders"] / outcome["orders"].replace(0, np.nan)

    records = stage_profiles["records"]
    required_records = {"order_id", "stage", "status", "calendar_hours", "eligible_hours"}
    if required_records.difference(records.columns) or records.duplicated(["order_id", "stage"]).any():
        raise ValueError("Stage records require unique order-stage rows and duration fields.")
    windows = stage_profiles["windows"]
    if {"order_id", "promised_days"}.difference(windows.columns) or windows["order_id"].duplicated().any():
        raise ValueError("Promised windows require one row per order.")
    if set(records["order_id"]) != set(orders["order_id"]) or set(windows["order_id"]) != set(orders["order_id"]):
        raise ValueError("Stage records and promised windows must cover analysis orders.")
    if (not records["stage"].isin(STAGES).all()
            or records.groupby("order_id", observed=True)["stage"].nunique().ne(len(STAGES)).any()):
        raise ValueError("Each order requires both delivery-stage records.")
    stage_rows = cohort[["order_id", "period", "group", "approval_slot", "is_late"]].merge(
        records[["order_id", "stage", "status", "calendar_hours", "eligible_hours"]],
        on="order_id", validate="one_to_many",
    )
    valid = stage_rows["status"].eq("valid")
    if (not stage_rows["status"].isin(["valid", "missing_endpoint", "reversed"]).all()
            or stage_rows.loc[valid, ["calendar_hours", "eligible_hours"]].isna().any().any()
            or stage_rows.loc[~valid, ["calendar_hours", "eligible_hours"]].notna().any().any()):
        raise ValueError("Stage status and duration values disagree.")

    def duration_summary(keys, grid):
        table = stage_rows.groupby(keys, observed=True).agg(
            orders=("order_id", "size"), valid_intervals=("calendar_hours", "count"),
            median_calendar_hours=("calendar_hours", "median"),
            p75_calendar_hours=("calendar_hours", lambda x: x.quantile(.75)),
            median_eligible_hours=("eligible_hours", "median"),
            p75_eligible_hours=("eligible_hours", lambda x: x.quantile(.75)),
        ).reindex(grid).reset_index()
        table[["orders", "valid_intervals"]] = table[["orders", "valid_intervals"]].fillna(0).astype(int)
        table["unavailable_intervals"] = table["orders"] - table["valid_intervals"]
        return table

    stage_grid = pd.MultiIndex.from_product(
        [[entry[0] for entry in PURCHASE_PERIODS], groups, SLOT_LABELS, STAGES],
        names=[*base_index.names, "stage"],
    )
    stages = duration_summary(list(stage_grid.names), stage_grid)
    outcome_grid = pd.MultiIndex.from_product(
        [[entry[0] for entry in PURCHASE_PERIODS], groups, SLOT_LABELS, STAGES, [False, True]],
        names=[*base_index.names, "stage", "is_late"],
    )
    by_outcome = duration_summary(list(outcome_grid.names), outcome_grid)
    promised = cohort[["order_id", "period", "group", "approval_slot"]].merge(
        windows[["order_id", "promised_days"]], on="order_id", validate="one_to_one")
    promises = promised.groupby(list(base_index.names), observed=True).agg(
        orders=("order_id", "size"), evaluated=("promised_days", "count"),
        median_days=("promised_days", "median"),
        p75_days=("promised_days", lambda x: x.quantile(.75)),
        negative_windows=("promised_days", lambda x: x.lt(0).sum()),
    ).reindex(base_index).reset_index()
    promises[["orders", "evaluated", "negative_windows"]] = (
        promises[["orders", "evaluated", "negative_windows"]].fillna(0).astype(int))
    promises["unavailable_windows"] = promises["orders"] - promises["evaluated"]
    return {"coverage": pd.DataFrame(coverage), "outcomes": outcome,
            "stages": stages, "by_outcome": by_outcome, "promises": promises}
