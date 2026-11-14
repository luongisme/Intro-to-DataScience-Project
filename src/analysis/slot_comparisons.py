"""Descriptive comparisons of overnight and afternoon approval cohorts."""

import numpy as np
import pandas as pd

from .investigation import REVIEW_DESTINATION, ROUTE_LABELS
from .orders import SLOT_LABELS
from .statistics import _wilson_interval


COMPARISON_SLOTS = (SLOT_LABELS[0], SLOT_LABELS[2])
_REVIEW_LABELS = ("Multiple seller states", REVIEW_DESTINATION)


def _describe_groups(table: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Attach rates, Wilson limits and shares using the complete slot totals."""
    table = table.copy()
    denominator = table["approval_slot"].map(totals["orders"])
    late_denominator = table["approval_slot"].map(totals["late_orders"]).replace(0, np.nan)
    table["late_rate_pct"] = 100 * table["late_orders"] / table["orders"].replace(0, np.nan)
    lower, upper = _wilson_interval(table["late_orders"], table["orders"])
    table["wilson_lower_pct"], table["wilson_upper_pct"] = lower, upper
    table["order_share_pct"] = 100 * table["orders"] / denominator.replace(0, np.nan)
    table["late_share_pct"] = 100 * table["late_orders"] / late_denominator
    return table


def compare_approval_slot_groups(
    night: pd.DataFrame, afternoon: pd.DataFrame, *, group_by: str,
    top_n: int = 5, rate_top_n: int = 3, min_orders: int = 100,
    reference_pair: tuple[str, str] | None = ("SP", "RJ"),
) -> dict[str, pd.DataFrame]:
    """Compare full cohorts by destination or eligible seller-customer pair.

    Inputs contain one row per order in the stated slot. Pair inputs come from
    summarize_state_pairs(..., include_intra_state=True)["orders"], including
    multiple-state and review orders. No orders are removed from scope totals.

    Select the union of count leaders, rate leaders with N >= min_orders, the
    largest absolute rate gap with both N >= min_orders, and the reference pair
    if observed. Selection uses unrounded rates and deterministic tie breaking.
    It is exploratory; Wilson intervals do not establish pairwise significance.

    Return full and selected group-slot rows, selection reasons, scope totals,
    and plot rows with unselected eligible groups pooled as one remainder.
    """
    if group_by not in ("customer_state", "state_pair"):
        raise ValueError("group_by must be customer_state or state_pair.")
    for name, value in (("top_n", top_n), ("rate_top_n", rate_top_n), ("min_orders", min_orders)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if reference_pair is not None and (
        not isinstance(reference_pair, tuple) or len(reference_pair) != 2
        or any(not isinstance(state, str) or not state.strip() for state in reference_pair)
    ):
        raise ValueError("reference_pair must contain two nonblank state codes or be None.")
    required = {"order_id", "approval_slot", "is_late", "customer_state", "location_status", "route_type"}
    if group_by == "state_pair":
        required |= {"seller_state", "seller_group"}
    frames = []
    for slot, frame in zip(COMPARISON_SLOTS, (night, afternoon)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {slot} comparison columns: {sorted(missing)}")
        if frame["order_id"].isna().any() or not frame["order_id"].is_unique:
            raise ValueError("Slot comparison requires one row per non-null order_id.")
        if not frame["approval_slot"].eq(slot).fillna(False).all():
            raise ValueError(f"Every input order must belong to the {slot} approval slot.")
        if frame["is_late"].isna().any() or not frame["is_late"].isin([False, True]).all():
            raise ValueError("Every comparison order requires a binary delivery outcome.")
        if not frame["route_type"].isin(ROUTE_LABELS).all():
            raise ValueError("Comparison orders contain an unrecognized route classification.")
        if not frame["location_status"].isin(["known", "missing", "incomplete", "conflicting"]).all():
            raise ValueError("Comparison orders contain an unrecognized location status.")
        view = frame.copy()
        customer = view["customer_state"].astype("string").str.strip().replace("", pd.NA)
        known = view["location_status"].eq("known")
        if (known & customer.isna()).any():
            raise ValueError("Known destinations require a nonblank customer state.")
        view["customer_state"] = customer
        if group_by == "customer_state":
            view["group"] = customer.where(known, REVIEW_DESTINATION)
            view["eligible"] = known
        else:
            if not view["seller_group"].isin(["Single seller state", *_REVIEW_LABELS]).all():
                raise ValueError("Comparison orders contain an unrecognized seller group.")
            seller = view["seller_state"].astype("string").str.strip().replace("", pd.NA)
            single = view["seller_group"].eq("Single seller state")
            if (single & (~known | seller.isna())).any():
                raise ValueError("Eligible state pairs require known seller and customer states.")
            expected_route = np.where(seller.eq(customer).fillna(False), "Intra-state", "Inter-state")
            if (single & ~view["route_type"].eq(expected_route)).any():
                raise ValueError("Eligible state pairs conflict with the route classification.")
            view["group"] = (seller + " → " + customer).where(single, view["seller_group"])
            view["eligible"] = single
        frames.append(view[["order_id", "approval_slot", "is_late", "group", "eligible"]])
    if set(night["order_id"]) & set(afternoon["order_id"]):
        raise ValueError("Approval-slot cohorts must contain disjoint order IDs.")
    orders = pd.concat(frames, ignore_index=True)
    totals = orders.groupby("approval_slot", observed=True).agg(
        orders=("order_id", "size"), late_orders=("is_late", "sum"),
    ).reindex(COMPARISON_SLOTS, fill_value=0)
    totals[["orders", "late_orders"]] = totals[["orders", "late_orders"]].astype("int64")
    eligible_groups = sorted(orders.loc[orders["eligible"], "group"].unique())
    special_groups = list(_REVIEW_LABELS if group_by == "state_pair" else (REVIEW_DESTINATION,))
    groups = eligible_groups + special_groups
    grid = pd.MultiIndex.from_product([groups, COMPARISON_SLOTS], names=["group", "approval_slot"])
    full = orders.groupby(["group", "approval_slot"], observed=True).agg(
        orders=("order_id", "size"), late_orders=("is_late", "sum"),
    ).reindex(grid, fill_value=0).reset_index()
    full[["orders", "late_orders"]] = full[["orders", "late_orders"]].astype("int64")
    full["eligible"] = full["group"].isin(eligible_groups)
    full = _describe_groups(full, totals)
    rates = full.pivot(index="group", columns="approval_slot", values="late_rate_pct")
    gaps = rates[COMPARISON_SLOTS[0]] - rates[COMPARISON_SLOTS[1]]
    full["difference_pp"] = full["group"].map(gaps)

    reasons = {group: [] for group in eligible_groups}
    for slot in COMPARISON_SLOTS:
        rows = full.loc[full["eligible"] & full["approval_slot"].eq(slot)]
        leaders = rows.loc[rows["late_orders"].gt(0)].sort_values(
            ["late_orders", "orders", "group"], ascending=[False, False, True],
        ).head(top_n)
        rate_leaders = rows.loc[rows["orders"].ge(min_orders) & rows["late_orders"].gt(0)].sort_values(
            ["late_rate_pct", "late_orders", "orders", "group"], ascending=[False, False, False, True],
        ).head(rate_top_n)
        for group in leaders["group"]:
            reasons[group].append(f"Top late count: {slot}")
        for group in rate_leaders["group"]:
            reasons[group].append(f"Top late rate (N ≥ {min_orders}): {slot}")
    sizes = full.loc[full["eligible"]].pivot(index="group", columns="approval_slot", values="orders")
    differences = pd.DataFrame({
        "absolute_gap": gaps.reindex(eligible_groups).abs(),
        "pooled_orders": sizes.sum(axis=1),
    })
    valid = sizes.ge(min_orders).all(axis=1) & differences["absolute_gap"].gt(0)
    largest = differences.loc[valid].rename_axis("group").reset_index().sort_values(
        ["absolute_gap", "pooled_orders", "group"], ascending=[False, False, True],
    ).head(1)
    for group in largest["group"]:
        reasons[group].append(f"Largest absolute gap (both N ≥ {min_orders})")
    if group_by == "state_pair" and reference_pair is not None:
        reference = " → ".join(state.strip() for state in reference_pair)
        if reference in reasons:
            reasons[reference].append("Reference state pair")
    rank = full.loc[full["eligible"]].groupby("group", observed=True)[["late_orders", "orders"]].sum()
    rank = rank.rename_axis("group").reset_index().sort_values(
        ["late_orders", "orders", "group"], ascending=[False, False, True],
    )
    chosen = [group for group in rank["group"] if reasons[group]]
    selection = pd.DataFrame({
        "group": chosen,
        "selection_reason": ["; ".join(reasons[group]) for group in chosen],
    })
    selected = full.loc[full["group"].isin(chosen)].copy()
    order = {group: index for index, group in enumerate(chosen)}
    selected = selected.sort_values("group", key=lambda series: series.map(order), kind="stable").reset_index(drop=True)
    remainder = "Other eligible pairs" if group_by == "state_pair" else "Other destinations"
    remaining = full.loc[full["eligible"] & ~full["group"].isin(chosen)].groupby("approval_slot", observed=True)[
        ["orders", "late_orders"]
    ].sum().reindex(COMPARISON_SLOTS, fill_value=0).reset_index()
    remaining["group"] = remainder
    plot_rows = pd.concat([
        selected[["group", "approval_slot", "orders", "late_orders"]],
        remaining[["group", "approval_slot", "orders", "late_orders"]],
        full.loc[~full["eligible"], ["group", "approval_slot", "orders", "late_orders"]],
    ], ignore_index=True)
    plot_rows = _describe_groups(plot_rows, totals)
    scope_summary = _describe_groups(totals.reset_index(), totals)
    scope_summary["difference_pp"] = (
        scope_summary["late_rate_pct"].iloc[0] - scope_summary["late_rate_pct"].iloc[1]
    )
    return {
        "all_groups": full, "selected": selected, "selection": selection,
        "scope_summary": scope_summary, "plot_data": plot_rows,
    }
