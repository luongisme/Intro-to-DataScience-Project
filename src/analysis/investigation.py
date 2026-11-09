"""Cohort stability and recorded delivery-stage durations."""

import numpy as np
import pandas as pd

from .orders import SLOT_LABELS
from .summary import summarize_delivery_by_slot


ROUTE_LABELS = ("Intra-state", "Inter-state", "Mixed", "Unknown")
REVIEW_DESTINATION = "Requires review"


def summarize_destination_states(focus: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Count destination outcomes for all focus orders and the inter-state subset.

    Late rate uses each destination's orders. Late share uses all late orders
    in the selected scope, including orders whose destination needs review.
    """
    required = {"order_id", "customer_state", "location_status", "route_type", "is_late"}
    missing = required.difference(focus.columns)
    if missing:
        raise ValueError(f"Missing focus columns: {sorted(missing)}")
    if focus["order_id"].isna().any() or not focus["order_id"].is_unique:
        raise ValueError("Destination profiles require one row per non-null order_id.")
    if focus["is_late"].isna().any() or not focus["is_late"].isin([False, True]).all():
        raise ValueError("Delivery outcome must be known for every focus order.")
    if not focus["location_status"].isin(["known", "missing", "incomplete", "conflicting"]).all():
        raise ValueError("Focus orders contain an unrecognized location status.")
    if not focus["route_type"].isin(ROUTE_LABELS).all():
        raise ValueError("Focus orders contain an unrecognized route classification.")
    state = focus["customer_state"].astype("string")
    known = focus["location_status"].eq("known")
    if (known & (state.isna() | state.str.strip().eq(""))).any():
        raise ValueError("Known destinations must have a nonblank customer state.")
    view = focus[["customer_state", "route_type", "is_late"]].copy()
    view["customer_state"] = state.where(known, REVIEW_DESTINATION)
    profiles = {}
    for name, selected in (("all_routes", view),
                           ("inter_state", view.loc[view["route_type"].eq("Inter-state")])):
        table = selected.groupby("customer_state", observed=True).agg(
            orders=("is_late", "size"), late_orders=("is_late", "sum"),
        ).reset_index()
        table["late_rate_pct"] = 100 * table["late_orders"] / table["orders"]
        total_late = int(selected["is_late"].sum())
        table["late_share_pct"] = 100 * table["late_orders"] / total_late if total_late else np.nan
        profiles[name] = table.sort_values(
            ["late_orders", "orders", "customer_state"], ascending=[False, False, True],
        ).reset_index(drop=True)
    return profiles


def build_focus_cohort(items: pd.DataFrame, orders: pd.DataFrame,
                       routes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Prepare the 12:00–18:00 order cohort with route and destination coverage.

    A destination is known only when every item has the same nonblank customer
    state. Missing or conflicting item states remain in the cohort for review.
    """
    required_items = {"order_id", "customer_state"}
    required_orders = {"order_id", "approval_slot", "is_late"}
    required_routes = {"order_id", "route_type", "items"}
    for frame, required, label in ((items, required_items, "items"),
                                    (orders, required_orders, "orders"),
                                    (routes, required_routes, "routes")):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {label} columns: {sorted(missing)}")
    if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
        raise ValueError("Analysis orders must have unique, non-null order_id values.")

    focus = orders.loc[orders["approval_slot"].eq(SLOT_LABELS[2])].copy()
    selected = items.loc[items["order_id"].isin(focus["order_id"]),
                         ["order_id", "customer_state"]].copy()
    selected["customer_state"] = selected["customer_state"].astype("string").str.strip().replace("", pd.NA)
    grouped = selected.groupby("order_id", sort=False, observed=True)["customer_state"]
    destinations = grouped.agg(items="size", distinct_states="nunique",
                               missing_items=lambda state: state.isna().sum(),
                               customer_state="first")
    destinations["location_status"] = np.select(
        [destinations["distinct_states"].gt(1),
         destinations["missing_items"].eq(destinations["items"]),
         destinations["missing_items"].gt(0)],
        ["conflicting", "missing", "incomplete"], default="known",
    )
    destinations["customer_state"] = destinations["customer_state"].where(
        destinations["location_status"].eq("known"), pd.NA)
    reviewed_ids = destinations.index[destinations["location_status"].ne("known")]
    observed = (selected.loc[selected["order_id"].isin(reviewed_ids)]
                .groupby("order_id", sort=False, observed=True)["customer_state"]
                .agg(lambda states: ", ".join(sorted(states.dropna().unique()))))
    destinations["observed_states"] = observed.reindex(destinations.index).fillna("")

    focus = focus.merge(destinations.reset_index(), on="order_id", how="left", validate="one_to_one")
    if focus["items"].isna().any():
        raise ValueError("Some focus orders have no source items.")
    focus = focus.merge(routes[["order_id", "route_type", "items"]], on="order_id",
                        how="left", validate="one_to_one", suffixes=("", "_route"))
    if focus["route_type"].isna().any() or focus["items_route"].isna().any():
        raise ValueError("Some focus orders have no route classification.")
    if not focus["items"].eq(focus["items_route"]).all():
        raise ValueError("Route item counts differ from focus source items.")
    focus.drop(columns=["items_route", "distinct_states", "missing_items", "observed_states"], inplace=True)

    review = destinations.loc[reviewed_ids, ["location_status", "items", "distinct_states",
                                              "missing_items", "observed_states"]].reset_index()
    late_orders = int(focus["is_late"].sum())
    summary = pd.DataFrame([{
        "approval_slot": SLOT_LABELS[2], "orders": len(focus),
        "on_time_orders": len(focus) - late_orders, "late_orders": late_orders,
        "late_rate_pct": 100 * late_orders / len(focus) if len(focus) else np.nan,
        "known_orders": int(focus["location_status"].eq("known").sum()),
        "review_orders": len(review),
    }])
    return focus, {"summary": summary, "location_review": review}


def classify_order_routes(items: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Classify each analyzed order from every retained seller/customer state pair.

    A missing or blank state on any item makes the whole route unknown. Items
    belonging to reviewed orders are outside the analyzed order cohort.
    """
    required_items = {"order_id", "customer_state", "seller_state"}
    missing_items = required_items.difference(items.columns)
    if missing_items:
        raise ValueError(f"Missing item route columns: {sorted(missing_items)}")
    if "order_id" not in orders:
        raise ValueError("Missing order_id in analysis orders.")
    if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
        raise ValueError("Analysis orders must have unique, non-null order_id values.")

    selected = items.loc[items["order_id"].isin(orders["order_id"]),
                         ["order_id", "customer_state", "seller_state"]].copy()
    if selected["order_id"].isna().any():
        raise ValueError("Selected items must have non-null order_id values.")
    known = (selected["customer_state"].notna() & selected["seller_state"].notna()
             & selected["customer_state"].astype("string").str.strip().ne("").fillna(False)
             & selected["seller_state"].astype("string").str.strip().ne("").fillna(False))
    selected["unknown"] = ~known
    selected["intra"] = known & selected["customer_state"].eq(selected["seller_state"]).fillna(False)
    selected["inter"] = known & ~selected["customer_state"].eq(selected["seller_state"]).fillna(False)
    grouped = selected.groupby("order_id", sort=False, observed=True).agg(
        items=("order_id", "size"), unknown=("unknown", "any"),
        intra=("intra", "any"), inter=("inter", "any"),
    )
    missing_ids = orders.loc[~orders["order_id"].isin(grouped.index), "order_id"]
    if not missing_ids.empty:
        raise ValueError(f"No item route for {len(missing_ids)} analysis orders.")
    grouped["route_type"] = np.select(
        [grouped["unknown"], grouped["intra"] & grouped["inter"], grouped["intra"], grouped["inter"]],
        ["Unknown", "Mixed", "Intra-state", "Inter-state"], default="Unknown",
    )
    routes = grouped.reset_index()[["order_id", "route_type", "items"]]
    routes["route_type"] = pd.Categorical(routes["route_type"], categories=ROUTE_LABELS, ordered=True)
    return routes


def summarize_routes(orders: pd.DataFrame, routes: pd.DataFrame,
                     stage_records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Count delivery outcomes by route and approval slot; summarize transit.

    Empty groups retain zero counts and undefined rates or durations. Transit
    statistics include only valid carrier-to-customer intervals.
    """
    required_orders = {"order_id", "approval_slot", "is_late"}
    required_routes = {"order_id", "route_type"}
    required_stages = {"order_id", "stage", "status", "calendar_hours", "eligible_hours"}
    for frame, required, label in ((orders, required_orders, "orders"),
                                    (routes, required_routes, "routes"),
                                    (stage_records, required_stages, "stage records")):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {label} columns: {sorted(missing)}")
    if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
        raise ValueError("Analysis orders must have unique, non-null order_id values.")
    if routes["order_id"].isna().any() or routes["order_id"].duplicated().any():
        raise ValueError("Routes must have unique, non-null order_id values.")
    if set(routes["order_id"]) != set(orders["order_id"]):
        raise ValueError("Route order IDs must match analysis order IDs exactly.")
    if routes["route_type"].isna().any() or not routes["route_type"].isin(ROUTE_LABELS).all():
        raise ValueError("Routes contain an unrecognized classification.")
    if orders["approval_slot"].isna().any() or not orders["approval_slot"].isin(SLOT_LABELS).all():
        raise ValueError("Orders contain an unrecognized approval slot.")
    if orders["is_late"].isna().any():
        raise ValueError("Delivery outcome must be known for every analysis order.")

    view = orders[["order_id", "approval_slot", "is_late"]].merge(
        routes[["order_id", "route_type"]], on="order_id", validate="one_to_one")
    view["route_type"] = pd.Categorical(view["route_type"], categories=ROUTE_LABELS, ordered=True)
    view["approval_slot"] = pd.Categorical(view["approval_slot"], categories=SLOT_LABELS, ordered=True)
    cell_index = pd.MultiIndex.from_product(
        [ROUTE_LABELS, SLOT_LABELS], names=["route_type", "approval_slot"])
    slots = view.groupby(["route_type", "approval_slot"], observed=True).agg(
        orders=("order_id", "size"), late_orders=("is_late", "sum"),
    ).reindex(cell_index, fill_value=0).reset_index()
    slots["orders"] = slots["orders"].astype("int64")
    slots["late_orders"] = slots["late_orders"].astype("int64")
    slots["on_time_orders"] = slots["orders"] - slots["late_orders"]
    slots["late_rate_pct"] = slots["late_orders"].div(slots["orders"].replace(0, np.nan)).mul(100)

    totals = slots.groupby("route_type", observed=True).agg(
        orders=("orders", "sum"), on_time_orders=("on_time_orders", "sum"),
        late_orders=("late_orders", "sum"),
    ).reindex(ROUTE_LABELS, fill_value=0).reset_index()
    totals["late_rate_pct"] = totals["late_orders"].div(totals["orders"].replace(0, np.nan)).mul(100)

    transit = stage_records.loc[stage_records["stage"].eq("Carrier transit")].copy()
    if transit["order_id"].isna().any() or transit["order_id"].duplicated().any():
        raise ValueError("Carrier transit must have one record per analysis order.")
    if set(transit["order_id"]) != set(orders["order_id"]):
        raise ValueError("Carrier transit order IDs must match analysis order IDs exactly.")
    transit = transit.merge(routes[["order_id", "route_type"]], on="order_id", validate="one_to_one")
    transit["route_type"] = pd.Categorical(transit["route_type"], categories=ROUTE_LABELS, ordered=True)
    valid = transit["status"].eq("valid")
    if transit.loc[valid, ["calendar_hours", "eligible_hours"]].isna().any().any():
        raise ValueError("Valid carrier transit records must have both durations.")
    if transit.loc[~valid, ["calendar_hours", "eligible_hours"]].notna().any().any():
        raise ValueError("Invalid carrier transit records must not have durations.")
    transit.loc[~valid, ["calendar_hours", "eligible_hours"]] = np.nan
    duration = transit.groupby("route_type", observed=True).agg(
        orders=("order_id", "size"), evaluated=("calendar_hours", "count"),
        mean_calendar_hours=("calendar_hours", "mean"), median_calendar_hours=("calendar_hours", "median"),
        mean_eligible_hours=("eligible_hours", "mean"), median_eligible_hours=("eligible_hours", "median"),
    ).reindex(ROUTE_LABELS).reset_index()
    duration[["orders", "evaluated"]] = duration[["orders", "evaluated"]].fillna(0).astype("int64")
    duration["unavailable"] = duration["orders"] - duration["evaluated"]
    return {"routes": routes, "totals": totals, "slots": slots, "transit": duration}


def summarize_time_periods(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Use purchase months for cohorts and approval times for slot assignment."""
    monthly = []
    purchase = orders["order_purchase_timestamp"]
    months = purchase.dt.to_period("M")
    if not orders.empty:
        for month in pd.period_range(months.min(), months.max(), freq="M"):
            monthly.append(summarize_delivery_by_slot(orders.loc[months.eq(month)]).assign(month=str(month)))
    columns = [*summarize_delivery_by_slot(orders.iloc[:0]).columns, "month"]
    monthly_table = pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame(columns=columns)
    matched = []
    for year in (2017, 2018):
        selected = orders.loc[purchase.dt.year.eq(year) & purchase.dt.month.le(7)]
        matched.append(summarize_delivery_by_slot(selected).assign(year=year))
    return {"monthly": monthly_table, "matched_years": pd.concat(matched, ignore_index=True)}


def compare_cleaning_cohorts(merged: pd.DataFrame, orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare slots only if the clean cohort is a matching subset of eligible source orders.

    On a mismatch, coverage counts the affected clean orders and slots is empty.
    """
    fields = ["order_status", "order_approved_at", "order_delivered_customer_date",
              "order_estimated_delivery_date"]
    distinct = merged.groupby("order_id")[fields].nunique(dropna=False)
    conflicts = distinct.index[distinct.gt(1).any(axis=1)]
    source = merged.loc[~merged["order_id"].isin(conflicts), ["order_id", *fields]].drop_duplicates("order_id").copy()
    delivered = source.loc[source["order_status"].eq("delivered")].copy()
    comparable = delivered.dropna(subset=fields[1:]).copy()
    clean_fields = orders.set_index("order_id")[fields]
    if not clean_fields.index.is_unique:
        raise ValueError("Clean cohort comparisons require one row per order_id.")
    before_fields = comparable.set_index("order_id")[fields]
    shared_ids = clean_fields.index.intersection(before_fields.index)
    missing_orders = len(clean_fields) - len(shared_ids)
    matches = clean_fields.loc[shared_ids].eq(before_fields.loc[shared_ids]).fillna(False)
    changed_orders = int((~matches.all(axis=1)).sum())
    comparable["approval_slot"] = pd.Categorical.from_codes(
        comparable["order_approved_at"].dt.hour // 6, categories=SLOT_LABELS, ordered=True)
    days = (comparable["order_delivered_customer_date"].dt.normalize() -
            comparable["order_estimated_delivery_date"].dt.normalize()).dt.days
    comparable["is_late"] = days.gt(0)
    comparable["delay_days"] = days.where(days.gt(0)).astype("Int64")
    coverage = pd.DataFrame([
        ["Orders with conflicting comparison fields", len(conflicts)],
        ["Delivered orders with consistent comparison fields", len(delivered)],
        ["Delivered orders missing comparison timestamps", len(delivered) - len(comparable)],
        ["Comparable delivered orders before cleaning", len(comparable)],
        ["Analyzable orders after cleaning", len(orders)],
        ["Clean orders absent from the comparable source", missing_orders],
        ["Matched orders with different comparison fields", changed_orders],
    ], columns=["measure", "orders"])
    if missing_orders or changed_orders:
        slots = summarize_delivery_by_slot(orders.iloc[:0]).iloc[:0].assign(cohort="").copy()
    else:
        slots = pd.concat([summarize_delivery_by_slot(comparable).assign(cohort="Before cleaning"),
                           summarize_delivery_by_slot(orders).assign(cohort="After cleaning")],
                          ignore_index=True)
    return {"coverage": coverage, "slots": slots}


def measure_stage_hours(start: pd.Series, end: pd.Series) -> pd.DataFrame:
    """Measure [start, end) in recorded-clock and eligible-day hours.

    Eligible days are Monday-Friday excluding Jan 1, May 1 and Dec 25.
    A full eligible day contributes 24 hours, with partial endpoints preserved.
    Missing or reversed intervals receive null durations and an explicit status.
    """
    if not start.index.equals(end.index):
        raise ValueError("Stage endpoints must have matching indexes.")
    if start.dt.tz is not None or end.dt.tz is not None:
        raise ValueError("Use the timezone-naive recorded timestamps for this convention.")
    valid = start.notna() & end.notna() & end.ge(start)
    result = pd.DataFrame({"calendar_hours": np.nan, "eligible_hours": np.nan,
                           "status": "missing_endpoint"}, index=start.index)
    result.loc[start.notna() & end.notna() & end.lt(start), "status"] = "reversed"
    if valid.any():
        first_year = int(start.loc[valid].dt.year.min())
        last_year = int(end.loc[valid].dt.year.max())
        holidays = [f"{year}-{day}" for year in range(first_year, last_year + 1)
                    for day in ("01-01", "05-01", "12-25")]
        calendar = np.busdaycalendar(holidays=holidays)

        def accumulated_hours(timestamps):
            times = timestamps.to_numpy(dtype="datetime64[us]")
            dates = times.astype("datetime64[D]")
            days = np.busday_count("1970-01-01", dates, busdaycal=calendar)
            fractions = (times - dates) / np.timedelta64(1, "h")
            return 24 * days + np.is_busday(dates, busdaycal=calendar) * fractions

        result.loc[valid, "calendar_hours"] = (end.loc[valid] - start.loc[valid]).dt.total_seconds() / 3600
        result.loc[valid, "eligible_hours"] = accumulated_hours(end.loc[valid]) - accumulated_hours(start.loc[valid])
        result.loc[valid, "status"] = "valid"
    return result


def investigate_delivery_stages(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return stage records, valid-duration summaries and promised-date windows."""
    if orders["order_id"].isna().any() or not orders["order_id"].is_unique:
        raise ValueError("Stage durations require one row per non-null order_id.")
    orders = orders.reset_index(drop=True)
    records = []
    for name, start, end in (
        ("Processing", "order_approved_at", "order_delivered_carrier_date"),
        ("Carrier transit", "order_delivered_carrier_date", "order_delivered_customer_date"),
    ):
        frame = orders[["order_id", "approval_slot", "is_late"]].copy()
        durations = measure_stage_hours(orders[start], orders[end])
        records.append(frame.join(durations).assign(stage=name))
    stages = pd.concat(records, ignore_index=True)
    hours = ["calendar_hours", "eligible_hours"]
    pooled = stages.groupby(["stage", "is_late"])[hours].median().unstack("is_late")
    pooled = pooled.reindex(index=["Processing", "Carrier transit"],
                            columns=pd.MultiIndex.from_product([hours, [False, True]]))
    pooled.columns = ["on_time_calendar_hours", "late_calendar_hours",
                      "on_time_eligible_hours", "late_eligible_hours"]
    summary = stages.groupby(["stage", "approval_slot", "is_late"], observed=True).agg(
        orders=("order_id", "size"), evaluated=("calendar_hours", "count"),
        mean_calendar_hours=("calendar_hours", "mean"), median_calendar_hours=("calendar_hours", "median"),
        mean_eligible_hours=("eligible_hours", "mean"), median_eligible_hours=("eligible_hours", "median"),
    ).reset_index()
    windows = orders[["order_id", "approval_slot", "order_approved_at",
                      "order_estimated_delivery_date"]].copy()
    windows["promised_days"] = (orders["order_estimated_delivery_date"].dt.normalize() -
                               orders["order_approved_at"].dt.normalize()).dt.days
    promised = windows.groupby("approval_slot", observed=False).agg(
        orders=("order_id", "size"), mean_days=("promised_days", "mean"),
        median_days=("promised_days", "median"), min_days=("promised_days", "min"),
        max_days=("promised_days", "max"),
    ).reset_index()
    quality = stages.groupby(["stage", "status"]).size().rename("orders").reset_index()
    return {"records": stages, "summary": summary, "pooled_medians": pooled.reset_index(), "quality": quality,
            "windows": windows, "promised_summary": promised,
            "negative_windows": windows.loc[windows["promised_days"].lt(0)].copy()}
