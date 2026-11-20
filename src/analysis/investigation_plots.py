"""Figures for temporal stability, stage times and delivery promises."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .orders import SLOT_LABELS
from .investigation import REVIEW_DESTINATION


ELIGIBLE_DAY_NOTE = "Eligible hours count Mon–Fri, excluding 1 Jan, 1 May and 25 Dec (24 hours per full eligible day)."


def plot_slot_group_months(monthly, *, group_by, groups=None):
    """Show full monthly late rates and order volumes for chosen comparison groups."""
    from .slot_comparisons import COMPARISON_SLOTS
    from matplotlib.ticker import MaxNLocator

    if group_by not in ("customer_state", "state_pair"):
        raise ValueError("group_by must be customer_state or state_pair.")
    required = {"group", "purchase_month", "approval_slot", "orders", "late_orders", "late_rate_pct"}
    if required.difference(monthly.columns):
        raise ValueError("Monthly figures require group, purchase month, slot and outcome counts.")
    if monthly.duplicated(["group", "purchase_month", "approval_slot"]).any():
        raise ValueError("Monthly figures require unique group-month-slot rows.")
    available = monthly["group"].drop_duplicates().tolist()
    chosen = available if groups is None else list(groups)
    if len(set(chosen)) != len(chosen) or not set(chosen).issubset(available):
        raise ValueError("Plot groups must be distinct groups present in the monthly table.")
    if not chosen:
        figure, ax = plt.subplots(figsize=(10, 3), layout="constrained")
        ax.text(.5, .5, "No selected groups have monthly observations.", ha="center", va="center")
        ax.set_axis_off()
        return figure
    selected = monthly.loc[monthly["group"].isin(chosen)]
    if not selected["approval_slot"].isin(COMPARISON_SLOTS).all():
        raise ValueError("Selected monthly groups require recognized overnight and afternoon approval slots.")
    months = sorted(selected["purchase_month"].unique())
    dates = pd.PeriodIndex(months, freq="M").to_timestamp()
    grid = pd.MultiIndex.from_product(
        [chosen, months, COMPARISON_SLOTS], names=["group", "purchase_month", "approval_slot"],
    )
    rows = selected.set_index(list(grid.names)).reindex(grid)
    counts = rows[["orders", "late_orders"]].to_numpy(dtype=float)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any()
            or rows["late_orders"].gt(rows["orders"]).any()):
        raise ValueError("Every plotted group and month requires both slots and valid outcome counts.")
    expected = 100.0 * rows["late_orders"] / rows["orders"].replace(0, np.nan)
    if not np.isclose(rows["late_rate_pct"].to_numpy(dtype=float),
                      expected.to_numpy(dtype=float), equal_nan=True).all():
        raise ValueError("Monthly rates must use the matching order denominators.")
    figure, axes = plt.subplots(len(chosen), 2, figsize=(14, max(4.2, 3.1 * len(chosen))),
                                sharex=True, squeeze=False, layout="constrained")
    colors = ("#377eb8", "#388443")
    for index, group in enumerate(chosen):
        for slot, color in zip(COMPARISON_SLOTS, colors):
            table = rows.xs((group, slot), level=("group", "approval_slot")).reindex(months)
            axes[index, 0].plot(dates, table["late_rate_pct"], color=color, marker="o",
                                markersize=3, linewidth=1.2, label=slot)
            axes[index, 1].plot(dates, table["orders"], color=color, marker="o",
                                markersize=3, linewidth=1.2, label=slot)
        axes[index, 0].set_title(f"{group} · late delivery rate")
        axes[index, 0].set_ylabel("Late rate (%)")
        axes[index, 1].set_title(f"{group} · order volume")
        axes[index, 1].set_ylabel("Orders")
        axes[index, 1].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        for ax in axes[index]:
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    import matplotlib.dates as mdates
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=35)
    subject = "customer state" if group_by == "customer_state" else "seller → customer state pair"
    figure.suptitle(f"Purchase-month patterns by {subject}")
    figure.supxlabel("Purchase month · approval time determines slot; gaps in a rate series mean no orders")
    return figure


def plot_slot_group_comparison(table, group_by):
    """Compare counts and 95% Wilson rate intervals for complete two-slot plot rows."""
    from .slot_comparisons import COMPARISON_SLOTS
    from .statistics import _wilson_interval

    if group_by not in ("customer_state", "state_pair"):
        raise ValueError("group_by must be customer_state or state_pair.")
    required = {"group", "approval_slot", "orders", "late_orders", "late_rate_pct",
                "wilson_lower_pct", "wilson_upper_pct"}
    if required.difference(table.columns):
        raise ValueError("Comparison figures require group-slot counts, rates and Wilson limits.")
    if table["group"].isna().any() or table.duplicated(["group", "approval_slot"]).any():
        raise ValueError("Each comparison group and approval slot requires one row.")
    groups = table["group"].drop_duplicates().tolist()
    if not groups or set(table["approval_slot"]) != set(COMPARISON_SLOTS):
        raise ValueError("Comparison figures require both approval slots.")
    grid = pd.MultiIndex.from_product([groups, COMPARISON_SLOTS], names=["group", "approval_slot"])
    rows = table.set_index(["group", "approval_slot"]).reindex(grid)
    counts = rows[["orders", "late_orders"]].to_numpy(dtype=float)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any()
            or rows["late_orders"].gt(rows["orders"]).any()):
        raise ValueError("Every group requires both slots and valid nonnegative integer counts.")
    expected = 100.0 * rows["late_orders"] / rows["orders"].replace(0, np.nan)
    lower, upper = _wilson_interval(rows["late_orders"], rows["orders"])
    for column, values in (("late_rate_pct", expected), ("wilson_lower_pct", lower), ("wilson_upper_pct", upper)):
        if not np.isclose(rows[column].to_numpy(dtype=float),
                          values.to_numpy(dtype=float), equal_nan=True).all():
            raise ValueError("Rates and Wilson limits must use the matching order counts.")

    figure, axes = plt.subplots(1, 2, figsize=(14, max(5.4, .58 * len(groups) + 2.1)),
                                sharey=True, layout="constrained")
    positions = np.arange(len(groups))
    colors = ("#377eb8", "#388443")
    for offset, slot, color in zip((-.18, .18), COMPARISON_SLOTS, colors):
        values = rows.xs(slot, level="approval_slot").reindex(groups)
        y = positions + offset
        axes[0].barh(y, values["late_orders"], height=.32, color=color, label=slot)
        for row, position in zip(values.itertuples(), y):
            axes[0].annotate(f"{int(row.late_orders):,} / {int(row.orders):,}",
                             (row.late_orders, position), xytext=(4, 0),
                             textcoords="offset points", va="center", fontsize=8)
        available = values["orders"].gt(0)
        rates = values.loc[available, "late_rate_pct"].to_numpy(dtype=float)
        low = values.loc[available, "wilson_lower_pct"].to_numpy(dtype=float)
        high = values.loc[available, "wilson_upper_pct"].to_numpy(dtype=float)
        axes[1].errorbar(rates, y[available.to_numpy()],
                         xerr=np.maximum(0, np.array([rates - low, high - rates])),
                         fmt="o", color=color, capsize=3, markersize=4, linewidth=1.2)
        for position in y[~available.to_numpy()]:
            axes[1].annotate("No orders", (0, position), xytext=(3, 0),
                             textcoords="offset points", va="center", color=color, fontsize=8)
    axes[0].set_yticks(positions, groups)
    axes[0].set_ylim(len(groups) - .55, -1.15)
    axes[0].set_xlim(0, max(1, rows["late_orders"].max()) * 1.65)
    axes[0].set_title("Late order counts")
    axes[0].set_xlabel("Late orders · labels show late / all orders")
    axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=9)
    largest = rows["wilson_upper_pct"].max(skipna=True)
    axes[1].set_xlim(0, max(1, largest if pd.notna(largest) else 0) * 1.12)
    axes[1].set_title("Late rates and 95% Wilson intervals")
    axes[1].set_xlabel("Late deliveries / group orders (%)")
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    subject = "customer state" if group_by == "customer_state" else "seller → customer state pair"
    figure.suptitle(f"Delivery outcomes by approval slot and {subject}")
    figure.supxlabel("Other groups are pooled from counts; unresolved orders remain separate. No orders = undefined rate.")
    return figure


def plot_focus_routes(routes, approval_slot):
    """Show complete route counts and within-route late rates in one approval slot."""
    from .investigation import ROUTE_LABELS
    from .plots import ON_TIME_COLOR, LATE_COLOR

    if approval_slot not in SLOT_LABELS:
        raise ValueError("A recognized approval slot is required.")
    required = {"route_type", "orders", "late_orders", "late_rate_pct"}
    if required.difference(routes.columns):
        raise ValueError("Route figures require route counts and late rates.")
    if routes["route_type"].duplicated().any() or set(routes["route_type"]) != set(ROUTE_LABELS):
        raise ValueError("Route figures require each of the four route types once.")
    rows = routes.set_index("route_type").reindex(ROUTE_LABELS)
    counts = rows[["orders", "late_orders"]].to_numpy(dtype=float)
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any()
            or rows["late_orders"].gt(rows["orders"]).any()):
        raise ValueError("Route counts must be nonnegative integers with late orders within totals.")
    expected = 100.0 * rows["late_orders"] / rows["orders"].replace(0, np.nan)
    if not np.isclose(rows["late_rate_pct"].to_numpy(dtype=float),
                      expected.to_numpy(dtype=float), equal_nan=True).all():
        raise ValueError("Late rates must use the matching route order denominators.")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    positions = np.arange(len(ROUTE_LABELS))
    on_time = rows["orders"] - rows["late_orders"]
    axes[0].bar(positions, on_time, color=ON_TIME_COLOR, label="On time")
    axes[0].bar(positions, rows["late_orders"], bottom=on_time, color=LATE_COLOR, label="Late")
    axes[0].set_title("Order volume and late deliveries")
    axes[0].set_ylabel("Orders")
    axes[0].legend()
    axes[0].set_ylim(0, max(1, rows["orders"].max()) * 1.27)
    for position, row in zip(positions, rows.itertuples()):
        axes[0].annotate(f"N={row.orders:,}\nLate={row.late_orders:,}",
                         (position, row.orders), xytext=(0, 5), textcoords="offset points",
                         ha="center", fontsize=9)
    axes[1].bar(positions, rows["late_rate_pct"].fillna(0), color="#388443")
    axes[1].set_title("Late deliveries / route orders")
    axes[1].set_ylabel("Late rate (%)")
    maximum = rows["late_rate_pct"].max(skipna=True)
    axes[1].set_ylim(0, max(1, maximum if pd.notna(maximum) else 0) * 1.25)
    for position, row in zip(positions, rows.itertuples()):
        label = f"{row.late_rate_pct:.2f}%" if row.orders else "No orders"
        axes[1].annotate(label, (position, row.late_rate_pct if row.orders else 0),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    for ax in axes:
        ax.set_xticks(positions, ROUTE_LABELS, rotation=15)
    figure.suptitle(f"Delivery outcomes by route type · {approval_slot} approvals")
    figure.supxlabel("N = all orders in each route; each order is counted once")
    return figure


def plot_route_period_slot_rates(outcomes):
    """Show matched late rates with order denominators on a shared color scale."""
    required = {"period", "group", "approval_slot", "orders", "late_orders", "late_rate_pct"}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"Missing route-period outcome columns: {sorted(missing)}")
    if outcomes.duplicated(["period", "group", "approval_slot"]).any():
        raise ValueError("Each period, group and approval slot requires one row.")
    groups = outcomes["group"].drop_duplicates().tolist()
    periods = outcomes["period"].drop_duplicates().tolist()
    if len(groups) != 2 or not periods:
        raise ValueError("Rate comparison requires two groups and at least one purchase period.")
    if ((outcomes["orders"] < 0).any() or (outcomes["late_orders"] < 0).any()
            or (outcomes["late_orders"] > outcomes["orders"]).any()):
        raise ValueError("Order counts must be nonnegative with late orders within total orders.")
    if ((outcomes["orders"].eq(0) & outcomes["late_rate_pct"].notna()).any()
            or (outcomes["orders"].gt(0) & ~np.isclose(
                outcomes["late_rate_pct"],
                100 * outcomes["late_orders"] / outcomes["orders"].replace(0, np.nan),
                equal_nan=False,
            )).any()):
        raise ValueError("Late rates must use the matching order denominators.")

    maximum = max(1, outcomes["late_rate_pct"].max(skipna=True))
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True, layout="constrained")
    color_map = plt.colormaps["Greens"].with_extremes(bad="#eeeeee")
    for ax, group in zip(axes, groups):
        rows = outcomes.loc[outcomes["group"].eq(group)].set_index(["period", "approval_slot"])
        grid = pd.MultiIndex.from_product([periods, SLOT_LABELS], names=["period", "approval_slot"])
        rows = rows.reindex(grid)
        if rows["orders"].isna().any():
            plt.close(figure)
            raise ValueError("Every group requires all purchase-period and approval-slot cells.")
        rates = rows["late_rate_pct"].to_numpy(dtype=float).reshape(len(periods), len(SLOT_LABELS))
        counts = rows["orders"].to_numpy(dtype=float).reshape(len(periods), len(SLOT_LABELS))
        heatmap = ax.imshow(np.ma.masked_invalid(rates), cmap=color_map, vmin=0, vmax=maximum,
                            aspect="auto")
        for row in range(len(periods)):
            for column in range(len(SLOT_LABELS)):
                count = counts[row, column]
                rate = rates[row, column]
                label = (f"{rate:.1f}%\nN={int(count):,}" if np.isfinite(rate)
                         else "—\nN=0")
                ax.text(column, row, label, ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(rate) and rate > maximum * .58 else "#222222")
        ax.set_xticks(range(len(SLOT_LABELS)), SLOT_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(periods)), periods)
        ax.set_title(group)
    figure.colorbar(heatmap, ax=axes, label="Late orders / group orders (%)", shrink=.85)
    figure.suptitle("Late delivery by purchase period, destination group and approval slot")
    figure.supxlabel("N = all orders in each cell; gray cells have no orders")
    return figure


def plot_route_period_medians(table, value, title, ylabel):
    """Compare two destination cohorts by approval slot within three purchase periods."""
    periods = table["period"].drop_duplicates().tolist()
    groups = table["group"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(periods), figsize=(17, 5.5), sharey=True,
                               layout="constrained")
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    colors = ("#377eb8", "#388443")
    for ax, period in zip(axes, periods):
        for group_number, (group, color) in enumerate(zip(groups, colors)):
            rows = table.loc[table["period"].eq(period) & table["group"].eq(group)]
            rows = rows.set_index("approval_slot").reindex(SLOT_LABELS)
            values = rows[value].to_numpy(dtype=float)
            positions = np.arange(len(SLOT_LABELS)) + (group_number - .5) * .36
            ax.bar(positions, values, width=.36, color=color, label=group)
            for x, height, count in zip(positions, values, rows["orders"].fillna(0)):
                ax.annotate(f"N={int(count):,}", (x, height if np.isfinite(height) else 0),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", rotation=90, fontsize=7)
        ax.set_xticks(range(len(SLOT_LABELS)), SLOT_LABELS, rotation=35, ha="right")
        ax.set_title(period)
        ax.margins(y=.25)
    axes[0].set_ylabel(ylabel)
    if groups:
        axes[-1].legend(frameon=False, fontsize=8)
    figure.suptitle(title)
    figure.supxlabel("N = all orders in each purchase period, destination group and approval slot")
    return figure


def plot_state_pair_slots(raw, state_pair):
    """Show late counts and rates for all four approval slots of one state pair."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    rows = raw.set_index("approval_slot").reindex(SLOT_LABELS)
    positions = np.arange(len(SLOT_LABELS))
    for ax, field, color, title, ylabel in (
        (axes[0], "late_orders", "#377eb8", "Late order counts", "Late orders"),
        (axes[1], "late_rate_pct", "#388443", "Late delivery rates", "Late orders / slot orders (%)"),
    ):
        values = rows[field].to_numpy(dtype=float)
        ax.bar(positions, values, color=color, width=0.72)
        for position, (value, count) in enumerate(zip(values, rows["orders"])):
            ax.annotate(f"N={int(count):,}", (position, value if np.isfinite(value) else 0),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(positions, SLOT_LABELS)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.margins(y=0.2)
    figure.suptitle(f"{state_pair[0]} → {state_pair[1]} | Delivery outcomes by approval slot")
    figure.supxlabel("N = all orders in the state pair and approval slot")
    return figure


def plot_state_pair_months(monthly, state_pair):
    """Show monthly slot rates alongside their denominators without interpolation."""
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, layout="constrained")
    months = monthly["purchase_month"].drop_duplicates().tolist()
    x = np.arange(len(months))
    colors = ("#377eb8", "#e76f51", "#388443", "#9467bd")
    for slot, color in zip(SLOT_LABELS, colors):
        rows = monthly.loc[monthly["approval_slot"].eq(slot)].set_index("purchase_month").reindex(months)
        axes[0].plot(x, rows["late_rate_pct"].to_numpy(dtype=float), marker="o",
                     markersize=3, color=color, label=slot)
        axes[1].plot(x, rows["orders"].to_numpy(dtype=float), marker="o",
                     markersize=3, color=color, label=slot)
    axes[0].set_ylabel("Late orders / monthly slot orders (%)")
    axes[1].set_ylabel("Orders (N)")
    axes[0].set_title("Late delivery rate by purchase month")
    axes[1].set_title("Order count by purchase month")
    axes[1].set_xticks(x[::2], months[::2], rotation=45, ha="right")
    if months:
        axes[0].legend(ncols=4, frameon=False)
    figure.suptitle(f"{state_pair[0]} → {state_pair[1]} | Purchase month × approval slot")
    figure.supxlabel("Missing points indicate months with no orders in that slot; exact counts appear in the table")
    return figure


def plot_focus_months(monthly):
    """Show monthly late rates and order denominators for both focus scopes."""
    figure, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, layout="constrained")
    months = monthly["purchase_month"].drop_duplicates().tolist()
    for scope, color in zip(monthly["scope"].drop_duplicates(), ("#377eb8", "#388443")):
        rows = monthly.loc[monthly["scope"].eq(scope)].set_index("purchase_month").reindex(months)
        x = np.arange(len(months))
        for ax in axes[:2]:
            ax.plot(x, rows["late_rate_pct"].to_numpy(dtype=float), marker="o", markersize=3,
                    label=scope, color=color)
        axes[2].plot(x, rows["orders"].to_numpy(dtype=float), marker="o", markersize=3,
                     label=scope, color=color)
    recent = monthly.loc[monthly["purchase_month"].ge("2017-01"), "late_rate_pct"].dropna()
    detail_limit = min(100, max(10, np.ceil(recent.max() / 10) * 10)) if not recent.empty else 100
    axes[1].set_ylim(0, detail_limit)
    for ax in axes[:2]:
        ax.set_ylabel("Late orders / monthly orders (%)")
    axes[2].set_ylabel("Orders (N)")
    axes[0].set_title("Monthly late rate · full range")
    axes[1].set_title(f"Monthly late rate · detail to {detail_limit:.0f}%")
    axes[2].set_title("Order counts by purchase month")
    axes[2].set_xticks(range(0, len(months), 2), months[::2], rotation=45, ha="right")
    if months:
        axes[0].legend(frameon=False)
    scope_title = ("Inter-state orders and leading state pair" if monthly["scope"].nunique() > 1
                   else "Inter-state orders")
    figure.suptitle(f"{scope_title} | Approved 12:00–18:00")
    figure.supxlabel("Purchase month · No-order months have no defined late rate")
    return figure


def plot_focus_approval_time(hours, weekdays):
    """Show late counts, rates and denominators by approval hour and weekday."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained")
    for row, (table, key, labels, period) in enumerate((
        (hours, "approval_hour", [f"{hour:02d}:00" for hour in range(12, 18)], "approval hour"),
        (weekdays, "approval_weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
         "approval weekday"),
    )):
        scopes = table["scope"].drop_duplicates().tolist()
        positions = np.arange(len(labels))
        width = 0.8 / max(len(scopes), 1)
        for offset, (scope, color) in enumerate(zip(scopes, ("#377eb8", "#388443"))):
            values = table.loc[table["scope"].eq(scope)].sort_values(key)
            x = positions + (offset - (len(scopes) - 1) / 2) * width
            axes[row, 0].bar(x, values["late_orders"], width=width, color=color, label=scope)
            rates = values["late_rate_pct"].to_numpy(dtype=float)
            axes[row, 1].bar(x, rates, width=width, color=color, label=scope)
            for position, rate, orders in zip(x, rates, values["orders"]):
                if np.isfinite(rate):
                    axes[row, 1].annotate(f"N={int(orders):,}", (position, rate),
                                          xytext=(0, 3), textcoords="offset points",
                                          rotation=90, ha="center", va="bottom", fontsize=7)
        for ax in axes[row]:
            ax.set_xticks(positions, labels)
        axes[row, 0].set_title(f"Late order counts by {period}")
        axes[row, 1].set_title(f"Late delivery rates by {period}")
        axes[row, 0].set_ylabel("Late orders")
        axes[row, 1].set_ylabel("Late orders / group orders (%)")
        axes[row, 1].set_ylim(0, max(table["late_rate_pct"].max(skipna=True) * 1.45, 1)
                              if not table.empty and table["late_rate_pct"].notna().any() else 1)
    if not hours.empty:
        axes[0, 0].legend(frameon=False)
    scope_title = ("Inter-state orders and leading state pair" if hours["scope"].nunique() > 1
                   else "Inter-state orders")
    figure.suptitle(f"{scope_title} | Approved 12:00–18:00")
    figure.supxlabel("N = all orders in the scope and approval-time group")
    return figure


def plot_state_pairs(pairs, inter_late_rate, top_n=5):
    """Plot leading seller-to-customer state pairs by late inter-state orders."""
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    rows = pairs.loc[pairs["late_orders"].gt(0)].sort_values(
        ["late_orders", "orders", "seller_state", "customer_state"],
        ascending=[False, False, True, True],
    ).head(top_n)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True, layout="constrained")
    figure.suptitle("Leading seller → customer state pairs | Inter-state orders approved 12:00–18:00")
    if rows.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No late orders in eligible state pairs", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_axis_off()
        return figure
    y = np.arange(len(rows))
    for ax, column, color, title, xlabel in zip(
        axes, ("late_orders", "late_rate_pct"), ("#377eb8", "#388443"),
        ("Late order counts", "Late delivery rates"),
        ("Late orders", "Late orders / pair orders (%)"),
    ):
        values = rows[column].to_numpy(dtype=float)
        ax.barh(y, values, color=color)
        for position, value in enumerate(values):
            label = f"{value:,.0f}" if column == "late_orders" else f"{value:.2f}%"
            ax.annotate(label, (value, position), xytext=(5, 0), textcoords="offset points", va="center")
        ax.set_xlim(0, max(values) * 1.3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    if np.isfinite(inter_late_rate):
        axes[1].set_xlim(0, max(rows["late_rate_pct"].max(), inter_late_rate) * 1.3)
        axes[1].axvline(inter_late_rate, color="#555555", linestyle="--", linewidth=1.2,
                        label=f"All inter-state orders: {inter_late_rate:.2f}%")
        axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_yticks(y, [f"{row.seller_state} → {row.customer_state} (N={int(row.orders):,})"
                           for row in rows.itertuples()])
    axes[0].invert_yaxis()
    figure.supxlabel("Ranked by late order count. N = all orders in each state pair.\n"
                     "The reference rate includes these pairs and all other inter-state orders.", fontsize=9)
    return figure


def plot_destination_states(inter_state, top_n=5):
    """Plot the leading known destinations by late count in the focus inter-state cohort."""
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    rows = inter_state.loc[
        inter_state["customer_state"].ne(REVIEW_DESTINATION) & inter_state["late_orders"].gt(0)
    ].sort_values(["late_orders", "orders", "customer_state"], ascending=[False, False, True]).head(top_n)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True, layout="constrained")
    figure.suptitle("Leading customer states by late inter-state orders | Approved 12:00–18:00")
    if rows.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No late orders with a known destination", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_axis_off()
        return figure
    y = np.arange(len(rows))
    for ax, column, color, title, xlabel in zip(
        axes, ("late_orders", "late_rate_pct"), ("#377eb8", "#388443"),
        ("Late order counts", "Late delivery rates"),
        ("Late orders", "Late orders / destination orders (%)"),
    ):
        values = rows[column].to_numpy(dtype=float)
        ax.barh(y, values, color=color)
        for position, value in enumerate(values):
            label = f"{value:,.0f}" if column == "late_orders" else f"{value:.2f}%"
            ax.annotate(label, (value, position), xytext=(5, 0), textcoords="offset points", va="center")
        ax.set_xlim(0, max(values) * 1.3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(y, [f"{row.customer_state} (N={int(row.orders):,})" for row in rows.itertuples()])
    axes[0].invert_yaxis()
    figure.supxlabel("Ranked by late order count. N = all inter-state orders to that state in the approval slot.\n"
                     "Destinations requiring review are listed in the table and excluded from this ranking.", fontsize=9)
    return figure


def plot_route_slot_rates(slots):
    """Compare intra/inter late rates with each cell's order denominator."""
    figure, ax = plt.subplots(figsize=(12, 5.8), layout="constrained")
    for position, (route, color) in enumerate((("Intra-state", "#377eb8"),
                                                ("Inter-state", "#e76f51"))):
        rows = slots.loc[slots["route_type"].eq(route)].set_index("approval_slot").reindex(SLOT_LABELS)
        x = np.arange(len(SLOT_LABELS)) + (position - 0.5) * 0.36
        values = rows["late_rate_pct"].to_numpy(dtype=float, na_value=np.nan)
        ax.bar(x, values, width=0.36, label=route, color=color)
        for i, row in enumerate(rows.itertuples()):
            if int(row.orders) > 0 and np.isfinite(values[i]):
                ax.annotate(f"N={int(row.orders):,}", (x[i], values[i]),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
            else:
                ax.annotate("N=0", (x[i], 0), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(len(SLOT_LABELS)), SLOT_LABELS)
    ax.set_xlabel("Approval time slot")
    ax.set_ylabel("Late deliveries (% of orders)")
    ax.set_title("Intra-state and inter-state late delivery rates by approval slot")
    figure.supxlabel("N = orders in each route and slot. Routes compare seller and customer states across all items.",
                     fontsize=9)
    ax.margins(y=0.18)
    ax.legend(frameon=False)
    return figure


def plot_route_transit_time(transit):
    """Compare median carrier-to-customer hours for known single-route groups."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True, layout="constrained")
    routes = ("Intra-state", "Inter-state")
    rows = transit.set_index("route_type").reindex(routes)
    colors = ("#377eb8", "#e76f51")
    for ax, column, title in zip(
        axes, ("median_calendar_hours", "median_eligible_hours"),
        ("Elapsed transit time", "Transit time on eligible days"),
    ):
        values = rows[column].to_numpy(dtype=float, na_value=np.nan)
        ax.bar(routes, values, color=colors, width=0.55)
        for i, row in enumerate(rows.itertuples()):
            count = int(row.evaluated)
            if count and np.isfinite(values[i]):
                ax.annotate(f"N={count:,}", (i, values[i]), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8)
            else:
                ax.annotate("N=0", (i, 0), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Median hours per evaluated order")
        ax.margins(y=0.18)
    figure.suptitle("Median time from carrier handoff to delivery by route type")
    figure.supxlabel("N = orders with a valid transit interval. " + ELIGIBLE_DAY_NOTE, fontsize=9)
    return figure


def plot_monthly_slot_rates(monthly):
    """Pair monthly rates with counts, including empty calendar months."""
    figure, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, layout="constrained")
    months = monthly["month"].drop_duplicates().tolist()
    for slot in SLOT_LABELS:
        rows = monthly.loc[monthly["approval_slot"].eq(slot)]
        axes[0].plot(range(len(rows)), rows["late_rate_pct"].to_numpy(dtype=float), marker=".", label=slot)
        axes[1].plot(range(len(rows)), rows["late_rate_pct"].to_numpy(dtype=float), marker=".")
        axes[2].plot(range(len(rows)), rows["orders"], marker=".")
    axes[0].set_ylabel("Late orders / slot orders (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(ncols=4, frameon=False)
    figure.suptitle("Delivery outcomes by purchase month and approval slot")
    axes[0].set_title("Monthly late delivery rates (full range)")
    axes[1].set_ylim(0, 25)
    axes[1].set_ylabel("Late rate (%)")
    axes[1].set_title("Monthly late delivery rates (0–25% scale)")
    axes[2].set_ylabel("Orders in slot")
    axes[2].set_title("Monthly order counts by approval slot")
    axes[2].set_xlabel("Purchase month")
    axes[2].set_xticks(range(len(months)), months, rotation=65)
    return figure


def plot_matched_years(table):
    """Compare January-July in 2017 and 2018 with visible denominators."""
    figure, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for position, year in enumerate((2017, 2018)):
        rows = table.loc[table["year"].eq(year)].set_index("approval_slot").reindex(SLOT_LABELS)
        x = np.arange(4) + (position - 0.5) * 0.36
        ax.bar(x, rows["late_rate_pct"].to_numpy(dtype=float, na_value=np.nan), width=0.36, label=str(year))
        for i, row in enumerate(rows.itertuples()):
            height = row.late_rate_pct if row.orders else 0
            ax.annotate(f"N={row.orders:,}", (x[i], height),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(4), SLOT_LABELS)
    ax.set_xlabel("Approval time slot")
    ax.set_ylabel("Late orders / slot orders (%)")
    ax.set_title("Late delivery rate by approval slot for January–July purchases in 2017 and 2018")
    figure.supxlabel("N = orders in each approval slot during the seven purchase months of each year",
                     fontsize=9)
    ax.margins(y=0.18)
    ax.legend(frameon=False)
    return figure


def plot_stage_comparison(summary, stage):
    """Compare late and on-time medians using both elapsed-hour conventions."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True, layout="constrained")
    for ax, column, title in zip(axes, ("median_calendar_hours", "median_eligible_hours"),
                                ("Recorded elapsed hours", "Eligible-day hours")):
        for position, late in enumerate((False, True)):
            rows = summary.loc[summary["stage"].eq(stage) & summary["is_late"].eq(late)]
            rows = rows.set_index("approval_slot").reindex(SLOT_LABELS)
            x = np.arange(4) + (position - 0.5) * 0.36
            values = rows[column].to_numpy(dtype=float, na_value=np.nan)
            ax.bar(x, values, width=0.36, label="Late" if late else "On time",
                   color="#e76f51" if late else "#377eb8")
            for i, row in enumerate(rows.itertuples()):
                if np.isfinite(values[i]):
                    ax.annotate(f"N={int(row.evaluated):,}", (x[i], values[i]),
                                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        ax.set_title(title)
        ax.set_xticks(range(4), SLOT_LABELS, rotation=20)
        ax.set_xlabel("Approval time slot")
        ax.set_ylabel("Median hours per evaluated order")
        ax.margins(y=0.20)
        ax.legend(frameon=False)
    interval = {"Processing": "approval to carrier handoff",
                "Carrier transit": "carrier handoff to customer delivery"}.get(stage, stage)
    figure.suptitle(f"Median time from {interval} by approval slot and delivery outcome")
    figure.supxlabel("N = orders with a valid interval. " + ELIGIBLE_DAY_NOTE, fontsize=9)
    return figure


def plot_promised_windows(windows):
    """Show full calendar-date promise windows by approval slot."""
    figure, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    samples = [windows.loc[windows["approval_slot"].eq(slot), "promised_days"].dropna().to_numpy(dtype=float)
               for slot in SLOT_LABELS]
    if any(len(values) for values in samples):
        ax.boxplot(samples, tick_labels=SLOT_LABELS, showmeans=True,
                   meanprops={"marker": "D", "markerfacecolor": "#e76f51", "markeredgecolor": "#e76f51"})
    else:
        ax.text(0.5, 0.5, "No analyzable orders", transform=ax.transAxes, ha="center")
    ax.set_ylabel("Estimated delivery date minus approval date (days)")
    ax.set_xlabel("Approval time slot")
    ax.set_title("Days from approval to the promised delivery date by approval slot")
    figure.supxlabel("Box line = median     Diamond = mean     Negative values indicate approval after the promised date.",
                     fontsize=9)
    return figure
