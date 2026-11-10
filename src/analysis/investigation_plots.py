"""Figures for temporal stability, stage times and delivery promises."""

import matplotlib.pyplot as plt
import numpy as np

from .orders import SLOT_LABELS
from .investigation import REVIEW_DESTINATION


ELIGIBLE_DAY_NOTE = "Eligible hours count Mon–Fri, excluding 1 Jan, 1 May and 25 Dec (24 hours per full eligible day)."


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
