"""Order-level delivery profiles and figures without file I/O."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .orders import SLOT_LABELS


ON_TIME_COLOR = "#377eb8"
LATE_COLOR = "#e76f51"
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def build_temporal_profiles(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Count orders and late orders by hour, weekday and their intersection."""
    required = {"order_id", "approval_hour", "approval_weekday", "is_late"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Missing temporal columns: {sorted(missing)}")
    if orders["order_id"].duplicated().any():
        raise ValueError("Temporal profiles require one row per order_id.")

    index = pd.MultiIndex.from_product(
        [range(7), range(24)], names=["approval_weekday", "approval_hour"]
    )
    cells = orders.groupby(["approval_weekday", "approval_hour"]).agg(
        orders=("order_id", "size"), late_orders=("is_late", "sum")
    ).reindex(index, fill_value=0).astype("int64")
    hours = cells.groupby(level="approval_hour").sum().reindex(range(24))
    weekdays = cells.groupby(level="approval_weekday").sum().reindex(range(7))

    profiles = {}
    for name, frame in (("hours", hours), ("weekdays", weekdays), ("hour_weekday", cells)):
        frame = frame.reset_index()
        frame["late_rate_pct"] = np.divide(
            100 * frame["late_orders"], frame["orders"],
            out=np.full(len(frame), np.nan), where=frame["orders"].ne(0)
        )
        profiles[name] = frame
    return profiles


def plot_slot_counts(summary: pd.DataFrame) -> Figure:
    """Show on-time and late counts without hiding the slot denominators."""
    figure, ax = plt.subplots(figsize=(10.5, 5.2), layout="constrained")
    x = np.arange(len(summary))
    ax.bar(x, summary["on_time_orders"], color=ON_TIME_COLOR, label="On time")
    ax.bar(x, summary["late_orders"], bottom=summary["on_time_orders"],
           color=LATE_COLOR, label="Late")
    for position, row in enumerate(summary.itertuples(index=False)):
        ax.annotate(f"Total {row.orders:,}\nLate {row.late_orders:,}",
                    (position, row.orders), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, summary["approval_slot"])
    ax.set_ylabel("Orders")
    ax.set_xlabel("Approval time slot")
    ax.set_title("On-time and late deliveries by approval slot")
    ax.set_ylim(0, max(summary["orders"].max() * 1.16, 1))
    ax.legend(frameon=False)
    return figure


def plot_rate_profile(profile: pd.DataFrame, column: str) -> Figure:
    """Show hourly or weekday late rates with each group's order count."""
    if column not in ("approval_hour", "approval_weekday"):
        raise ValueError("Profile column must be approval_hour or approval_weekday.")
    hourly = column == "approval_hour"
    figure, ax = plt.subplots(
        figsize=(14, 5.6) if hourly else (10.5, 5.2), layout="constrained"
    )
    x = profile[column].to_numpy()
    rates = profile["late_rate_pct"].to_numpy(dtype=float, na_value=np.nan)
    ax.bar(x, rates, color=ON_TIME_COLOR, width=0.78)
    for position, row in enumerate(profile.itertuples(index=False)):
        height = rates[position] if row.orders else 0
        ax.annotate(f"N={row.orders:,}", (x[position], height),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8, rotation=90 if hourly else 0)
    ax.set_xticks(x, [f"{value:02d}" for value in x] if hourly else WEEKDAYS)
    ax.set_xlabel("Approval hour (recorded clock)" if hourly else "Approval weekday")
    ax.set_ylabel("Late deliveries (% of orders)")
    ax.set_title("Late delivery rate by approval hour" if hourly else
                 "Late delivery rate by approval weekday")
    figure.supxlabel("N = orders approved in each hour" if hourly else
                     "N = orders approved on each weekday", fontsize=9)
    ax.set_ylim(0, max(np.nanmax(profile["late_rate_pct"].to_numpy(dtype=float)) *
                       (1.35 if hourly else 1.20), 1) if profile["orders"].sum() else 1)
    return figure


def plot_hour_weekday_heatmap(cells: pd.DataFrame) -> Figure:
    """Pair each late-rate cell with the corresponding order-count cell."""
    rate = cells.pivot(index="approval_weekday", columns="approval_hour", values="late_rate_pct")
    counts = cells.pivot(index="approval_weekday", columns="approval_hour", values="orders")
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), layout="constrained")
    for ax, values, title, label, cmap in (
        (axes[0], rate, "Late delivery rate by approval hour and weekday",
         "Late deliveries (%)", "Greens"),
        (axes[1], counts.where(counts.gt(0)), "Orders approved by hour and weekday",
         "Orders", "Blues"),
    ):
        palette = plt.colormaps[cmap].with_extremes(bad="#eeeeee")
        image = ax.imshow(np.ma.masked_invalid(values.to_numpy(dtype=float)),
                          cmap=palette, aspect="auto", origin="upper", vmin=0)
        ax.set_title(title)
        ax.set_yticks(range(7), WEEKDAYS)
        ax.set_xticks(range(24), [f"{hour:02d}" for hour in range(24)])
        ax.set_ylabel("Approval weekday")
        ax.set_xlabel("Approval hour (recorded clock)")
        figure.colorbar(image, ax=ax, label=label)
    figure.supxlabel("Grey cells have no orders. Rates use all orders in each cell.", fontsize=9)
    return figure


def plot_delay_distribution(orders: pd.DataFrame) -> Figure:
    """Compare late-only delays at full range and within a labelled zoom."""
    late = orders.loc[orders["is_late"], ["approval_slot", "delay_days"]]
    samples = [late.loc[late["approval_slot"].eq(slot), "delay_days"].dropna().to_numpy(dtype=float)
               for slot in SLOT_LABELS]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    observed = [values for values in samples if len(values)]
    titles = ("Days late across the full distribution", "Days late up to the pooled 99th percentile")
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Approval time slot")
        ax.set_ylabel("Calendar days past the promised date")
    if not observed:
        for ax in axes:
            ax.text(0.5, 0.5, "No late orders", ha="center", va="center", transform=ax.transAxes)
    else:
        for ax in axes:
            ax.boxplot(samples, tick_labels=SLOT_LABELS, showmeans=True,
                       meanprops={"marker": "D", "markerfacecolor": LATE_COLOR,
                                  "markeredgecolor": LATE_COLOR})
            ax.tick_params(axis="x", rotation=25)
        p99 = float(np.percentile(np.concatenate(observed), 99))
        axes[1].set_ylim(0, p99)
        figure.supxlabel("Box line = median     Diamond = mean     Circles = values beyond the whiskers",
                         fontsize=9)
    figure.suptitle("Delay duration by approval time slot (late orders only)")
    return figure
