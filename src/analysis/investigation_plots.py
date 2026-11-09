"""Figures for temporal stability, stage times and delivery promises."""

import matplotlib.pyplot as plt
import numpy as np

from .orders import SLOT_LABELS


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
    axes[0].set_title("Late rates by purchase month and approval slot")
    axes[1].set_ylim(0, 25)
    axes[1].set_ylabel("Late rate (%)")
    axes[1].set_title("Rate detail, 0-25% display window; full range above")
    axes[2].set_ylabel("Orders in slot")
    axes[2].set_xlabel("Purchase month")
    axes[2].set_xticks(range(len(months)), months, rotation=65)
    return figure


def plot_matched_years(table):
    """Compare January-July in 2017 and 2018 with visible denominators."""
    figure, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    for position, year in enumerate((2017, 2018)):
        rows = table.loc[table["year"].eq(year)].set_index("approval_slot").reindex(SLOT_LABELS)
        x = np.arange(4) + (position - 0.5) * 0.36
        ax.bar(x, rows["late_rate_pct"].fillna(0).to_numpy(dtype=float), width=0.36, label=str(year))
        for i, row in enumerate(rows.itertuples()):
            if row.orders:
                ax.annotate(f"N={row.orders:,}", (x[i], row.late_rate_pct),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(4), SLOT_LABELS)
    ax.set_xlabel("Approval time slot")
    ax.set_ylabel("Late orders / slot orders (%)")
    ax.set_title("Same purchase months, January-July")
    ax.margins(y=0.18)
    ax.legend(frameon=False)
    return figure


def plot_stage_comparison(summary, stage):
    """Compare late and on-time medians using both elapsed-hour conventions."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
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
        ax.set_ylabel("Median hours per evaluated order")
        ax.margins(y=0.20)
        ax.legend(frameon=False)
    figure.suptitle(stage)
    return figure


def plot_promised_windows(windows):
    """Show full calendar-date promise windows by approval slot."""
    figure, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    samples = [windows.loc[windows["approval_slot"].eq(slot), "promised_days"].dropna().to_numpy(dtype=float)
               for slot in SLOT_LABELS]
    if any(len(values) for values in samples):
        ax.boxplot(samples, tick_labels=SLOT_LABELS, showmeans=True)
    else:
        ax.text(0.5, 0.5, "No analyzable orders", transform=ax.transAxes, ha="center")
    ax.set_ylabel("Estimated delivery date minus approval date (days)")
    ax.set_xlabel("Approval time slot")
    ax.set_title("Promised delivery window at approval, full range")
    return figure
