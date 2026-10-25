"""Reproducible cleaning review tables and figures without display or file I/O."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from src.cleaning.measurements import PHYSICAL_COLUMNS
from src.cleaning.presentation import COLUMN_LABELS, EXCLUSION_LABELS
from src.cleaning.validation import validate_cleaning_output
from src.eda.checks import build_entity_views
from src.eda.plots import plot_frequency, plot_numeric_histogram
from src.eda.profiling import limit_frequencies, profile_frequencies, profile_numeric, require_columns


BEFORE_COLOR = "#377eb8"
AFTER_COLOR = "#2a9d8f"
REMOVED_COLOR = "#e76f51"


def plot_filtering_flow(flow: pd.DataFrame) -> Figure:
    """Plot retained/excluded orders and disjoint, sequential exclusion counts."""
    require_columns(flow, ["step", "removed_orders", "remaining_orders"])
    if flow.empty:
        raise ValueError("Filtering flow requires its input row.")
    before = int(flow.iloc[0]["remaining_orders"])
    after = int(flow.iloc[-1]["remaining_orders"])
    removed = flow.iloc[1:]["removed_orders"].to_numpy(dtype=float)
    if before < 0 or after < 0 or after > before or (removed < 0).any() or removed.sum() != before - after:
        raise ValueError("Sequential order counts must reconcile with retained totals.")
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.5), layout="constrained",
                                gridspec_kw={"height_ratios": [1, 3]})
    axes[0].barh([0], [after], height=0.35, color=AFTER_COLOR, label="Retained")
    axes[0].barh([0], [before - after], left=[after], height=0.35, color=REMOVED_COLOR, label="Excluded")
    axes[0].set_xlim(0, max(before, 1))
    axes[0].set_ylim(-0.55, 0.55)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Orders")
    rate = f"{100 * after / before:.2f}%" if before else "N/A"
    axes[0].set_title(f"Input: {before:,} orders | Retained: {after:,} ({rate}) | Excluded: {before-after:,}")
    axes[0].legend(loc="lower left", ncols=2, frameon=False)
    labels = [EXCLUSION_LABELS.get(value, value) for value in flow.iloc[1:]["step"]]
    positions = np.arange(len(labels))
    axes[1].barh(positions, removed, color=REMOVED_COLOR)
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    maximum = max(float(removed.max()) if len(removed) else 0, 1)
    axes[1].set_xlim(0, maximum * 1.2)
    for y, count in zip(positions, removed):
        axes[1].text(count + maximum * 0.015, y, f"{int(count):,}", va="center")
    axes[1].set_xlabel("Newly excluded orders; each order counted once")
    axes[1].set_title("Exclusions by first applicable rule")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].set_axisbelow(True)
    figure.suptitle("Cleaning outcome at order grain", fontsize=14)
    return figure


def profile_missingness_comparison(source: pd.DataFrame, cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all-column counts/rates and a compact display of observed missingness.

    Percentages use each table's own item denominator. Group physical fields
    only when their before/after counts match; grouped counts remain per field,
    not the sum across fields. Undefined zero-row percentages stay NaN.
    """
    records = []
    for column in source.columns:
        before, after = int(source[column].isna().sum()), int(cleaned[column].isna().sum())
        records.append([column, before, after,
                        100 * before / len(source) if len(source) else np.nan,
                        100 * after / len(cleaned) if len(cleaned) else np.nan])
    full = pd.DataFrame(records, columns=[
        "column", "missing_before", "missing_after", "missing_pct_before", "missing_pct_after",
    ])
    selected = full.loc[full["missing_before"].gt(0) | full["missing_after"].gt(0)]
    groups = {}
    for row in selected.itertuples(index=False):
        key = ("physical", row.missing_before, row.missing_after) if row.column in PHYSICAL_COLUMNS else (row.column,)
        groups.setdefault(key, []).append(row)
    compact = []
    for rows in groups.values():
        first = rows[0]
        columns = tuple(row.column for row in rows)
        if len(rows) == len(PHYSICAL_COLUMNS):
            label = f"Physical measurements ({len(rows)} columns, each)"
        elif len(rows) > 1:
            names = [COLUMN_LABELS[column].removeprefix("Product ") for column in columns]
            label = " / ".join(names) + " (each)"
        else:
            label = COLUMN_LABELS.get(first.column, first.column)
        compact.append([label, columns, len(rows), first.missing_before, first.missing_after,
                        first.missing_pct_before, first.missing_pct_after])
    display = pd.DataFrame(compact, columns=[
        "attribute", "columns", "column_count", "missing_before", "missing_after",
        "missing_pct_before", "missing_pct_after",
    ])
    return full, display


def plot_missingness_comparison(table: pd.DataFrame, before_rows: int, after_rows: int) -> Figure:
    """Plot missing percentages with original counts and explicit denominators."""
    figure, axes = plt.subplots(figsize=(10.5, max(4.2, len(table) * 0.72 + 1.8)), layout="constrained")
    if table.empty:
        message = "No missing values in either table" if after_rows else "No input missingness; retained table has no rows"
        axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
        axes.set_yticks([])
    else:
        positions = np.arange(len(table))
        offsets = [-0.18, 0.18]
        maximum = max(float(table[["missing_pct_before", "missing_pct_after"]].max().max()), 0.1)
        for suffix, offset, color, label in zip(
            ["before", "after"], offsets, [BEFORE_COLOR, AFTER_COLOR], ["Before", "After"],
        ):
            rates = table[f"missing_pct_{suffix}"].to_numpy(dtype=float)
            axes.barh(positions + offset, np.nan_to_num(rates), height=0.32, color=color, label=label)
            for y, rate, count in zip(positions + offset, rates, table[f"missing_{suffix}"]):
                text = "N/A (0 rows)" if np.isnan(rate) else f"{rate:.4f}% ({int(count):,})"
                axes.text((rate if np.isfinite(rate) else 0) + maximum * 0.015, y, text, va="center", fontsize=8)
        axes.set_yticks(positions, table["attribute"])
        axes.invert_yaxis()
        axes.set_xlim(0, maximum * 1.5)
        axes.legend(frameon=False, loc="lower right")
    axes.set_xlabel("Missing item values (%) | labels: percentage (count)")
    axes.set_title(f"Missingness before and after cleaning\nItem rows: before N={before_rows:,}; after N={after_rows:,}")
    axes.grid(axis="x", alpha=0.2)
    axes.set_axisbelow(True)
    return figure


def plot_retained_numeric(cleaned: pd.DataFrame, products: pd.DataFrame, numeric: pd.DataFrame) -> Figure:
    """Show full ranges and 1st-99th percentile display windows; retain all data."""
    specifications = [
        (cleaned["price"], "Price (BRL)", "item"),
        (cleaned["freight_value"], "Freight (BRL)", "item"),
        (products["product_weight_g"], "Product weight (g)", "product"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    indexed = numeric.set_index("column")
    for position, (series, label, grain) in enumerate(specifications):
        row = indexed.loc[series.name]
        low, high = row["p01"], row["p99"]
        bounds = (float(low), float(high)) if pd.notna(low) and pd.notna(high) and low < high else None
        plot_numeric_histogram(series, bins=35, xlabel=label, title=f"Full range | {grain} grain", ax=axes[0, position])
        plot_numeric_histogram(
            series, bins=35, value_range=bounds, xlabel=label,
            title=f"1st-99th percentile window | {grain} grain" if bounds else f"Full range (no percentile span) | {grain} grain",
            ax=axes[1, position],
        )
    figure.suptitle("Retained numeric distributions | windows affect display only", fontsize=14)
    return figure


def profile_retained_calendar(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count purchase months, including calendar gaps, and approved clock hours."""
    periods = orders["order_purchase_timestamp"].dt.to_period("M")
    if periods.empty:
        months = pd.DataFrame({"month": pd.Series(dtype="string"), "orders": pd.Series(dtype="int64")})
    else:
        span = pd.period_range(periods.min(), periods.max(), freq="M")
        counts = periods.value_counts().reindex(span, fill_value=0)
        months = pd.DataFrame({"month": span.astype(str), "orders": counts.to_numpy(dtype=int)})
    counts = orders["order_approved_at"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    hours = pd.DataFrame({"hour": range(24), "orders": counts.to_numpy(dtype=int)})
    return months, hours


def plot_retained_calendar(months: pd.DataFrame, hours: pd.DataFrame) -> Figure:
    """Plot observed order volumes without generating delivery outcome features."""
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
    total = int(hours["orders"].sum())
    if months.empty:
        axes[0].text(0.5, 0.5, "No retained orders", ha="center", va="center", transform=axes[0].transAxes)
    else:
        axes[0].bar(range(len(months)), months["orders"], color=BEFORE_COLOR)
        ticks = sorted(set(range(0, len(months), max(1, int(np.ceil(len(months) / 10))))) | {len(months) - 1})
        axes[0].set_xticks(ticks, months.iloc[ticks]["month"], rotation=45, ha="right")
    axes[0].set_title(f"Purchase month | order grain | N={total:,}")
    axes[0].set_xlabel("Purchase month; gaps have zero counts")
    axes[1].bar(hours["hour"], hours["orders"], color=AFTER_COLOR)
    axes[1].set_xticks([0, 4, 8, 12, 16, 20, 23])
    axes[1].set_title(f"Approval clock hour | order grain | N={total:,}")
    axes[1].set_xlabel("Stored approval hour (0-23)")
    for ax in axes:
        ax.set_ylabel("Orders")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    figure.suptitle("Retained temporal coverage", fontsize=14)
    return figure


def _plot_frequency_review(full: pd.DataFrame, title: str, ax=None):
    """Reuse EDA frequency accounting with visible counts and special groups."""
    shown = limit_frequencies(full, top_n=10)
    if int(shown["count"].sum()):
        shown = shown.loc[shown["count"].gt(0)].copy()
    figure, axes = plot_frequency(shown, top_n=None, title=title, ax=ax)
    if ax is None:
        figure.set_size_inches(10, 6.2)
    maximum = max(int(shown["count"].max()), 1)
    for bar, row in zip(axes.patches, shown.itertuples(index=False)):
        if row.kind in {"other", "missing"}:
            bar.set_facecolor("#9aa0a6")
        axes.text(bar.get_width() + maximum * 0.015, bar.get_y() + bar.get_height()/2,
                  f"{int(row.count):,}", va="center", fontsize=8)
    axes.set_xlim(0, maximum * 1.2)
    axes.set_title(f"{title}\nN={int(shown.iloc[0]['total_rows']):,}; counts include displayed special groups", fontsize=10)
    return figure, shown


def build_cleaning_visuals(
    source: pd.DataFrame,
    cleaned: pd.DataFrame,
    details: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Build six reproducible figures and their full/compact data tables.

    Verify the retained cohort against source before profiling. Item amounts,
    order timestamps and unique product/customer/seller attributes use their
    proper grain. Return figures for caller-controlled display/close/export;
    never save files, update documents, create outcome features or mutate data.
    """
    validate_cleaning_output(source, cleaned)
    flow = details["filter_flow"]
    if (int(flow.iloc[0]["remaining_orders"]) != source["order_id"].nunique()
            or int(flow.iloc[-1]["remaining_orders"]) != cleaned["order_id"].nunique()):
        raise ValueError("Filtering diagnostics must match the supplied input and output.")
    views = build_entity_views(cleaned)
    missingness, missing_display = profile_missingness_comparison(source, cleaned)
    numeric = pd.concat([
        profile_numeric(cleaned, ["price", "freight_value"]).assign(grain="item", unit="BRL"),
        profile_numeric(views["products"], list(PHYSICAL_COLUMNS)).assign(grain="product", unit=["g", "cm", "cm", "cm"]),
    ], ignore_index=True)
    months, hours = profile_retained_calendar(views["orders"])
    categories = profile_frequencies(views["products"]["product_category_name"])
    customer_states = profile_frequencies(views["customers"]["customer_state"])
    seller_states = profile_frequencies(views["sellers"]["seller_state"])
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9}):
        figures = {
            "filtering": plot_filtering_flow(flow),
            "missingness": plot_missingness_comparison(missing_display, len(source), len(cleaned)),
            "numeric": plot_retained_numeric(cleaned, views["products"], numeric),
            "calendar": plot_retained_calendar(months, hours),
        }
        figures["categories"], category_display = _plot_frequency_review(categories, "Product categories | product grain | top 10 plus special groups")
        geo, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), layout="constrained")
        _, customer_display = _plot_frequency_review(customer_states, "Customer state | customer-record grain", axes[0])
        _, seller_display = _plot_frequency_review(seller_states, "Seller state | seller grain", axes[1])
        geo.suptitle("Retained geographic composition | top 10 plus Other", fontsize=14)
        figures["geography"] = geo
    return {"figures": figures, "tables": {
        "missingness": missingness, "missingness_display": missing_display,
        "numeric": numeric, "purchase_months": months, "approval_hours": hours,
        "categories": categories, "category_display": category_display,
        "customer_states": customer_states, "customer_state_display": customer_display,
        "seller_states": seller_states, "seller_state_display": seller_display,
    }}
