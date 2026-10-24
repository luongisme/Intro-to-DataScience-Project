"""Matplotlib distribution plots without automatic display, saving or styles."""

from numbers import Real

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from src.eda.profiling import finite_numeric_values, limit_frequencies, require_columns


def _get_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Use a caller's axes or create one without changing global styles."""
    if ax is None:
        return plt.subplots(figsize=figsize, layout="constrained")
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib Axes or None.")
    return ax.figure, ax


def plot_frequency(
    table: pd.DataFrame,
    *,
    top_n: int | None = 15,
    horizontal: bool = True,
    title: str = "Value frequencies",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a frequency table using absolute counts and an explicit row total.

    Use top_n for value-ranked categories with Other/Blank/Missing accounting.
    With top_n=None, keep every supplied row in its supplied order, useful for
    calendar distributions or an already limited display table. Special kinds
    have bracketed labels; literal string values retain repr() quoting so a
    category named Missing/Other cannot be confused with the special rows.
    Return (Figure, Axes); caller controls display/close/save. Input is unchanged.
    """
    require_columns(table, ["value", "kind", "count", "total_rows"])
    shown = limit_frequencies(table, top_n) if top_n is not None else table.copy()
    if shown.empty or not shown["kind"].isin(["value", "blank", "missing", "other"]).all():
        raise ValueError("Use a frequency table with supported value kinds.")
    totals = shown["total_rows"].unique()
    if (
        len(totals) != 1 or not pd.api.types.is_integer_dtype(shown["count"].dtype)
        or shown["count"].isna().any() or shown["count"].lt(0).any()
        or shown["count"].sum() != totals[0]
    ):
        raise ValueError("Frequency counts must sum to one consistent total_rows value.")
    total = int(totals[0])
    special_labels = {"other": "[Other values]", "blank": "[Blank strings]", "missing": "[Missing]"}
    labels = [
        special_labels[kind] if kind != "value" else repr(value) if isinstance(value, str) else str(value)
        for value, kind in zip(shown["value"], shown["kind"])
    ]
    figure, axes = _get_axes(ax, (8, max(3, min(12, len(shown) * 0.35)) if horizontal else 4))
    if total:
        positions = np.arange(len(shown))
        if horizontal:
            axes.barh(positions, shown["count"].to_numpy(), color="#377eb8")
            axes.set_yticks(positions, labels)
            axes.invert_yaxis()
            axes.set_xlabel("Count")
            axes.grid(axis="x", alpha=0.2)
        else:
            axes.bar(positions, shown["count"].to_numpy(), color="#377eb8")
            axes.set_xticks(positions, labels, rotation=90, fontsize=8)
            axes.set_ylabel("Count")
            axes.grid(axis="y", alpha=0.2)
    else:
        axes.text(0.5, 0.5, "No observations", ha="center", va="center", transform=axes.transAxes)
    axes.set_title(f"{title}\nN={total:,}; percentages in table use all rows", fontsize=10)
    axes.set_axisbelow(True)
    return figure, axes


def plot_numeric_histogram(
    series: pd.Series,
    *,
    bins: int = 40,
    value_range: tuple[float, float] | None = None,
    xlabel: str | None = None,
    title: str = "Measurement distribution",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot finite measurements, optionally restricting the displayed range.

    Require a positive integer bins value and finite increasing range bounds.
    Show all-row, finite, missing, non-finite and outside-window counts. Range
    restriction affects only this histogram; do not alter/winsorize the source.
    Return (Figure, Axes) without automatic display, saving or global styles.
    """
    values = finite_numeric_values(series)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer.")
    if value_range is not None:
        if (
            not isinstance(value_range, tuple) or len(value_range) != 2
            or not all(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value) for value in value_range)
            or value_range[0] >= value_range[1]
        ):
            raise ValueError("value_range must be a pair of finite increasing bounds.")
        outside = int((values.lt(value_range[0]) | values.gt(value_range[1])).sum())
    else:
        outside = 0
    figure, axes = _get_axes(ax, (6, 4))
    if len(values) - outside:
        axes.hist(values.to_numpy(), bins=bins, range=value_range, color="#377eb8", edgecolor="white", linewidth=0.3)
    else:
        axes.text(0.5, 0.5, "No finite values in display range", ha="center", va="center", transform=axes.transAxes)
    if value_range is not None:
        axes.set_xlim(value_range)
    missing = int(series.isna().sum())
    non_finite = int(series.notna().sum()) - len(values)
    axes.set_title(
        f"{title}\nN={len(series):,}; finite={len(values):,}; missing={missing:,}\n"
        f"non-finite={non_finite:,}; outside window={outside:,}", fontsize=9,
    )
    axes.set_xlabel(xlabel or str(series.name or "Value"))
    axes.set_ylabel("Count")
    axes.grid(axis="y", alpha=0.2)
    axes.set_axisbelow(True)
    return figure, axes


def plot_numeric_boxplot(
    series: pd.Series,
    *,
    xlabel: str | None = None,
    title: str = "Measurement boxplot",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot all finite measurements with 1.5-IQR whiskers and visible fliers.

    Fliers are descriptive points, not invalid observations or removal rules.
    Show input/finite/missing/non-finite counts and preserve the source values.
    Return (Figure, Axes) without automatic display, saving or global styles.
    """
    values = finite_numeric_values(series)
    figure, axes = _get_axes(ax, (6, 3))
    if len(values):
        axes.boxplot(
            values.to_numpy(), orientation="horizontal", whis=1.5, showfliers=True,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
        )
    else:
        axes.text(0.5, 0.5, "No finite measurements", ha="center", va="center", transform=axes.transAxes)
    axes.set_title(
        f"{title}\nN={len(series):,}; finite={len(values):,}; "
        f"missing={int(series.isna().sum()):,}; "
        f"non-finite={int(series.notna().sum()) - len(values):,}", fontsize=9,
    )
    axes.set_xlabel(xlabel or str(series.name or "Value"))
    axes.set_yticks([])
    axes.grid(axis="x", alpha=0.2)
    axes.set_axisbelow(True)
    return figure, axes
