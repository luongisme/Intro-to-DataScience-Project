"""Plot contracts based on plotted data, labels and input preservation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.eda.plots import (
    plot_frequency, plot_missingness_heatmap, plot_numeric_boxplot, plot_numeric_histogram,
)
from src.eda.profiling import profile_frequencies

pytestmark = pytest.mark.usefixtures("eda_plot_cleanup")


def test_histogram_window_accounts_for_all_finite_measurements():
    series = pd.Series([-2., 0., 1., 2., 100., np.inf, np.nan], name="price")
    figure, axes = plot_numeric_histogram(series, bins=4, value_range=(0., 2.), xlabel="Price (BRL)")
    figure.canvas.draw()
    assert sum(rect.get_height() for rect in axes.patches) == 3
    assert axes.get_xlim() == (0., 2.) and axes.get_xlabel() == "Price (BRL)"
    assert "N=7; finite=5; missing=1" in axes.get_title()
    assert "non-finite=1; outside window=2" in axes.get_title()


@pytest.mark.parametrize("series", [
    pd.Series([], dtype=float, name="price"),
    pd.Series([pd.NA], dtype="Float64", name="price"),
    pd.Series([2., 2., 2.], name="price"),
])
@pytest.mark.parametrize("plotter", [plot_numeric_histogram, plot_numeric_boxplot])
def test_measurement_plots_render_empty_missing_and_constant_data(series, plotter):
    figure, axes = plotter(series)
    figure.canvas.draw()
    assert axes.figure is figure and plt.fignum_exists(figure.number)
    if series.dropna().empty:
        assert any("No finite" in text.get_text() for text in axes.texts)


@pytest.mark.parametrize("options", [
    {"bins": 0}, {"bins": True}, {"bins": 1.5},
    {"value_range": (2., 1.)}, {"value_range": (1., 1.)},
    {"value_range": (0., np.inf)}, {"value_range": [0., 1.]},
])
def test_histogram_invalid_display_options_fail_before_creating_figures(options):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_numeric_histogram(pd.Series([1., 2.], name="price"), **options)
    assert plt.get_fignums() == before


def test_frequency_plot_keeps_special_kinds_separate_from_literal_names():
    full = profile_frequencies(pd.Series(["A", "A", "Missing", "Other", None]))
    figure, axes = plot_frequency(full, top_n=None)
    figure.canvas.draw()
    labels = [tick.get_text() for tick in axes.get_yticklabels()]
    assert repr("Missing") in labels and "[Missing]" in labels and repr("Other") in labels
    assert sum(rect.get_width() for rect in axes.patches) == 5
    figure, axes = plot_frequency(full, top_n=1, horizontal=False)
    figure.canvas.draw()
    assert sum(rect.get_height() for rect in axes.patches) == 5
    assert len(axes.patches) == 3


def test_frequency_plot_empty_input_is_not_a_zero_observation_bar():
    figure, axes = plot_frequency(profile_frequencies(pd.Series([], dtype="string")))
    figure.canvas.draw()
    assert len(axes.patches) == 0
    assert any("No observations" in text.get_text() for text in axes.texts)


def test_frequency_plot_rejects_inconsistent_count_tables():
    full = profile_frequencies(pd.Series(["A", "B"]))
    with pytest.raises(ValueError):
        plot_frequency(full.assign(count=[1, 0, 0]), top_n=None)
    with pytest.raises(ValueError):
        plot_frequency(full.assign(count=[0.5, 1.5, 0.]), top_n=None)


def test_heatmap_distinguishes_zero_small_rates_and_undefined_denominators():
    matrix = pd.DataFrame([[0., 0.01, 99.99, 100., np.nan]], index=["given"], columns=["zero", "small", "near_all", "all", "undefined"])
    before = matrix.copy(deep=True)
    figure, axes = plot_missingness_heatmap(matrix)
    figure.canvas.draw()
    assert [text.get_text() for text in axes.texts] == ["0.0", "<0.1", ">99.9", "100.0", chr(0x2014)]
    assert np.ma.getmaskarray(axes.images[0].get_array()).tolist() == [[False, False, False, False, True]]
    pd.testing.assert_frame_equal(matrix, before)


@pytest.mark.parametrize("matrix", [pd.DataFrame(), pd.DataFrame(np.nan, index=["a", "b"], columns=["x"])])
def test_heatmap_empty_or_entirely_undefined_matrix_renders(matrix):
    figure, axes = plot_missingness_heatmap(matrix)
    figure.canvas.draw()
    assert axes.figure is figure


@pytest.mark.parametrize("matrix,error", [
    (pd.DataFrame([[-1.]]), ValueError), (pd.DataFrame([[101.]]), ValueError),
    (pd.DataFrame([[np.inf]]), ValueError), (pd.DataFrame([["25"]]), TypeError),
    (pd.DataFrame([[True]]), TypeError), (pd.DataFrame([[1j]]), TypeError),
    (pd.DataFrame([[1.], [2.]], index=["same", "same"]), ValueError),
    (pd.DataFrame([[1., 2.]], columns=["same", "same"]), ValueError),
])
def test_heatmap_rejects_invalid_percentages_or_ambiguous_axes(matrix, error):
    with pytest.raises(error):
        plot_missingness_heatmap(matrix)


@pytest.mark.parametrize("kind", ["histogram", "boxplot", "frequency", "heatmap"])
def test_plots_reuse_provided_axes_without_automatic_io_or_style_mutation(monkeypatch, kind):
    def forbidden(*args, **kwargs):
        pytest.fail("A plotting helper attempted automatic display or saving.")
    monkeypatch.setattr(plt, "show", forbidden)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", forbidden)
    original_rc = matplotlib.rcParams.copy()
    series = pd.Series([1., 2., 3.], name="price", index=[7, 7, 2])
    table = profile_frequencies(pd.Series(["A", "B", None])) if kind == "frequency" else pd.DataFrame([[0., np.nan]])
    original_series, original_table = series.copy(deep=True), table.copy(deep=True)
    figure, axes = plt.subplots()
    count = len(plt.get_fignums())
    if kind == "histogram":
        result = plot_numeric_histogram(series, ax=axes)
    elif kind == "boxplot":
        result = plot_numeric_boxplot(series, ax=axes)
    elif kind == "frequency":
        result = plot_frequency(table, ax=axes)
    else:
        result = plot_missingness_heatmap(table, ax=axes)
    figure.canvas.draw()
    assert result == (figure, axes) and len(plt.get_fignums()) == count
    assert matplotlib.rcParams == original_rc
    pd.testing.assert_series_equal(series, original_series)
    pd.testing.assert_frame_equal(table, original_table)


def test_invalid_axes_are_reported_explicitly():
    with pytest.raises(TypeError, match="Axes"):
        plot_numeric_histogram(pd.Series([1.], name="price"), ax="wrong")
