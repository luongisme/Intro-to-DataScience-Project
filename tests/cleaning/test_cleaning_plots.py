"""Plot-data contracts for reproducible post-cleaning review."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.cleaning import clean_order_items
from src.cleaning.plots import (
    build_cleaning_visuals, plot_filtering_flow, profile_missingness_comparison,
)


pytestmark = pytest.mark.usefixtures("eda_plot_cleanup")


@pytest.fixture
def visual_source(eda_full_frame):
    """Supply repeated item exposure with valid source-shaped identifiers."""
    source = eda_full_frame.copy(deep=True)
    for group, column in enumerate(("order_id", "product_id", "seller_id", "customer_id"), 1):
        mapping = {value: f"{group:02x}{position:030x}"
                   for position, value in enumerate(source[column].unique(), 1)}
        source[column] = source[column].map(mapping).astype("string")
    return source


def test_review_uses_correct_grains_preserves_inputs_and_renders(visual_source):
    cleaned, details = clean_order_items(visual_source)
    before_source, before_clean = visual_source.copy(deep=True), cleaned.copy(deep=True)
    font_size = plt.rcParams["font.size"]
    review = build_cleaning_visuals(visual_source, cleaned, details)
    tables = review["tables"]
    numeric = tables["numeric"].set_index("column")

    assert set(review["figures"]) == {"filtering", "missingness", "numeric", "calendar", "categories", "geography"}
    assert numeric.loc["price", "rows"] == 3
    assert numeric.loc["product_weight_g", "rows"] == 2
    assert numeric.loc["product_weight_g", "median"] == 150.0
    assert tables["categories"]["count"].sum() == 2
    assert tables["category_display"]["count"].sum() == 2
    assert tables["customer_states"]["count"].sum() == 2
    assert tables["seller_states"]["count"].sum() == 2
    assert tables["purchase_months"]["orders"].sum() == 2
    assert tables["approval_hours"]["orders"].sum() == 2
    assert tables["approval_hours"].set_index("hour").loc[11, "orders"] == 2
    for figure in review["figures"].values():
        figure.canvas.draw()
    assert plt.rcParams["font.size"] == font_size
    pd.testing.assert_frame_equal(visual_source, before_source, check_exact=True)
    pd.testing.assert_frame_equal(cleaned, before_clean, check_exact=True)


def test_flow_figure_uses_orders_not_item_counts(visual_source):
    source = visual_source.copy(deep=True)
    source.iloc[1, source.columns.get_loc("product_weight_g")] = 0.0
    _, details = clean_order_items(source)
    figure = plot_filtering_flow(details["filter_flow"])
    assert sum(bar.get_width() for bar in figure.axes[0].patches) == 2
    assert sum(bar.get_width() for bar in figure.axes[1].patches) == 1
    assert "each order counted once" in figure.axes[1].get_xlabel()
    figure.canvas.draw()


def test_matching_physical_missingness_is_grouped_per_field_not_summed(visual_source):
    source = visual_source.copy(deep=True)
    for column in ("product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"):
        source.iloc[[0, 2], source.columns.get_loc(column)] = pd.NA
    source.iloc[2, source.columns.get_loc("order_status")] = "canceled"
    cleaned, details = clean_order_items(source)
    review = build_cleaning_visuals(source, cleaned, details)
    table = review["tables"]["missingness_display"]
    assert len(table) == 1
    row = table.iloc[0]
    assert row["column_count"] == 4
    assert row["missing_before"] == 2 and row["missing_after"] == 1
    assert row["missing_pct_before"] == pytest.approx(100 * 2 / 3)
    assert row["missing_pct_after"] == 50.0


def test_empty_retained_review_marks_undefined_rates_and_renders(visual_source):
    source = visual_source.copy(deep=True)
    source["order_status"] = pd.Series("canceled", index=source.index, dtype="string")
    source.iloc[[0, 2], source.columns.get_loc("product_category_name")] = pd.NA
    cleaned, details = clean_order_items(source)
    review = build_cleaning_visuals(source, cleaned, details)
    assert review["tables"]["missingness_display"]["missing_pct_after"].isna().all()
    assert any("N/A" in text.get_text() for text in review["figures"]["missingness"].axes[0].texts)
    for figure in review["figures"].values():
        figure.canvas.draw()


@pytest.mark.parametrize("problem", ["input_total", "removed_total"])
def test_inconsistent_filtering_counts_fail_before_creating_figures(visual_source, problem):
    cleaned, details = clean_order_items(visual_source)
    details["filter_flow"] = details["filter_flow"].copy(deep=True)
    if problem == "input_total":
        details["filter_flow"].loc[0, "remaining_orders"] += 1
    else:
        details["filter_flow"].loc[1, "removed_orders"] += 1
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        build_cleaning_visuals(visual_source, cleaned, details)
    assert plt.get_fignums() == before


def test_distinct_physical_missingness_groups_have_unambiguous_labels():
    source = pd.DataFrame({
        "product_weight_g": [None, None, 1.0],
        "product_length_cm": [None, None, 2.0],
        "product_height_cm": [None, 1.0, 2.0],
        "product_width_cm": [None, 1.0, 2.0],
    })
    _, compact = profile_missingness_comparison(source, source.copy(deep=True))
    assert len(compact) == 2
    assert compact["attribute"].nunique() == 2
    assert compact["column_count"].tolist() == [2, 2]
    assert "Weight" in compact.iloc[0]["attribute"]
    assert "Height" in compact.iloc[1]["attribute"]

