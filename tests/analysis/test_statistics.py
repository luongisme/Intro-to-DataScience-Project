"""Independent contingency-table checks and unavailable-test regressions."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.orders import SLOT_LABELS
from src.analysis.statistics import test_slot_association as evaluate_association


def make_summary(on_time, late):
    return pd.DataFrame({
        "approval_slot": SLOT_LABELS,
        "orders": np.array(on_time) + np.array(late),
        "on_time_orders": on_time, "late_orders": late,
    })


def test_equal_rates_give_zero_chi_square_and_preserve_input():
    summary = make_summary([30, 60, 90, 120], [10, 20, 30, 40])
    before = summary.copy(deep=True)
    tables = evaluate_association(summary)
    result = tables["result"].iloc[0]
    assert result["test_performed"]
    assert result["chi2"] == 0
    assert result["p_value"] == 1
    assert result["cramers_v"] == 0
    assert result["rate_range_pp"] == 0
    assert result["dof"] == 3
    assert tables["checks"]["passed"].all()
    pd.testing.assert_frame_equal(summary, before)


def test_known_unequal_table_matches_hand_calculated_statistic():
    tables = evaluate_association(make_summary([90, 80, 70, 60], [10, 20, 30, 40]))
    np.testing.assert_allclose(tables["expected"], [[75, 25]] * 4)
    result = tables["result"].iloc[0]
    assert result["chi2"] == pytest.approx(80 / 3)
    assert result["cramers_v"] == pytest.approx(np.sqrt((80 / 3) / 400))
    assert result["rate_range_pp"] == 30
    assert result["p_value"] < 0.05
    assert result["decision"] == "Reject equal rates"


@pytest.mark.parametrize("on_time,late", [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([0, 30, 30, 30], [0, 10, 10, 10]),
    ([30, 30, 30, 30], [0, 0, 0, 0]),
    ([0, 0, 0, 0], [30, 30, 30, 30]),
    ([2, 2, 2, 2], [1, 1, 1, 1]),
])
def test_degenerate_or_small_tables_return_counts_and_unavailable_test(on_time, late):
    tables = evaluate_association(make_summary(on_time, late))
    result = tables["result"].iloc[0]
    assert not result["test_performed"]
    assert pd.isna(result["p_value"])
    assert pd.isna(result["cramers_v"])
    assert not tables["checks"]["passed"].all()
    assert tables["observed"].shape == (4, 2)
    assert tables["observed"].to_numpy().sum() == sum(on_time) + sum(late)


def test_inconsistent_counts_are_rejected_before_testing():
    summary = make_summary([30] * 4, [10] * 4)
    summary.loc[0, "orders"] = 99
    with pytest.raises(ValueError, match="must sum"):
        evaluate_association(summary)
