"""Exploratory association testing across the four approval-time slots."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq

from .orders import SLOT_LABELS


def test_slot_association(summary: pd.DataFrame, alpha: float = 0.05) -> dict[str, pd.DataFrame]:
    """Test equal late rates using the unrounded counts from the slot summary.

    Retain all four slots. Empty margins or expected frequencies below five
    return descriptive tables with an unavailable test, rather than a p-value.
    Checks concern counts; independence of orders requires substantive review.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    required = {"approval_slot", "orders", "on_time_orders", "late_orders"}
    if required.difference(summary.columns):
        raise ValueError(f"Missing test columns: {sorted(required.difference(summary.columns))}")
    if summary["approval_slot"].duplicated().any() or set(summary["approval_slot"]) != set(SLOT_LABELS):
        raise ValueError("Provide exactly one summary row for each of the four slots.")
    indexed = summary.set_index("approval_slot").reindex(SLOT_LABELS)
    counts = indexed[["orders", "on_time_orders", "late_orders"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("Counts must be finite nonnegative integers.")
    if not np.array_equal(counts[:, 0], counts[:, 1] + counts[:, 2]):
        raise ValueError("On-time and late counts must sum to all orders in each slot.")

    observed = indexed[["on_time_orders", "late_orders"]].astype("int64")
    expected = pd.DataFrame(np.nan, index=observed.index, columns=observed.columns)
    total = int(observed.to_numpy().sum())
    all_slots_present = bool(observed.sum(axis=1).gt(0).all())
    both_outcomes_present = bool(observed.sum(axis=0).gt(0).all())
    enough_expected = False
    minimum_expected = float("nan")
    if all_slots_present and both_outcomes_present:
        expected.loc[:, :] = expected_freq(observed.to_numpy())
        minimum_expected = float(expected.to_numpy().min())
        enough_expected = minimum_expected >= 5

    checks = pd.DataFrame([
        ["At least one analyzable order", total > 0],
        ["All four slots have orders", all_slots_present],
        ["Both delivery outcomes are present", both_outcomes_present],
        ["Every expected count is at least five", enough_expected],
    ], columns=["check", "passed"])
    rates = np.divide(100 * counts[:, 2], counts[:, 0],
                      out=np.full(4, np.nan), where=counts[:, 0] > 0)
    rate_range = float(np.nanmax(rates) - np.nanmin(rates)) if total else float("nan")
    result = {
        "orders": total, "alpha": alpha, "test_performed": False,
        "chi2": float("nan"), "dof": float("nan"), "p_value": float("nan"),
        "min_expected": minimum_expected, "cramers_v": float("nan"),
        "rate_range_pp": rate_range, "decision": "Test unavailable; inspect failed checks",
    }
    if checks["passed"].all():
        test = chi2_contingency(observed.to_numpy(), correction=False)
        result.update(
            test_performed=True, chi2=float(test.statistic), dof=int(test.dof),
            p_value=float(test.pvalue), cramers_v=float(np.sqrt(test.statistic / total)),
            decision="Reject equal rates" if test.pvalue < alpha else "Insufficient evidence to reject equal rates",
        )
    return {"observed": observed, "expected": expected,
            "checks": checks, "result": pd.DataFrame([result])}
