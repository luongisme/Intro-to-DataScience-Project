"""Check denominators and empty-slot behavior in delivery summaries."""

import pandas as pd
import pytest

from src.analysis import summarize_delivery_by_slot
from src.analysis.orders import SLOT_LABELS


def make_orders() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["A", "B", "C", "D"],
        "approval_slot": pd.Categorical(SLOT_LABELS, categories=SLOT_LABELS, ordered=True),
        "is_late": [False, True, False, False],
        "delay_days": pd.array([pd.NA, 1, pd.NA, pd.NA], dtype="Int64"),
    })


def test_slot_summary_counts_orders_and_uses_late_only_for_delay():
    orders = make_orders()
    original = orders.copy(deep=True)
    summary = summarize_delivery_by_slot(orders)

    assert summary["approval_slot"].tolist() == list(SLOT_LABELS)
    assert summary["orders"].tolist() == [1, 1, 1, 1]
    assert summary["on_time_orders"].tolist() == [1, 0, 1, 1]
    assert summary["late_orders"].tolist() == [0, 1, 0, 0]
    assert summary["late_rate_pct"].tolist() == [0.0, 100.0, 0.0, 0.0]
    assert summary.loc[1, "mean_delay_days"] == 1
    assert summary.loc[1, "median_delay_days"] == 1
    assert summary.loc[[0, 2, 3], "mean_delay_days"].isna().all()
    pd.testing.assert_frame_equal(orders, original)


def test_empty_slots_and_empty_cohort_keep_four_rows():
    orders = make_orders()
    summary = summarize_delivery_by_slot(orders.loc[orders["order_id"].eq("B")])
    assert summary["orders"].tolist() == [0, 1, 0, 0]
    assert summary.loc[[0, 2, 3], "late_rate_pct"].isna().all()
    assert summary.loc[1, "late_rate_pct"] == 100.0

    empty = summarize_delivery_by_slot(orders.iloc[0:0])
    assert empty["orders"].tolist() == [0, 0, 0, 0]
    assert empty["late_orders"].sum() == 0
    assert empty["late_rate_pct"].isna().all()
    assert empty["mean_delay_days"].isna().all()


def test_equal_leaders_remain_equal_and_duplicate_orders_are_rejected():
    orders = make_orders()
    summary = summarize_delivery_by_slot(orders)
    leaders = summary.loc[summary["late_orders"].eq(summary["late_orders"].max())]
    assert len(leaders) == 1

    tied = orders.loc[orders["order_id"].isin(["A", "D"])].copy()
    tied["is_late"] = True
    tied["delay_days"] = pd.Series([2, 2], index=tied.index, dtype="Int64")
    tied_summary = summarize_delivery_by_slot(tied)
    assert tied_summary.loc[tied_summary["late_orders"].eq(1), "approval_slot"].tolist() == [
        SLOT_LABELS[0], SLOT_LABELS[3]
    ]
    with pytest.raises(ValueError, match="one row per order_id"):
        summarize_delivery_by_slot(pd.concat([orders, orders.iloc[[0]]], ignore_index=True))
