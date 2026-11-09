"""Order-grain and late-date regressions for delivery analysis."""

import pandas as pd

from src.analysis import build_order_analysis_view


def make_items() -> pd.DataFrame:
    rows = []
    for order_id, approval, actual, estimated in [
        ("A", "2018-01-31 05:59", "2018-02-01 20:00", "2018-02-01"),
        ("B", "2018-01-31 06:00", "2018-02-02 00:01", "2018-02-01"),
        ("C", "2018-01-31 12:00", "2018-01-30 09:00", "2018-02-01"),
        ("D", "2018-01-31 18:00", "2018-02-01 11:00", "2018-02-01"),
    ]:
        for item_id in (1, 2):
            rows.append({
                "order_id": order_id,
                "order_item_id": item_id,
                "order_status": "delivered",
                "order_purchase_timestamp": pd.Timestamp("2018-01-30 08:00"),
                "order_approved_at": pd.Timestamp(approval),
                "order_delivered_carrier_date": pd.Timestamp("2018-01-31 20:00"),
                "order_delivered_customer_date": pd.Timestamp(actual),
                "order_estimated_delivery_date": pd.Timestamp(estimated),
            })
    return pd.DataFrame(rows)


def test_order_view_counts_each_order_once_and_compares_calendar_dates():
    items = make_items()
    before = items.copy(deep=True)
    orders, details = build_order_analysis_view(items)

    assert orders["order_id"].tolist() == ["A", "B", "C", "D"]
    assert orders["approval_slot"].astype(str).tolist() == [
        "00:00–06:00", "06:00–12:00", "12:00–18:00", "18:00–24:00"
    ]
    assert orders["is_late"].tolist() == [False, True, False, False]
    assert orders["delay_days"].tolist()[1] == 1
    assert orders["delay_days"].isna().tolist() == [True, False, True, True]
    assert details["review_reasons"].empty
    assert details["cohort_summary"]["orders"].tolist() == [4, 4, 0, 1]
    pd.testing.assert_frame_equal(items, before)


def test_conflicted_and_missing_orders_are_reviewed_without_blocking_valid_orders():
    items = make_items()
    before = items.copy(deep=True)
    items.loc[items["order_id"].eq("A") & items["order_item_id"].eq(2),
              "order_approved_at"] = pd.Timestamp("2018-01-31 14:20")
    items.loc[items["order_id"].eq("C"), "order_approved_at"] = pd.NaT
    items.loc[items["order_id"].eq("D") & items["order_item_id"].eq(2),
              "order_delivered_customer_date"] = pd.NaT
    before = items.copy(deep=True)
    orders, details = build_order_analysis_view(items)

    assert orders["order_id"].tolist() == ["B"]
    assert set(details["review_items"]["order_id"]) == {"A", "C", "D"}
    assert len(details["review_items"]) == 6
    assert set(details["review_reasons"].itertuples(index=False, name=None)) == {
        ("A", "order_approved_at", "conflicting_values"),
        ("C", "order_approved_at", "missing_value"),
        ("D", "order_delivered_customer_date", "conflicting_values"),
        ("D", "order_delivered_customer_date", "missing_value"),
    }
    assert details["cohort_summary"]["orders"].tolist() == [4, 1, 3, 1]
    pd.testing.assert_frame_equal(items, before)


def test_all_orders_requiring_review_return_empty_analysis_with_a_complete_reconciliation():
    items = make_items().loc[lambda frame: frame["order_id"].eq("A")].copy()
    items.loc[items["order_item_id"].eq(2), "order_approved_at"] = pd.NaT
    orders, details = build_order_analysis_view(items)

    assert orders.empty
    assert len(details["review_items"]) == len(items)
    assert details["cohort_summary"]["orders"].tolist() == [1, 0, 1, 0]


def test_midnight_slot_and_delay_across_year_boundary():
    items = make_items().loc[lambda frame: frame["order_id"].eq("A")].copy()
    items["order_approved_at"] = pd.Timestamp("2017-12-30 00:00")
    items["order_estimated_delivery_date"] = pd.Timestamp("2017-12-31")
    items["order_delivered_customer_date"] = pd.Timestamp("2018-01-02 00:01")

    orders, _ = build_order_analysis_view(items)

    assert orders.loc[0, "approval_slot"] == "00:00–06:00"
    assert orders.loc[0, "is_late"]
    assert orders.loc[0, "delay_days"] == 2
