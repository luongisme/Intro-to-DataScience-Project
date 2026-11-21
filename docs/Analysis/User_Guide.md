# Analysis User Guide

This guide explains how to run the delivery analysis, create its tables and figures, and interpret their scope. See the [Analysis notebook](../../notebooks/03_analysis.ipynb) for the full workflow. Function contracts and calculations are documented in the [Technical Reference](Technical_Reference.md). Results for the reference dataset are presented in the [Analysis Report](../../reports/Analysis_Report.md).

## Contents

1. [Inputs and outputs](#1-inputs-and-outputs)
2. [Run the notebook](#2-run-the-notebook)
3. [Shared setup](#3-shared-setup)
4. [Workflow by task](#4-workflow-by-task)
5. [Read and verify outputs](#5-read-and-verify-outputs)
6. [Troubleshooting](#6-troubleshooting)

## 1. Inputs and outputs

| Item | Description |
|---|---|
| Research question | Which delivery time slots cause the most late orders? |
| Slot anchor | Order approval time, `order_approved_at`, divided into four six-hour intervals |
| Input | Item-level `cleaned_order_items` in `data/processed` |
| Reference dataset | 108,571 items × 25 columns, belonging to 95,075 delivered orders |
| Additional input | `merged_order_items` in `data/interim` for the before-and-after cleaning comparison |
| Notebook | [03_analysis.ipynb](../../notebooks/03_analysis.ipynb) |
| Outputs | Tables and figures in memory; saved report figures in [outputs/figures/Analysis](../../outputs/figures/Analysis/) |

Each eligible order is counted once. An order is late when its delivery date is after its estimated delivery date. Approval time anchors the observed comparisons. It does not identify a scheduled delivery slot or the causal effect of changing approval time. Reference counts describe the current dataset and do not constrain other valid inputs.

## 2. Run the notebook

Open the notebook with the project's Python kernel, restart the kernel and run the cells in order. Later sections use tables created earlier.

| Notebook sections | Scope | Outputs |
|---|---|---|
| 1–2 | Load the cleaned table, create the order-level view and record review cases | `orders`, `order_diagnostics` |
| 3–8 | Approval slots, statistical test, purchase months, stages and route types | Summary tables, figures 01–12 |
| 9–12 | Orders approved 12:00–18:00, routes, destinations, state pairs and time | Afternoon cohort tables, figures 13–16 |
| 13–15 | Compare the selected inter-state pair across four slots and delivery stages by purchase period | Rate, stage and promise tables, figures 17–22 |
| 16 | Orders approved 00:00–06:00, routes, destinations, pairs and coverage | Overnight cohort tables, figure 23 |
| 17–18 | Compare locations and purchase months across two slots | Group tables, month standardization, figures 24–27 |
| 19–20 | Compare delivery stages and promises, then synthesize the answer | Stage tables and conclusion |

The notebook displays tables and figures in memory. Reports and saved figures are maintained separately. If input data change, rerun the notebook and compare its static commentary with the new output.

## 3. Shared setup

Run this block once in a notebook or IPython session with the working directory inside the repository. The examples below share these variables.

```python
import sys
from pathlib import Path

for root in (Path.cwd(), *Path.cwd().parents):
    if (root / "src").is_dir() and (root / "requirements.txt").is_file():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        break
else:
    raise FileNotFoundError("Run this code inside the repository.")

import matplotlib.pyplot as plt
from IPython.display import display
from src.data_loader import load_data
from src.analysis.orders import build_order_analysis_view
from src.analysis.summary import summarize_delivery_by_slot
from src.analysis.statistics import test_slot_association
from src.analysis.presentation import format_analysis_table
from src.analysis.plots import (
    build_temporal_profiles, plot_slot_counts, plot_rate_profile,
    plot_hour_weekday_heatmap, plot_delay_distribution,
)
from src.analysis.investigation import (
    summarize_time_periods, compare_cleaning_cohorts, investigate_delivery_stages,
    classify_order_routes, summarize_routes, build_focus_cohort, summarize_focus_routes,
    summarize_destination_states, summarize_state_pairs,
    summarize_focus_time, summarize_focus_comparisons,
)
from src.analysis.route_slots import build_state_pair_orders, summarize_state_pair_slots
from src.analysis.slot_comparisons import (
    COMPARISON_SLOTS, compare_approval_slot_groups, summarize_slot_group_months,
)
from src.analysis.route_stages import (
    compare_route_delivery_stages, summarize_afternoon_route_evidence,
    summarize_route_slot_evidence, summarize_stage_slots,
)
from src.analysis.investigation_plots import (
    plot_monthly_slot_rates, plot_matched_years, plot_stage_comparison,
    plot_promised_windows, plot_route_slot_rates, plot_route_transit_time,
    plot_destination_states, plot_state_pairs, plot_focus_months,
    plot_focus_approval_time, plot_state_pair_slots, plot_state_pair_months,
    plot_route_period_medians, plot_route_period_slot_rates,
    plot_focus_routes, plot_slot_group_comparison, plot_slot_group_months,
)

cleaned_items = load_data("cleaned_order_items", stage="processed")
orders, order_diagnostics = build_order_analysis_view(cleaned_items)
slot_summary = summarize_delivery_by_slot(orders)
order_routes = classify_order_routes(cleaned_items, orders)
stage_profiles = investigate_delivery_stages(orders)
```

By default, the loader tries Parquet before CSV. To select CSV directly, replace the load call with `cleaned_items = load_data("cleaned_order_items", stage="processed", file_format="csv")`, then recreate the dependent tables. If the input is unavailable, see the [Cleaning User Guide](../Cleaning/User_Guide.md).

## 4. Workflow by task

### All orders and four approval slots

These tables show counts, late rates and distributions before examining individual slots.

#### Order-level view and overview figures

```python
display(order_diagnostics["cohort_summary"])
display(order_diagnostics["review_reasons"])
display(order_diagnostics["review_items"])
display(format_analysis_table(slot_summary))

temporal_profiles = build_temporal_profiles(orders)
for figure in (
    plot_slot_counts(slot_summary),
    plot_rate_profile(temporal_profiles["hours"], "approval_hour"),
    plot_rate_profile(temporal_profiles["weekdays"], "approval_weekday"),
    plot_hour_weekday_heatmap(temporal_profiles["hour_weekday"]),
    plot_delay_distribution(orders),
):
    display(figure)
    plt.close(figure)
```

`order_diagnostics` identifies orders requiring review. Each late rate uses the orders in its group; delay-day summaries use late orders only.

#### Statistical test and purchase time

```python
slot_test = test_slot_association(slot_summary, alpha=0.05)
display(slot_test["observed"])
display(slot_test["expected"])
display(slot_test["checks"])
display(slot_test["result"])

period_profiles = summarize_time_periods(orders)
display(format_analysis_table(period_profiles["monthly"]))
display(format_analysis_table(period_profiles["matched_years"]))
for figure in (
    plot_monthly_slot_rates(period_profiles["monthly"]),
    plot_matched_years(period_profiles["matched_years"]),
):
    display(figure)
    plt.close(figure)
```

The test assesses the overall difference across four slots. Months use the purchase date; slots use approval time. The p-value depends on independence between orders.

To compare cohorts before and after cleaning, use the merged table from the same source.

```python
merged_items = load_data("merged_order_items", stage="interim")
cohort_comparison = compare_cleaning_cohorts(merged_items, orders)
display(cohort_comparison["coverage"])
if cohort_comparison["slots"].empty:
    print("Comparison unavailable. Check coverage.")
else:
    display(format_analysis_table(cohort_comparison["slots"]))
```

If comparison fields do not match, `coverage` records the issue and `slots` may be empty.

#### Delivery stages and route types

```python
route_profiles = summarize_routes(orders, order_routes, stage_profiles["records"])
for name in ("quality", "summary", "pooled_medians", "promised_summary"):
    display(format_analysis_table(stage_profiles[name]))
display(stage_profiles["negative_windows"])
for name in ("totals", "slots", "transit"):
    display(format_analysis_table(route_profiles[name]))

for figure in (
    plot_stage_comparison(stage_profiles["summary"], "Processing"),
    plot_stage_comparison(stage_profiles["summary"], "Carrier transit"),
    plot_promised_windows(stage_profiles["windows"]),
    plot_route_slot_rates(route_profiles["slots"]),
    plot_route_transit_time(route_profiles["transit"]),
):
    display(figure)
    plt.close(figure)
```

Calendar hours measure the entire elapsed interval. Eligible-day hours count Monday–Friday, excluding January 1, May 1 and December 25, with up to 24 hours per eligible day. They do not represent actual staffed working hours.

### Orders approved from 12:00–18:00

This section starts with all afternoon orders, then examines customer states and inter-state pairs with the most late orders.

#### Cohort and customer states

```python
focus_orders, focus_diagnostics = build_focus_cohort(
    cleaned_items, orders, order_routes
)
display(format_analysis_table(focus_diagnostics["summary"]))
display(focus_diagnostics["location_review"])

destination_profiles = summarize_destination_states(focus_orders)
display(format_analysis_table(destination_profiles["all_routes"]))
display(format_analysis_table(destination_profiles["inter_state"]))
figure = plot_destination_states(destination_profiles["inter_state"])
display(figure)
plt.close(figure)
```

`focus_diagnostics` reports cohort totals and locations requiring review. `all_routes` retains all routes; `inter_state` contains only inter-state deliveries. Share denominators change with the scope.

#### Seller → customer state pairs

Run the afternoon cohort and customer-state section first. Pairs are ranked by late-order count; the first pair with late orders is selected for further analysis.

```python
pair_profiles = summarize_state_pairs(cleaned_items, focus_orders)
display(format_analysis_table(pair_profiles["coverage"]))
display(format_analysis_table(pair_profiles["pairs"]))

leading_pair = None
if not pair_profiles["pairs"].empty and pair_profiles["pairs"].iloc[0]["late_orders"] > 0:
    first = pair_profiles["pairs"].iloc[0]
    leading_pair = (first["seller_state"], first["customer_state"])

focus_comparisons = summarize_focus_comparisons(pair_profiles["orders"], leading_pair)
if leading_pair is not None:
    display(format_analysis_table(focus_comparisons["same_destination"]))

n_inter = int(pair_profiles["coverage"]["orders"].sum())
n_late = int(pair_profiles["coverage"]["late_orders"].sum())
inter_rate = 100 * n_late / n_inter if n_inter else float("nan")
figure = plot_state_pairs(pair_profiles["pairs"], inter_rate)
display(figure)
plt.close(figure)
```

`coverage` retains orders with multiple seller states or locations requiring review. `pairs` contains identifiable pairs only. `leading_pair` is `None` if no eligible pair has late orders.

#### Purchase and approval timing

Run the seller-to-customer state-pair section first.

```python
focus_time = summarize_focus_time(pair_profiles["orders"], leading_pair)
for name in ("monthly", "matched_years", "hours", "weekdays"):
    display(format_analysis_table(focus_time[name]))
if leading_pair is not None:
    display(format_analysis_table(focus_comparisons["concentration"]))
    display(format_analysis_table(focus_comparisons["periods"]))

for figure in (
    plot_focus_months(focus_time["monthly"]),
    plot_focus_approval_time(focus_time["hours"], focus_time["weekdays"]),
):
    display(figure)
    plt.close(figure)
```

`periods` compares the selected pair with all remaining inter-state orders, across all destinations. Approval hour and weekday are separate groupings.

#### Inter-state pair across four slots

Run the seller-to-customer state-pair section first. The selected pair is retrieved from all `orders` across four approval slots.

```python
pair_orders = build_state_pair_orders(cleaned_items, orders, order_routes, leading_pair)
pair_slots = summarize_state_pair_slots(pair_orders)
if leading_pair is not None:
    for name in ("raw", "monthly", "standardized"):
        display(format_analysis_table(pair_slots[name]))
    for figure in (
        plot_state_pair_slots(pair_slots["raw"], leading_pair),
        plot_state_pair_months(pair_slots["monthly"], leading_pair),
    ):
        display(figure)
        plt.close(figure)
else:
    print("No leading state pair is available.")
```

`raw` contains rates and Wilson intervals; `monthly` uses purchase months; `standardized` applies common month weights to four slots. Replace `leading_pair` with another inter-state tuple such as `("MG", "RJ")` if needed, then recreate the dependent pair and stage tables.

#### Delivery stages and promised windows by purchase period

Run the four-slot pair section first. The comparison group contains other inter-state orders to the **same customer state**, distinct from all remaining inter-state orders in the afternoon purchase and approval timing section.

```python
route_comparison = None
if leading_pair is not None:
    route_comparison = compare_route_delivery_stages(
        cleaned_items, orders, order_routes, pair_orders, stage_profiles, leading_pair
    )
    for name in ("coverage", "outcomes", "stages", "promises"):
        display(format_analysis_table(route_comparison[name]))
    afternoon = route_comparison["by_outcome"].loc[
        route_comparison["by_outcome"]["approval_slot"].eq("12:00–18:00")
    ]
    display(format_analysis_table(afternoon))

    for stage, title in (
        ("Processing", "Approval to carrier handoff by purchase period and approval slot"),
        ("Carrier transit", "Carrier handoff to delivery by purchase period and approval slot"),
    ):
        stage_table = route_comparison["stages"]
        figure = plot_route_period_medians(
            stage_table.loc[stage_table["stage"].eq(stage)],
            "median_calendar_hours", title, "Median elapsed hours",
        )
        display(figure)
        plt.close(figure)
    figure = plot_route_period_medians(
        route_comparison["promises"], "median_days",
        "Approval to promised date by purchase period and approval slot",
        "Median calendar days",
    )
    display(figure)
    plt.close(figure)
else:
    print("No leading state pair is available for the route comparison.")
```

`stages` contains medians, P75 and valid interval counts under both calendar-hour and eligible-day conventions. `by_outcome` separates orders by observed delivery outcome. Stage medians cannot be added to obtain the median end-to-end duration.

#### Evidence by period and approval slot

Run the matched-period delivery-stage section first.

```python
if route_comparison is not None:
    figure = plot_route_period_slot_rates(route_comparison["outcomes"])
    display(figure)
    plt.close(figure)
    afternoon_evidence = summarize_afternoon_route_evidence(route_comparison)
    display(format_analysis_table(afternoon_evidence))
else:
    print("No route-stage comparison is available.")
```

`afternoon_evidence` combines counts, delivery outcomes, stage medians and promised windows for afternoon orders from the existing tables. Figure 22 displays purchase periods, groups and slots using a common color scale.

### Orders approved from 00:00–06:00

Run section 3 first. The overnight cohort retains all route types, then examines destinations and eligible state pairs separately.

```python
night_focus_orders, night_diagnostics = build_focus_cohort(
    cleaned_items, orders, order_routes, approval_slot="00:00–06:00"
)
night_routes = summarize_focus_routes(night_focus_orders)
night_destinations = summarize_destination_states(night_focus_orders)
night_pairs = summarize_state_pairs(
    cleaned_items, night_focus_orders, include_intra_state=True
)
for table in (
    night_diagnostics["summary"], night_diagnostics["location_review"],
    night_routes, night_destinations["all_routes"], night_destinations["inter_state"],
    night_pairs["coverage"], night_pairs["pairs"],
):
    display(format_analysis_table(table))
figure = plot_focus_routes(night_routes, "00:00–06:00")
display(figure)
plt.close(figure)
```

`night_routes` contains the four route types. `night_destinations` contains all-route and inter-state tables. `night_pairs["coverage"]` retains orders with multiple seller states or locations requiring review; `pairs` ranks identifiable pairs. In the expanded mode, shares use the entire overnight cohort as their denominator.

### Compare slots by location, purchase month and delivery stage

#### Compare the same customer state or state pair

Run the afternoon and overnight cohort sections first.

```python
afternoon_pairs = summarize_state_pairs(
    cleaned_items, focus_orders, include_intra_state=True
)
destination_comparison = compare_approval_slot_groups(
    night_focus_orders, focus_orders, group_by="customer_state"
)
pair_comparison = compare_approval_slot_groups(
    night_pairs["orders"], afternoon_pairs["orders"], group_by="state_pair"
)
for comparison, group_by in (
    (destination_comparison, "customer_state"),
    (pair_comparison, "state_pair"),
):
    for name in ("scope_summary", "selection", "selected", "all_groups"):
        display(format_analysis_table(comparison[name]))
    figure = plot_slot_group_comparison(comparison["plot_data"], group_by)
    display(figure)
    plt.close(figure)
```

`all_groups` retains every group, `selection` records why each was chosen, `selected` contains the chosen rows, and `plot_data` adds the remaining groups and locations requiring review. `difference_pp` is the **overnight minus afternoon** rate in percentage points; rates are undefined for groups with no orders.

#### Purchase months and standardized rates

Run the same-group comparison section first.

```python
destination_months = summarize_slot_group_months(
    night_focus_orders, focus_orders, group_by="customer_state"
)
pair_months = summarize_slot_group_months(
    night_pairs["orders"], afternoon_pairs["orders"], group_by="state_pair"
)
for results in (destination_months, pair_months):
    for name in ("selection", "monthly", "standardized", "weights", "differences"):
        display(format_analysis_table(results[name]))

for results, group_by, preferred in (
    (destination_months, "customer_state", ("CE", "AL")),
    (pair_months, "state_pair", ("SP → SP", "SP → MA", "PR → RS")),
):
    chosen = [name for name in preferred if name in results["selection"]["group"].tolist()]
    figure = plot_slot_group_months(
        results["monthly"], group_by=group_by, groups=chosen or None
    )
    display(figure)
    plt.close(figure)
```

`monthly` contains the complete monthly series for each selected group. `standardized` reports full-cohort raw, common-month raw and standardized rates, plus retained and excluded order and late counts. `weights` contains monthly weights; `differences` contains the two-slot gaps. The two figures display illustrative groups. Standardization here uses two slots, so weights differ from the four-slot comparison in the four-slot pair section.

#### Delivery stages and promised windows

Run the matched-period delivery-stage and purchase-month comparison sections first.

```python
if route_comparison is not None:
    two_slot_evidence = summarize_route_slot_evidence(
        route_comparison, approval_slots=COMPARISON_SLOTS
    )
    display(format_analysis_table(two_slot_evidence))
    for name in ("stages", "promises", "by_outcome"):
        two_slot_rows = route_comparison[name].loc[
            route_comparison[name]["approval_slot"].isin(COMPARISON_SLOTS)
        ]
        display(format_analysis_table(two_slot_rows))
else:
    print("No destination-matched route comparison is available.")

sp_sp_stage_profiles = None
if not pair_months["selection"]["group"].eq("SP → SP").any():
    print("SP → SP was not selected for the monthly comparison.")
else:
    common_months = pair_months["weights"].loc[
        pair_months["weights"]["group"].eq("SP → SP"), "purchase_month"
    ]
    if common_months.empty:
        print("No shared purchase months are available for SP → SP.")
    else:
        sp_sp_orders = build_state_pair_orders(
            cleaned_items, orders, order_routes, ("SP", "SP"), include_intra_state=True
        )
        sp_sp_orders = sp_sp_orders.loc[
            sp_sp_orders["approval_slot"].isin(COMPARISON_SLOTS)
            & sp_sp_orders["order_purchase_timestamp"].dt.strftime("%Y-%m").isin(common_months)
        ].copy()
        sp_sp_stage_profiles = investigate_delivery_stages(sp_sp_orders)
        sp_sp_whole_stages = summarize_stage_slots(
            sp_sp_stage_profiles["records"], approval_slots=COMPARISON_SLOTS
        )
        display(format_analysis_table(sp_sp_whole_stages))
        display(format_analysis_table(
            sp_sp_stage_profiles["promised_summary"].loc[
                sp_sp_stage_profiles["promised_summary"]["approval_slot"].isin(COMPARISON_SLOTS),
                ["approval_slot", "orders", "median_days"],
            ]
        ))
        display(format_analysis_table(sp_sp_stage_profiles["summary"]))
```

`two_slot_evidence` and the `route_comparison` tables compare the selected inter-state pair with other inter-state orders to the same customer state across three purchase periods. For SP → SP, `sp_sp_whole_stages` pools all orders in each slot across common months, while `sp_sp_stage_profiles["summary"]` separates outcomes after delivery. Neither stage summary applies monthly weights. Standardized rates are in `pair_months["standardized"]`.

## 5. Read and verify outputs

| Output | Interpretation |
|---|---|
| `slot_summary` | Counts and rates by slot for the complete analysis cohort |
| `focus_diagnostics`, `destination_profiles`, `pair_profiles` | Afternoon cohort, customer states and inter-state pairs |
| `night_diagnostics`, `night_destinations`, `night_pairs` | Overnight cohort, customer states and pair coverage |
| `destination_comparison`, `pair_comparison` | Same-group comparisons across two slots, including remaining groups |
| `destination_months`, `pair_months` | Purchase months, retained/excluded counts and standardized rates |
| `route_comparison`, `two_slot_evidence`, `sp_sp_whole_stages` | Stage comparisons by period and comparison group, or the full SP → SP cohort when available |

A late rate uses the orders in its group as the denominator; a late-order share uses all late orders in the stated scope. Carrier transit measures handoff to customer delivery; it is not time past the promised date. Wilson intervals describe uncertainty around individual rates. Overlapping intervals do not establish equal rates.

## 6. Troubleshooting

| Situation | What to check |
|---|---|
| Dataset cannot be loaded | Check the dataset name, `data/processed` or `data/interim`, the format and the loader message. |
| Order or location requires review | Inspect `order_diagnostics["review_reasons"]`, `review_items`, `focus_diagnostics` or `night_diagnostics`. |
| P-value unavailable or group has no orders | Inspect `slot_test["checks"]` and group counts; an empty group's rate is undefined. |
| No suitable inter-state pair | Inspect `pair_profiles["coverage"]` and `pairs`; `leading_pair` may be `None`. |
| Missing/duplicate items, incomplete pair or invalid route label | Use complete cleaned items, check item keys, counts and route types, then recreate `pair_orders` for the same pair. |
| SP → SP was not selected for monthly analysis | Check `pair_months["selection"]`; its stage branch is skipped. |
| SP → SP was selected but has no common months | Inspect `pair_months["monthly"]` and `weights`; a common month needs orders in both slots. |
| Invalid stage duration | Inspect statuses and timestamps in `stage_profiles["records"]`. |
| Results differ from the report | Check the input and code; the report describes the reference dataset and notebook commentary is written manually. |

Run the analysis tests from the repository root.

```bash
python -m pytest tests/analysis -q
```

`build_focus_cohort` accepts `approval_slot`, defaulting to 12:00–18:00. Two-slot comparisons use `COMPARISON_SLOTS`; purchase periods in stage comparisons are defined in code. See the [Technical Reference](Technical_Reference.md) for details.
