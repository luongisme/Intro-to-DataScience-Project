# Analysis Technical Reference

This reference describes the delivery analysis architecture, data contracts, functions and calculations. The [Analysis notebook](../../notebooks/03_analysis.ipynb) executes the workflow. The [User Guide](User_Guide.md) provides runnable examples, and the [Analysis Report](../../reports/Analysis_Report.md) presents results for the reference dataset.

Paths below are relative to the repository root. The notebook calls the functions and displays results in memory. Documentation and report figures are maintained separately.

## Contents

1. [Architecture and execution flow](#architecture)
2. [Data contracts](#data-contracts)
3. [Interfaces by workflow stage](#interfaces)
4. [Rules and calculations](#rules-and-calculations)
5. [Outputs and storage](#outputs-and-storage)
6. [Validation, tests and extension](#tests)

<a id="architecture"></a>
## 1. Architecture and execution flow

```text
notebooks/03_analysis.ipynb
src/
├── data_loader.py                 # Schema-aware dataset loading
├── paths.py                       # Repository paths
└── analysis/
    ├── __init__.py
    ├── orders.py                  # Order-level view and approval slots
    ├── summary.py                 # Delivery outcomes by slot
    ├── statistics.py              # Statistical test and Wilson intervals
    ├── plots.py                   # Overview figures
    ├── investigation.py           # Periods, stages, routes and focus cohorts
    ├── route_slots.py             # State pairs, slot and monthly summaries
    ├── route_stages.py            # Stage comparisons and evidence tables
    ├── slot_comparisons.py        # Two-slot location and month comparisons
    ├── investigation_plots.py     # Detailed analysis figures
    └── presentation.py            # Display formatting
tests/analysis/                    # Analysis tests
outputs/figures/Analysis/          # Saved report figures
```

Functions are imported from their defining modules. The notebook-local `find_project_root` locates the directory containing `src` and `requirements.txt`, adds it to the import path, and raises `FileNotFoundError` if none is found. It does not change the working directory.

Sources: [Analysis notebook](../../notebooks/03_analysis.ipynb), [analysis package](../../src/analysis/), [loader](../../src/data_loader.py).

<a id="execution-flow"></a>
### Execution flow

```text
cleaned_order_items
  → build_order_analysis_view → orders, diagnostics
  → summarize_delivery_by_slot → overall test and figures
  → summarize_time_periods / compare_cleaning_cohorts
  → investigate_delivery_stages → classify_order_routes → summarize_routes
  → 12:00–18:00 cohort → destinations, state pairs and timing
      → leading inter-state pair → recover the complete route across four slots
      → compare stages with other inter-state orders to the same destination
  → 00:00–06:00 cohort → route types, destinations and state pairs
  → compare the same groups across two slots → purchase months and common weights
  → revisit inter-state pair stages and examine SP → SP in common months
  → notebook tables, figures and conclusions
```

The leading inter-state pair selected in the afternoon is retrieved from **all `orders`** for the four-slot comparison. The overnight cohort is independently selected from the same order table. Group tables and figures are in-memory results. `format_analysis_table` changes their display only.

<a id="data-contracts"></a>
## 2. Data contracts

| Item | Contract |
|---|---|
| Main input | Item-level `cleaned_order_items`; the reference extract has 108,571 items across 25 columns and 95,075 delivered orders |
| Item key | `(order_id, order_item_id)`; the loader restores the registered schema and types |
| Order attributes | `order_status`, purchase, approval, handoff, customer-delivery and estimated-delivery timestamps must be complete and consistent across items when creating `orders` |
| Analysis grain | One row per valid order; missing or conflicting orders are retained for review |
| Approval slots | `[00:00, 06:00)`, `[06:00, 12:00)`, `[12:00, 18:00)`, `[18:00, 24:00)` from `order_approved_at` |
| Delivery outcome | `is_late=True` when the customer-delivery **date** is after the estimated date. Delivery on the same date is on time |
| `delay_days` | Calendar days past the estimated date for late orders; null for on-time orders |
| Late rate | `100 × late_orders / orders` within a group; undefined when there are zero orders |
| Shares | Group orders or late orders divided by the respective total in the stated scope |
| Purchase period and approval time | Month or period uses `order_purchase_timestamp`; slot, hour and weekday use `order_approved_at` |
| Promised window | Estimated delivery date minus approval date, in calendar days |

`build_order_analysis_view` relies on the loader contract and does not independently filter `delivered` status. Timestamps use the recorded clock without timezone conversion. Approval time is the observed anchor selected for the original delivery-slot question.

`merged_order_items` in `data/interim` is used to compare cohorts before and after cleaning.

<a id="interfaces"></a>
## 3. Interfaces by workflow stage

<a id="overall-analysis"></a>
### Overall, time and route analysis

| Function | Input and purpose | Output and checks |
|---|---|---|
| `build_order_analysis_view(items)` | `order_id` and the six order-level fields in section 2; checks missing or conflicting item values. | `(orders, diagnostics)`; diagnostics contain `cohort_summary`, `review_reasons`, `review_items`. Orders gain hour, weekday, slot, `is_late` and `delay_days`. Missing columns raise `ValueError`. |
| `summarize_delivery_by_slot(orders)` | `order_id`, `approval_slot`, `is_late`, `delay_days`; one row per order. | Four rows with order counts, on-time and late counts, rates, and mean and median delay days. Empty slots have zero counts and undefined measures. |
| `test_slot_association(summary, alpha=0.05)` | Order, on-time and late counts for all four slots. | `observed`, `expected`, `checks`, `result`. Checks counts and totals; p-value is unavailable for empty groups/outcomes or expected frequencies below five. |
| `summarize_time_periods(orders)` | Orders by purchase date and approval slot. | `monthly`, `matched_years`; matched years use January–July 2017 and 2018. |
| `compare_cleaning_cohorts(merged, orders)` | Items before cleaning and orders after cleaning. | `coverage`, `slots`; `slots` is empty if orders or comparison fields do not match. |
| `measure_stage_hours(start, end)` | Two timezone-naive datetime Series with matching index. | `calendar_hours`, `eligible_hours`, `status`; missing endpoints or reversed intervals have null durations. |
| `investigate_delivery_stages(orders)` | Unique order ID, slot, outcome, stage timestamps and promised date. | `records`, `summary`, `pooled_medians`, `quality`, `windows`, `promised_summary`, `negative_windows`. |
| `classify_order_routes(items, orders)` | Order IDs and seller/customer states for every item. | `order_id`, `route_type`, `items`; missing items or invalid order IDs raise an error. |
| `summarize_routes(orders, routes, stage_records)` | Orders, routes and stage records with matching IDs. | `routes`, `totals`, `slots`, `transit`; empty groups have zero counts and null rates or durations. |

`ROUTE_LABELS` contains `Intra-state`, `Inter-state`, `Mixed` and `Unknown`. An order is intra-state if every item has matching seller and customer states, inter-state if every item has different states, and mixed if it contains both. A missing state in any item yields `Unknown`. State strings are stripped before comparison. States serve as a geographic proxy and do not measure actual distance.

Stage durations cover `[start, end)`. Calendar hours count all elapsed time. Eligible-day hours count Monday–Friday excluding 1 January, 1 May and 25 December, with up to 24 hours per eligible day. This project calendar does not represent actual staffed hours or a complete Brazilian holiday calendar.

The Chi-square test assesses an overall difference across four slots. `cramers_v = sqrt(chi2 / N)`; `rate_range_pp` is the difference between the highest and lowest rates. Frequency checks do not establish independence between orders.

Sources: [orders.py](../../src/analysis/orders.py), [summary.py](../../src/analysis/summary.py), [statistics.py](../../src/analysis/statistics.py), [investigation.py](../../src/analysis/investigation.py).

<a id="afternoon"></a>
### Orders approved from 12:00–18:00

#### Cohort and destination

| Function | Input and purpose | Output and notes |
|---|---|---|
| `build_focus_cohort(items, orders, routes, *, approval_slot="12:00–18:00")` | Select a recognized slot. Items require `order_id`, `customer_state`; orders require ID, slot, outcome; routes require ID, type and item count. | `(focus, diagnostics)` with `summary`, `location_review`. Checks binary outcomes, complete items and matching counts. When `order_item_id` is present, scoped item keys must be complete and unique. A reduced item table without this column is accepted, subject to the count check. |
| `summarize_focus_routes(focus)` | Order ID, route type and binary outcome. | Four route-type rows with counts, rates and shares of the full focus cohort. |
| `summarize_destination_states(focus)` | ID, customer state, location status, route and outcome. | `all_routes`, `inter_state`; destination counts, rates and shares in each scope. |
| `summarize_state_pairs(items, focus, *, include_intra_state=False)` | Item ID and seller state; focus ID, customer state, location status, route, outcome and item count. | `orders`, `coverage`, `pairs`. Default scope is inter-state; `True` retains the complete focus cohort and considers eligible intra- and inter-state pairs. |

`location_status` is `known`, `missing`, `incomplete` or `conflicting`. A pair is ranked only when all items have the same complete seller state and a known customer state. Multiple-seller-state and unresolved orders remain in `coverage` without duplicating the order. Pair shares use the complete focus cohort with `include_intra_state=True`, and the inter-state cohort by default.

#### State pairs and timing

| Function | Input and purpose | Output |
|---|---|---|
| `summarize_focus_time(inter, leading_pair)` | Order-level inter-state data with purchase date, approval hour and weekday, location and outcome. Approval hour must be 12–17 and weekday 0–6. | `monthly`, `hours`, `weekdays`, `matched_years`; `None` summarizes inter-state orders only. |
| `summarize_focus_comparisons(inter, leading_pair)` | Unique inter-state orders with location, outcome and purchase date. | `same_destination`, `periods`, `concentration`; `None` yields empty tables. |
| `build_state_pair_orders(items, orders, routes, state_pair, *, include_intra_state=False)` | Complete items, analysis orders, routes and `(seller_state, customer_state)`. Item columns must include `order_item_id`. | Pair orders across all four slots. Checks IDs, route labels, item keys and counts, outcomes and purchase dates; intra-state pairs require `True`. `state_pair=None` returns an empty cohort after common validation. |

Destinations and pairs are ranked by descending late count, descending total orders, then ascending state code. The notebook selects the first inter-state pair with late deliveries. `same_destination` compares the pair with other inter-state orders **to the same customer state**; `periods` uses **all remaining inter-state orders**, including other destinations. Comparison groups do not overlap.

`build_focus_cohort` can select other slots, but `summarize_focus_time` is restricted to afternoon approval hours. The titles of `plot_destination_states` and `plot_state_pairs` in figures 13–14 are also fixed to 12:00–18:00. Use functions with a suitable slot contract when analyzing another cohort.

<a id="overnight"></a>
### Orders approved from 00:00–06:00

Reuse `build_focus_cohort(..., approval_slot="00:00–06:00")` with the complete `orders` and `routes`. `summarize_focus_routes` reports all four route types; `summarize_destination_states` returns all-route and inter-state tables.

Call `summarize_state_pairs(..., include_intra_state=True)` to rank eligible intra- and inter-state pairs. `orders` retains orders with multiple seller states, mixed routes or locations requiring review; `coverage` describes orders outside eligible pairs. Unless a table states an inter-state scope, extended-scope shares use the complete overnight cohort.

Shared functions require complete binary outcomes; route labels must match the four values described in the route analysis interface. Reference results in the report describe the current dataset and are not input requirements.

<a id="two-slot"></a>
### Two-slot comparisons

Source: [slot_comparisons.py](../../src/analysis/slot_comparisons.py). `COMPARISON_SLOTS` contains 00:00–06:00 and 12:00–18:00.

`compare_approval_slot_groups(night, afternoon, *, group_by, top_n=5, rate_top_n=3, min_orders=100, reference_pair=("SP", "RJ"))` accepts two order-level tables with their correct slots and unique, non-null, disjoint IDs. `group_by` is `customer_state` or `state_pair`. For pairs, use `orders` from `summarize_state_pairs(..., include_intra_state=True)`.

| Output | Content |
|---|---|
| `scope_summary` | Total orders and late orders, rate, Wilson interval and difference for each full slot |
| `all_groups` | Each `group × approval_slot`, including review groups: counts, rate, Wilson interval, shares and difference |
| `selection` | Selected groups and reasons |
| `selected` | Corresponding rows from `all_groups` |
| `plot_data` | Selected groups, a remainder pooled from counts and special location groups |

Default selection combines up to five leaders by **late count** in each slot, three leaders by **rate** with at least 100 orders and one late order in each slot, the largest nonzero absolute rate difference with at least 100 orders in both slots, and the reference pair when present in pair comparisons.

Count ties use total orders then code; rate ties use late counts, total orders then code; gap ties use pooled orders then code. Counts, rates and totals sort descending, codes ascending. Ranking uses unrounded rates. Unselected groups remain in `all_groups`; the plotted remainder is recomputed from pooled counts.

`difference_pp` is the **overnight minus afternoon** rate in percentage points. Shares use the full-slot order or late-order total. Empty groups have null rates and Wilson intervals. Wilson 95% intervals assume independent orders; overlapping intervals do not by themselves establish whether rates differ.

<a id="rules-and-calculations"></a>
## 4. Rules and calculations

<a id="standardization"></a>
### Purchase months and standardization

#### Two-slot comparison within each group

`summarize_slot_group_months(night, afternoon, *, group_by, min_orders=100, reference_pair=("SP", "RJ"))` takes the two-slot comparison tables with complete purchase datetimes. It selects one late-count leader and one rate leader meeting the threshold in each slot, the largest qualifying gap, and the reference pair when applicable. This yields at most five states or six pairs. `min_orders` and `reference_pair` are parameters; the two slots and number of monthly leaders are fixed.

| Output | Key and content |
|---|---|
| `selection` | `group`, selection reason |
| `monthly` | `group`, `purchase_month`, `approval_slot`; counts and rate, including empty month cells |
| `standardized` | `group`, `approval_slot`; common months, included/excluded orders and late orders, full-cohort raw, common-month raw and standardized rates |
| `weights` | `group`, `purchase_month`; pooled two-slot orders and weight |
| `differences` | One row per group with common months and differences for the three rates |

For each group, common months have at least one order in **both slots**. A month’s weight is its pooled two-slot orders divided by pooled orders in all common months. A slot’s standardized rate is the sum of its monthly rates multiplied by those shared weights. No common months produce an undefined standardized rate; one common month limits the comparison to that month. Full-cohort raw rates retain orders outside common months.

#### Four-slot comparison for one pair

`summarize_state_pair_slots(pair_orders)` takes one row per pair order with ID, slot, outcome and purchase date. It returns `raw`, `monthly`, `standardized`. `raw` contains rate and Wilson limits; `monthly` contains counts by purchase month × slot; `standardized` contains raw and standardized rates and differences from 12:00–18:00.

Common months here require orders in **all four slots**. Weights use pooled orders from all four slots in those months. `Interior common months` excludes the first and last observed months and recomputes weights. The two standardizations can differ because their scopes and weights differ. For `p = late_orders / orders`, `n = orders` and `z ≈ 1.95996`, the Wilson center is `(p + z²/(2n))/(1 + z²/n)` and its half-width is `z × sqrt(p(1-p)/n + z²/(4n²))/(1 + z²/n)`. The limits are expressed as percentages.

Sources: [route_slots.py](../../src/analysis/route_slots.py), [statistics.py](../../src/analysis/statistics.py).

<a id="stages"></a>
### Delivery stages and promised windows

#### Inter-state pair and destination-matched group

`compare_route_delivery_stages(items, orders, routes, pair_orders, stage_profiles, state_pair)` compares an inter-state pair with other inter-state orders to the same customer state by purchase period and approval slot. `pair_orders` must contain exactly the complete cohort whose every item belongs to the pair. `records` and `windows` must cover the analysis orders. `PURCHASE_PERIODS` fixes November–December 2017, February–March 2018 and April–July 2018. `STAGES` contains `Processing` and `Carrier transit`.

| Table | Key | Contents |
|---|---|---|
| `coverage` | `period` | Counts of inter-state orders, missing destinations, orders to the destination, selected pair and other inter-state orders to that destination |
| `outcomes` | `period`, `group`, `approval_slot` | `orders`, `late_orders`, `late_rate_pct` |
| `stages` | `outcomes` key + `stage` | Order and valid/unavailable interval counts, median and P75 under both hour conventions |
| `by_outcome` | `stages` key + `is_late` | Stage summaries split by observed delivery outcome |
| `promises` | `outcomes` key | Orders, available/unavailable counts, median, P75 and negative promised windows |

The output grid retains empty cells with zero counts and null rates or medians. Route labels must be valid. Missing or duplicated items, an incomplete pair, mismatched item counts, invalid timestamps or durations raise `ValueError`. A multiple-seller-state order may belong to the comparator if it is inter-state, all items share the destination and it is outside the selected pair.

#### Stage evidence by slot

`summarize_route_slot_evidence(comparison, *, approval_slots=(SLOT_LABELS[0], SLOT_LABELS[2]))` joins `outcomes`, `stages` and `promises` by `period`, `group`, `approval_slot`. Output includes order and late counts, rate, median duration for each stage and median promised window. It checks keys, both stages, consistent nonnegative integer counts and rates matching counts. Zero orders require a null rate. `summarize_afternoon_route_evidence(comparison)` selects the afternoon and drops the slot column.

`summarize_stage_slots(records, *, approval_slots=SLOT_LABELS)` uses `records` from `investigate_delivery_stages`. Each order must have both stages in one slot. It returns one row per `approval_slot × stage` with `orders`, `valid_intervals`, `unavailable_intervals`, median and P75 calendar and eligible-day hours. The default covers four slots; a tuple of valid slots can be supplied.

`status="valid"` requires finite, nonnegative durations and eligible hours no more than `1e-9` hours above calendar hours. `missing_endpoint` or `reversed` requires null durations. P75 uses linear interpolation over valid intervals. Stage medians cannot be added to obtain the median end-to-end duration.

The notebook selects SP → SP orders in shared months across the two slots and summarizes stages over **all orders in each slot**, without monthly weighting. `summary` from `investigate_delivery_stages` splits orders by observed delivery outcome. SP → RJ uses the destination-matched comparison across the three periods. If SP → SP was not selected for monthly analysis, the notebook reports that selection status; if selected without common months, it reports the absence of common months.

Source: [route_stages.py](../../src/analysis/route_stages.py).

<a id="figures"></a>
### Presentation and figures

`format_analysis_table(table)` returns a `Styler` with column labels and formatting for counts, rates, percentage points, durations and missing values. The DataFrame used in calculations keeps its values and types. Figure labels or captions define each N denominator.

| Figure function | Main input | Figure and contents |
|---|---|---|
| `plot_slot_counts(summary)` | Four-slot summary | 01: on-time and late order counts |
| `build_temporal_profiles(orders)`, `plot_rate_profile(profile, column)` | Hour/weekday order profile | 02–03: rate and N by hour/weekday |
| `plot_hour_weekday_heatmap(cells)` | `hour_weekday` | 04: rate and count by hour × weekday |
| `plot_delay_distribution(orders)` | Analysis orders | 05: late-day distribution; P99 limits the viewing range |
| `plot_monthly_slot_rates(monthly)` | Purchase month × slot | 06: rate and count by month |
| `plot_matched_years(table)` | `matched_years` | 07: matched January–July periods |
| `plot_stage_comparison(summary, stage)` | Stage summary | 08–09: medians by slot and delivery outcome |
| `plot_promised_windows(windows)` | Promised windows | 10: days from approval to estimated date |
| `plot_route_slot_rates(slots)` | Route type × slot | 11: intra- and inter-state rates |
| `plot_route_transit_time(transit)` | Durations by route | 12: median transit |
| `plot_destination_states(inter_state, top_n=5)` | Inter-state customer states | 13: late counts, rates and N for selected states |
| `plot_state_pairs(pairs, inter_late_rate, top_n=5)` | Inter-state pairs and reference rate | 14: late counts, rates, N and total inter-state rate |
| `plot_focus_months(monthly)` | Afternoon monthly profile | 15: rate and count by purchase month |
| `plot_focus_approval_time(hours, weekdays)` | Hour and weekday tables | 16: late counts and rates by separate approval-hour and weekday groups |
| `plot_state_pair_slots(raw, state_pair)` | Pair's four-slot raw table | 17: counts and rates by slot |
| `plot_state_pair_months(monthly, state_pair)` | Purchase month × four slots for a pair | 18: rates and counts |
| `plot_route_period_medians(table, value, title, ylabel)` | Stage or promised window by period | 19–21: median by period × group × slot |
| `plot_route_period_slot_rates(outcomes)` | Outcomes by period × group × slot | 22: two rate panels on a shared color scale |
| `plot_focus_routes(routes, approval_slot)` | Four route-type rows for a cohort | 23: on-time and late counts and rate |
| `plot_slot_group_comparison(table, group_by)` | Two-slot `plot_data` | 24–25: late/all counts, rates and Wilson intervals, including remainder |
| `plot_slot_group_months(monthly, *, group_by, groups=None)` | Monthly table, group type and displayed groups | 26–27: rate and order count by purchase month, with a color per slot |

The plotting functions return `Figure`; callers display and close it. Count and rate checks use floating-point arithmetic to avoid small-integer overflow. `groups` limits monthly figures while `monthly` retains all selected groups. A group with zero orders has an undefined rate. Figures 24–25 display late/total counts and Wilson intervals to show volume and uncertainty.

Figures 13–14 have titles fixed to inter-state orders approved from 12:00–18:00. Figures 04 and 22 use green for late rates; figure 04 uses blue for order counts. Figures 26–27 use consistent colors for the two approval slots. Empty groups or cells have no defined rate and are identified in the respective figure.

Sources: [plots.py](../../src/analysis/plots.py), [investigation_plots.py](../../src/analysis/investigation_plots.py), [presentation.py](../../src/analysis/presentation.py).

<a id="outputs-and-storage"></a>
## 5. Outputs and storage

Runtime tables and figures remain in memory. The saved report figures and separately maintained documentation are listed below.

| Component | Location or use |
|---|---|
| Main input | `data/processed/cleaned_order_items`; loader prefers Parquet then CSV, with direct CSV selection available |
| Cleaning comparison | `data/interim/merged_order_items` |
| Runtime tables and figures | In-memory DataFrames, Stylers and Figures |
| Report figures | `outputs/figures/Analysis/figure_01_*.png` through `figure_27_*.png` |
| Documentation | User Guide, Technical Reference and Report are authored separately |

<a id="tests"></a>
## 6. Validation, tests and extension

| Test file | Coverage |
|---|---|
| [test_orders.py](../../tests/analysis/test_orders.py), [test_summary.py](../../tests/analysis/test_summary.py) | Order grain, slots, missing/conflicting values, counts and empty groups |
| [test_statistics.py](../../tests/analysis/test_statistics.py), [test_plots.py](../../tests/analysis/test_plots.py) | Test, intervals and overview figures |
| [test_investigation.py](../../tests/analysis/test_investigation.py), [test_routes.py](../../tests/analysis/test_routes.py) | Cohorts, locations, stages, calendar convention and routes |
| [test_route_slots.py](../../tests/analysis/test_route_slots.py) | State pairs, item keys, common months and four-slot weights |
| [test_route_stages.py](../../tests/analysis/test_route_stages.py) | Pair/item coverage, route labels, durations, tolerance, P75 and stage tables |
| [test_slot_comparisons.py](../../tests/analysis/test_slot_comparisons.py) | Group selection, denominators, Wilson, common months, weights and two-slot figures |

Run from the repository root:

```bash
python -m pytest tests/analysis -q
```

When inputs or logic change, run the notebook in order and reconcile counts, denominators, figures and static commentary. Interpretations should account for route or period selection after observation, orders sharing sellers or dates, and small groups. Month standardization does not control all within-month differences or determine the effect of changing approval times.
