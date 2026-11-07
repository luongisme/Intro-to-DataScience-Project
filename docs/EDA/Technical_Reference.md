# EDA Technical Reference

This reference documents the pre-cleaning inspection architecture, data contracts, function interfaces and execution dependencies. For executable examples, see the [EDA User Guide](User_Guide.md). Dataset observations belong in the [EDA Report](../../reports/EDA/EDA_Report.md).

Paths are relative to the repository root unless linked. EDA operates on in-memory data and does not clean, persist datasets or generate documentation.

## Contents

1. [Architecture](#1-architecture)
2. [Contracts and conventions](#2-contracts-and-conventions)
3. [Execution flow](#3-execution-flow)
4. [Profiling reference](#4-profiling-reference)
5. [Domain checks](#5-domain-checks)
6. [Notebook-facing workflow](#6-notebook-facing-workflow)
7. [Plot primitives](#7-plot-primitives)
8. [Artifacts and testing](#8-artifacts-and-testing)
9. [Maintenance and failure handling](#9-maintenance-and-failure-handling)

## 1. Architecture

```text
src/eda/
├── __init__.py      # Package marker; no re-exported API
├── profiling.py     # General, non-mutating summaries
├── checks.py        # Keys, entity consistency and domain diagnostics
├── plots.py         # Plot primitives accepting prepared data
└── workflow.py      # Notebook-facing tables, figures and investigations
notebooks/
└── 01_exploratory_data_analysis.ipynb
tests/eda/
├── test_eda_profiling.py
├── test_eda_checks.py
├── test_eda_plots.py
└── test_eda_notebook.py
```

| Dependency | Responsibility |
|---|---|
| [`src/data_loader.py`](../../src/data_loader.py) | `load_data()` reads the registered `merged_order_items` input |
| [`src/paths.py`](../../src/paths.py) | Derives repository-relative directories |
| [`pipeline/contracts/schema.py`](../../src/pipeline/contracts/schema.py) | Source selections, keys, role metadata and `FULL_MERGE_SCHEMA` |
| [`pipeline/contracts/validator.py`](../../src/pipeline/contracts/validator.py) | Shared key validation before constructing entity views |
| [`tests/conftest.py`](../../tests/conftest.py) | Synthetic source frames, `eda_full_frame`, plot cleanup |

Imports from `src.eda` do not re-export the functions below. Import from the defining module. `workflow.py` uses IPython display and closes figures after displaying them; the plot primitives leave figure lifetime to the caller.

Importing shared `src.paths` can create configured data/output directories. That shared initialization does not imply that EDA writes datasets, figures or report content.

## 2. Contracts and conventions

### 2.1. Full input and source-grain views

The notebook expects the registered 25-column full merge at item grain with key `(order_id, order_item_id)`. Its schema and typed reader are defined by the merge layer. The EDA helpers themselves validate the columns and types needed for each operation; `require_columns()` alone is not a full schema validator.

`SOURCE_SPECS` in `profiling.py` reuses the five merge source specifications. `build_entity_views()` returns `order_items`, `orders`, `customers`, `products` and `sellers`, retaining each source's selected columns and dtypes. Before collapsing repeated source values, it requires valid keys and consistent attributes. Views cover observed merged entities only.

### 2.2. Percentage and diagnostic rules

| Concept | Contract |
|---|---|
| Missing versus blank | Missing uses `isna()`; only actual empty/whitespace strings are blank |
| Repeated keys | Valid repeats of entity IDs are expected at item grain; item keys must remain unique for entity views |
| Attribute conflict | Different values within one ID, including value versus null; all-null attributes are consistent but incomplete |
| Numeric statistics | Real finite values only; identifiers, ZIP codes, bool and complex data are not measurements |
| Numeric spread | Linear quantiles; sample standard deviation (`ddof=1`); 1.5-IQR limits are descriptive |
| Frequency percentages | All input rows, including special Missing/Blank groups |
| Conditional missingness | Missing rows of the conditioning column; a zero denominator yields NaN |
| Nullable conditions | Missing endpoints remain unknown; evaluated and all-row percentages are separate |
| Timestamp precision | Stored clock; estimated-before-purchase uses calendar dates |
| Selection/window limits | Limit display or temporary diagnostic views without changing the input |

Index alignment is part of grouped-condition validation. Keep upstream objects from the same input and preserve row order. Some workflow results intentionally reference earlier inspection frames; treat all such references as read-only.

## 3. Execution flow

```text
load_data("merged_order_items", stage="interim")
    → capture notebook baseline
    → profile dataset / columns / keys / entity consistency
    → build validated entity_views
    → inspect distributions
    → inspect missing patterns → missing pairs / grouped context → status matrix
    → inspect text / timeline / measurements / composition
    → detailed investigations using earlier evidence
    → compare source with baseline
```

The notebook's local `find_project_root(start_path: Path) -> Path` searches the supplied directory and parents for `src/` plus `requirements.txt`; it raises `FileNotFoundError` if none matches. It sets the import root and does not change working directories.

No full-pipeline command or persistent EDA result object is defined. The notebook orchestrates named return values. Run it from top to bottom to reproduce all inspections.

## 4. Profiling reference

Source: [`src/eda/profiling.py`](../../src/eda/profiling.py). Functions return new summaries or calculation views without modifying inputs. In the tables below, `df` is a DataFrame, `series` a Series and `columns` an ordered column-name sequence.

### 4.1. Structure and numeric interfaces

| Signature | Purpose and inputs | Returns / notes |
|---|---|---|
| `require_columns(df, columns=())` | Check frame type, unique names and requested columns | `None`; TypeError for invalid types, ValueError for repeated names, KeyError for absent columns; permits extra columns |
| `blank_string_mask(series)` | Inspect original values without string coercion | Boolean Series with original index; nulls and nonstrings are not blanks |
| `get_column_roles()` | Read registered full-merge source metadata | Independent DataFrame in canonical column order, including source, grain, role and dtype |
| `profile_dataset(df)` | Profile dimensions, null cells/rows, duplicates and observed entities | DataFrame; distinguishes duplicate participants from excess copies |
| `profile_columns(df)` | Profile every supplied column | DataFrame of dtypes, distinct values, missing/blank counts and percentages; unregistered names have role `unregistered` |
| `finite_numeric_values(series)` | Select real finite measurements | Series excluding null/infinite values for calculation; rejects nonnumeric, bool, complex or registered nonmeasurement columns |
| `profile_numeric(df, columns)` | Summarize selected measurements at the caller's row grain | One row per measurement: completeness, finite/nonfinite/zero/negative counts, moments, quantiles, IQR limits and tail counts |

For empty/all-missing data, undefined statistics and rates remain NaN. Missing rates use all rows; IQR tail rates use finite values. Caller-selected grain is not inferred from a measurement name.

### 4.2. Frequencies, timestamps and missingness

| Signature | Purpose and inputs | Returns / notes |
|---|---|---|
| `profile_frequencies(series)` | Count every observed value, preserving case/whitespace | DataFrame sorted by count, with explicit Missing even when zero; blank values are identified separately |
| `limit_frequencies(table, top_n=15)` | Collapse a full, uncollapsed frequency table for display | Top values plus Other, Blank and Missing; preserves the original all-row denominator; rejects invalid limits/accounting |
| `profile_timestamp_coverage(df, columns)` | Summarize datetime coverage and precision | DataFrame of range, nulls, non-midnight and subsecond counts; does not parse strings or convert timezones |
| `profile_temporal_frequencies(series, frequency="month")` | Count datetime values by `month`, `date`, `weekday` or `hour` | Ordered frequency table; weekdays/hours include all 7/24 buckets; month/date include observed buckets; Missing remains separate |
| `profile_missing_patterns(df, columns=None)` | Count missing fields and exact selected-column patterns | Dict: `row_distribution`, `patterns`; `()` denotes a complete row; default selection is all columns |
| `profile_paired_missingness(df, columns=None)` | Count every unordered column pair and directional missingness | Dict: `pairs`, `conditional_pct`; matrix row is the conditioning missing field |

The four pair states sum to the supplied row count. `conditional_pct[row, column]` is `P(column missing | row missing) × 100`; its diagonal is undefined if that field is never missing.

### 4.3. Internal helpers

| Signature | Purpose and returns |
|---|---|
| `_frequency_table(series, counts)` | Attach explicit missing counts and all-row percentage accounting to prepared frequencies; returns a DataFrame |
| `_missing_mask(df, columns)` | Validate a nonempty selection and return its Boolean missingness frame |

These underscore-prefixed helpers support the public functions and are not application entry points.

## 5. Domain checks

Source: [`src/eda/checks.py`](../../src/eda/checks.py). No function loads, saves or cleans data.

| Signature | Purpose and inputs | Returns / constraints |
|---|---|---|
| `profile_keys(df, keys=None)` | Profile default source keys or a mapping of names to component sequences | DataFrame of missing, blank, invalid-union and repeated-valid-key counts; reports invalid key values without rejecting them; invalid declarations raise |
| `profile_entity_consistency(df)` | Compare selected order/customer/product/seller attributes within valid IDs | Dict: `summary`, `attributes`, `conflicts`; invalid IDs are reported separately; null-versus-value differences count as conflicts |
| `build_entity_views(df)` | Require complete unique item keys and consistent entity attributes | Dict of five independent source-grain DataFrames in first-occurrence order; refuses invalid entity IDs/conflicts |
| `summarize_condition_by_group(df, condition, group_columns)` | Summarize aligned bool/nullable-bool conditions by observed groups | DataFrame with rows, evaluated, unknown, affected and evaluated/all-row percentages; retains missing group keys; rejects misalignment or coercion |
| `profile_text_formats(df, columns, *, patterns=None, sample_size=5)` | Inspect original strings, optional full-match patterns and candidate variants | Dict: `summary`, `variants`, `examples`; examples bounded per check, variants retained fully; positional references disambiguate duplicate indexes |
| `profile_timestamp_ordering(orders, *, sample_size=5)` | Compare lifecycle and estimated timestamps once per validated order | Dict: `summary`, `conditions`, `examples`; six physical comparisons and one estimate-date check; equality allowed, missing endpoints unknown |
| `profile_order_composition(df)` | Inspect item numbering, products, sellers, deadlines and observed state pairs | Dict: `summary`, `orders`, `seller_deadlines`; validated item/entity structure; consecutive numbering means positive integer IDs exactly `1..N` |

Text regex checks use original strings and never coerce numeric ZIP values. Candidate strip/casefold keys identify possible spelling variants only. State equality is a comparison of observed strings, not a distance or route feature. Deadline ranges describe multiplicity without choosing an order-level deadline.

## 6. Notebook-facing workflow

Source: [`src/eda/workflow.py`](../../src/eda/workflow.py). Every function accepts in-memory frames or earlier results, displays its evidence, and returns the objects below. Displayed figures are closed afterward; there is no file I/O. Full profiles remain available even where examples or top groups are bounded.

Argument names in signatures identify the same notebook objects. Do not mix profiles from different datasets.

### 6.1. Distributions and missingness

| Signature | Purpose / required inputs | Return structure |
|---|---|---|
| `inspect_numeric_distributions(entity_views, column_roles)` | Six measurements at their source grain; full/window histograms and boxplots | Dict keyed by source, containing numeric DataFrames |
| `inspect_category_distributions(entity_views, column_roles)` | Source-grain category and ZIP frequencies | Dict: `frequency_tables`, `frequency_display_tables`, keyed by `(source, column)` |
| `inspect_temporal_distributions(entity_views, column_roles)` | Coverage and month/date/weekday/hour distributions | Dict: `timestamp_profiles`, `temporal_frequency_tables`, `temporal_display_tables`; calendar keys are `(source, column, frequency)` |
| `inspect_missing_patterns(df, entity_views, column_profile)` | Full-item and entity-view completeness/exact patterns | Dict: `missingness_frames`, `missingness_profiles`, `native_column_profiles` |
| `inspect_missing_pairs(missingness_frames, native_column_profiles)` | Full pairs plus displayed active-column heatmaps for full items, orders and products | Dict keyed by view, each with `pairs` and `conditional_pct` |
| `inspect_conditional_missingness(column_roles, df, entity_views)` | Missingness by status, purchase month, customer state, seller and product category | Dict: `order_timestamp_columns`, `core_timestamp_columns`, `product_measurement_columns`, `order_context`, `condition_profiles`, `condition_overview` |
| `inspect_status_missingness(condition_profiles, order_timestamp_columns)` | Timestamp/any-core missingness within each status | DataFrame: status-by-timestamp missing percentages |

`condition_profiles` keys are `(target, grain, grouping_tuple)`. `order_context` is a validated order view with customer state and a temporary purchase-month label. The four core timestamps exclude Estimated Delivery.

### 6.2. Initial domain inspections

| Signature | Purpose / required inputs | Return structure |
|---|---|---|
| `inspect_text_formats(column_roles, missingness_frames)` | Text format checks at full-item and entity grain | Dict keyed by view, each containing `summary`, `variants`, `examples` |
| `inspect_order_timeline(entity_views)` | Timestamp ordering and status-specific comparisons | Dict: `timeline_profile`, `timeline_by_status` |
| `inspect_measurement_plausibility(numeric_profiles, df)` | Measurement counts and item deadlines before purchase | Dict: `measurement_review`, `deadline_before_purchase`, `deadline_by_status` |
| `inspect_order_structure(df)` | Item/product/seller/deadline counts and composition frequencies | Dict: `order_composition`, `composition_frequency_tables` |

### 6.3. Detailed investigations

| Signature | Purpose / required inputs | Return structure |
|---|---|---|
| `inspect_timeline_details(entity_views, timeline_profile, order_context)` | Same-day versus earlier-day reversals, flag overlap and monthly evidence | `orders`, `flags`, `any_ordering_flag`, `precision_examples`, `timeline_precision`, `timeline_combinations`, `timeline_month_profiles`, `flagged_month_frequencies` |
| `inspect_delivered_completeness(orders, core_timestamp_columns, any_ordering_flag, order_context)` | Delivered-order missing-core patterns and chronology overlap | `delivered_orders`, `core_missing`, `delivered_missing_records`, `delivered_missing_profile`, `delivered_missing_patterns`, `delivered_issue_overlap`, `delivered_missing_by_month` |
| `inspect_product_completeness(entity_views, product_measurement_columns, df, orders)` | Missing/nonpositive measurements, product exposure and zero freight | `physical_issues`, `physical_issue_summary`, `physical_product_records`, `missing_physical_category`, `zero_freight`, `zero_freight_by_status`, `zero_freight_by_category` |
| `inspect_numeric_tails(numeric_profiles, df, entity_views)` | Existing IQR tails by category and bounded original extreme records | `iqr_group_profiles`, `numeric_tail_records`, `iqr_summary` |
| `inspect_deadline_calendar(orders, df, order_timestamp_columns)` | Deadlines beyond the latest full-input purchase year | `deadline_after_purchase_year`, `far_deadline_records`, `far_deadline_counts`, `far_deadlines_by_status` |
| `inspect_seller_city_spacing(entity_views, df, orders)` | Repeated seller-city spaces and original forms within state | `city_space_records`, `city_affected_items`, `city_form_counts` |
| `inspect_structure_overlap(order_composition, orders, df)` | Overlapping seller/deadline/state multiplicity with status and examples | `composition_context`, `composition_flags`, `composition_combinations`, `composition_by_status`, `composition_item_examples` |

All rows in the last column are dictionary keys. `delivered_orders` retains the validated order-view reference; `core_missing` is aligned to all orders, while delivered-specific records/patterns select the delivered cohort. Use the cohort-specific outputs for delivered-order counts.

Detailed-investigation prerequisites:

- `timeline_profile` comes from `inspect_order_timeline()["timeline_profile"]`.
- `order_context` and timestamp/measurement column lists come from `inspect_conditional_missingness()`.
- `any_ordering_flag` comes from `inspect_timeline_details()`.
- `order_composition` comes from `inspect_order_structure()["order_composition"]`.
- `orders` is `entity_views["orders"]`; `numeric_profiles` comes from numeric inspection.

Same-day reversals retain full timestamp precision. Flagged-cohort month frequencies are counts, not monthly failure rates. The future-year deadline comparison is descriptive in EDA, not an SLA threshold. Temporary delivered cohorts, comparison keys and diagnostic flags do not change the supplied input or become persisted features.

## 7. Plot primitives

Source: [`src/eda/plots.py`](../../src/eda/plots.py). Public functions return `(Figure, Axes)` and never call display/save or change global styles. Caller owns figure lifetime. Optional `ax` reuses an existing Matplotlib Axes.

| Signature | Input and purpose | Behavior / errors |
|---|---|---|
| `plot_frequency(table, *, top_n=15, horizontal=True, title="Value frequencies", ax=None)` | Prepared frequency table | Absolute counts with total; `top_n=None` preserves supplied order; special groups distinct from quoted literal strings; validates accounting |
| `plot_numeric_histogram(series, *, bins=40, value_range=None, xlabel=None, title="Measurement distribution", ax=None)` | Measurement Series and optional `(lower, upper)` display range | Finite values, explicit missing/nonfinite/outside-window counts; positive integer bins and finite increasing bounds required |
| `plot_numeric_boxplot(series, *, xlabel=None, title="Measurement boxplot", ax=None)` | Measurement Series | All finite values, 1.5-IQR whiskers and visible fliers; does not remove outliers |
| `plot_missingness_heatmap(matrix, *, title="Conditional missingness", ax=None)` | Unique-labeled numeric percentage matrix | Values must be 0–100 or NaN; undefined cells grey; empty matrices show no-data text; tiny rates retain boundary notation |
| `_get_axes(ax, figsize)` | Internal caller-Axes selection | Returns `(Figure, Axes)`; creates axes only when absent |

## 8. Artifacts and testing

| Artifact | Producer / role |
|---|---|
| `data/interim/merged_order_items.parquet` or `.csv` | Merge output read by EDA |
| Notebook tables and figures | EDA execution; retained in the kernel/output until cleared |
| `reports/EDA/EDA_Report.md` | Separately maintained narrative and selected observations |
| `reports/figures/EDA/` | Images referenced by the report, exported separately |
| `docs/EDA/User_Guide.md`, `Technical_Reference.md` | Maintained usage and implementation documentation |

The EDA notebook does not export the report or report images. Re-running it does not refresh these maintained artifacts.

| Test module | Coverage |
|---|---|
| [`test_eda_profiling.py`](../../tests/eda/test_eda_profiling.py) | Roles, selection, blanks/nulls, finite statistics, frequencies, timestamps and missingness denominators |
| [`test_eda_checks.py`](../../tests/eda/test_eda_checks.py) | Keys, entity conflicts, aligned nullable conditions, original text, chronology and order composition |
| [`test_eda_plots.py`](../../tests/eda/test_eda_plots.py) | Empty/constant data, special labels, ranges, accounting, undefined rates, axes and style preservation |
| [`test_eda_notebook.py`](../../tests/eda/test_eda_notebook.py) | Code compilation, no exports/additional sources, execution on a small full-table fixture and non-mutation |

Run from the repository root:

```powershell
python -m pytest tests/eda -q
```

Shared fixtures `synthetic_order_items`, `synthetic_orders`, `synthetic_customers`, `synthetic_products`, `synthetic_sellers` construct the merge-like input. `eda_full_frame` supplies its canonical full table; `eda_plot_cleanup` closes figures and restores plot state. Domain-check tests also define `timeline_orders` locally. Tests use synthetic edge cases rather than fixed production-row counts.

## 9. Maintenance and failure handling

Validate input columns before profiling and use source-grain views before entity statistics. Do not bypass attribute-conflict errors by dropping duplicates arbitrarily. Preserve null versus blank distinctions, named denominators, nullable comparisons, literal text and unknown percentages when extending a helper.

Type/selection errors are explicit (`TypeError`, `ValueError`, `KeyError` as appropriate). Loading failures originate in the shared typed reader. A missing prerequisite variable in the notebook is an execution-order issue; rerun from initialization. The preservation baseline checks shape, column order, dtypes, indexed values and row order.

When adding a check, place its reusable calculation in `profiling.py` or `checks.py`, its plot primitive in `plots.py` where appropriate, and its notebook presentation in `workflow.py`. Keep the notebook as orchestration. Add meaningful edge-case tests and update the corresponding interface entry without introducing automatic report generation.
