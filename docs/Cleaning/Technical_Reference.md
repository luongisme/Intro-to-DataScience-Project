# Cleaning Technical Reference

This reference documents cleaning contracts, whole-order eligibility, function interfaces, display helpers and storage integration. For executable examples, see the [Cleaning User Guide](User_Guide.md). Dataset observations are maintained in the [Cleaning Report](../../reports/Cleaning/Cleaning_Report.md).

Paths are relative to the repository root unless linked. The workflow separates preparation, diagnostics, order filtering, output validation and explicit persistence. It does not create analysis or machine-learning features.

## Contents

1. [Architecture](#1-architecture)
2. [Data contracts](#2-data-contracts)
3. [Execution flow](#3-execution-flow)
4. [Workflow entry points](#4-workflow-entry-points)
5. [Transformations and eligibility](#5-transformations-and-eligibility)
6. [Validation reference](#6-validation-reference)
7. [Presentation and visual review](#7-presentation-and-visual-review)
8. [Storage and artifacts](#8-storage-and-artifacts)
9. [Testing and maintenance](#9-testing-and-maintenance)

## 1. Architecture

```text
src/cleaning/
├── __init__.py      # Selected public exports
├── schema.py        # Inspection/final schemas and required timestamps
├── text.py          # Source text checks and city whitespace normalization
├── measurements.py  # Nonpositive physical cells and quality flag
├── timeline.py      # Core chronology and deadline-review flags
├── filtering.py     # Whole-order eligibility with exclusion evidence
├── validation.py    # Input, allowed-change, cohort and roundtrip checks
├── workflow.py      # Inspect, clean and explicitly save
├── presentation.py  # Display-only labels and scalar formatting
└── plots.py         # Six figures and supporting profile tables
notebooks/
└── 02_data_cleaning.ipynb
tests/cleaning/
├── test_cleaning.py
└── test_cleaning_plots.py
```

| Shared dependency | Responsibility |
|---|---|
| [`src/data_loader.py`](../../src/data_loader.py) | Registered interim/processed loading and protection against schema overrides |
| [`src/paths.py`](../../src/paths.py) | `INTERIM_DIR` input and `PROCESSED_DIR` output; derives paths from the repository |
| [`src/eda/checks.py`](../../src/eda/checks.py) | Source-grain views, key/attribute checks and temporal diagnostics |
| [`src/eda/profiling.py`](../../src/eda/profiling.py), [`plots.py`](../../src/eda/plots.py) | Reused numeric/frequency profiles and plot primitives |
| [`pipeline/contracts/schema.py`](../../src/pipeline/contracts/schema.py) | `TableSchema` and the original full-merge schema |
| [`pipeline/contracts/validator.py`](../../src/pipeline/contracts/validator.py) | Shared column, dtype, key and non-null checks |
| [`pipeline/io/dataset_io.py`](../../src/pipeline/io/dataset_io.py) | Schema-aware Parquet/CSV read/write and fallback |

`__init__.py` re-exports workflow functions, cleaning transforms, selected validators, display formatting and the clean name/schema. Baseline/input-validation helpers and plotting functions are imported from their defining modules. Plotting is not re-exported through the package root, avoiding an eager Matplotlib import for loader-only use.

## 2. Data contracts

Source: [`src/cleaning/schema.py`](../../src/cleaning/schema.py). This module declares constants and schemas; it defines no functions.

| Contract | Columns and requirements |
|---|---|
| `FULL_MERGE_SCHEMA` | Registered 25-column source, imported from merge; item key `(order_id, order_item_id)` |
| `FLAGGED_ORDER_ITEMS_SCHEMA` | Original 25 columns plus three non-null bool flags; all items/statuses retained at inspection stage |
| `CLEANED_ORDER_ITEMS_SCHEMA` | Original 25 columns/order/dtypes; same item key; additionally non-null status and five required timestamps |
| `CLEANED_DATASET_NAME` | `cleaned_order_items` |
| `REQUIRED_TIMESTAMPS` | Purchase, Approved, Carrier Handoff, Customer Delivery and Estimated Delivery |
| `QUALITY_FLAG_COLUMNS` | `has_invalid_physical_measurement`, `has_invalid_timeline`, `needs_deadline_review` |

The final dtype declarations are **12 string, 6 datetime64[us], 6 float64 and 1 int64**. Canonical field names remain unchanged. ZIP prefixes remain strings. The final schema permits optional product-category/physical nulls. Behavioral eligibility, complete-order retention and allowed values are enforced by `validate_cleaning_output()`, beyond the structural schema checks.

Input validation rejects an empty source. A nonempty valid source may produce an empty retained result; final validation, figures and storage support that result.

### 2.1. Policy constants

| Module / declaration | Meaning |
|---|---|
| `text.CITY_COLUMNS` | Customer/seller city fields with their entity IDs |
| `text.STRING_ID_COLUMNS`, `TEXT_PATTERNS` | Four 32-character lowercase hexadecimal IDs and two five-ASCII-digit ZIP prefixes |
| `text.STATE_CODES`, `ORDER_STATUSES`, `TEXT_DOMAINS` | Fixed exact source domains: 27 state codes and eight statuses, including raw `created` |
| `measurements.PHYSICAL_COLUMNS` | Weight, length, height and width |
| `measurements.INVALID_PHYSICAL_FLAG` | `has_invalid_physical_measurement` |
| `timeline.CORE_TIMESTAMP_COLUMNS` | Purchase → Approved → Carrier Handoff → Customer Delivery |
| `timeline.ADJACENT_TIMESTAMP_CHECKS` | Approved before purchase; carrier before approval; customer before carrier |
| `timeline.INVALID_TIMELINE_FLAG`, `DEADLINE_REVIEW_FLAG` | Names of the temporal quality flags |
| `filtering.EXCLUSION_REASONS` | Status → required timestamps → chronology → physical values → deadlines |

City cleanup preserves case/spelling/nulls. Category text is not translated or standardized in the stored data. Deadline-year review uses the latest purchase year in the full input, not an order-specific year or a reference recomputed after exclusions.

## 3. Execution flow

```text
full merged input
    → capture baseline and validate input
    → normalize city whitespace
    → replace physical cells <= 0 with null and flag affected items
    → flag adjacent core timestamp reversals
    → flag deadline review conditions using the full input
    → validate the 28-column inspection result and source preservation
    → aggregate item-level reasons to orders and exclude complete orders
    → select the original 25 columns
    → independently validate cohort, retained values and source preservation
    → return cleaned data plus diagnostics
    → caller optionally builds review figures and explicitly saves/reloads
```

Missing timestamps are not chronology violations by themselves. They are unknown comparisons in diagnostics and a separate exclusion reason in the eligibility stage. A pre-existing optional physical null differs from an observed nonpositive measurement and is retained when the order otherwise qualifies.

The notebook calls these reusable interfaces and displays their evidence. Its local `find_project_root(start_path: Path) -> Path` searches the supplied directory and parents for `src/` plus `requirements.txt`, raising `FileNotFoundError` if none matches. It does not change the working directory.

## 4. Workflow entry points

Source: [`src/cleaning/workflow.py`](../../src/cleaning/workflow.py). Inputs below are pandas DataFrames; frame-transforming functions return independent data rather than mutating the source.

### `inspect_order_items(df) -> (inspected, details)`

**Purpose:** prepare and inspect the complete valid 25-column full merge, before order exclusions.

**Returns:** a 28-column DataFrame preserving every item, index and relative row order, plus seven diagnostic DataFrames:

| Key | Contents / grain |
|---|---|
| `text_changes` | Per-city-column whitespace change counts on the full input |
| `measurements` | Per-measurement missing/replacement counts and affected products/orders |
| `timeline` | Per-adjacent-check distinct-order chronology, comparable and unknown counts |
| `core_timestamps` | Distinct-order completeness of the four core timestamps |
| `deadlines` | Two item-level review checks with affected order/order-seller counts |
| `deadline_records` | Every flagged original item record with nullable per-check conditions |
| `inspection_checks` | Successful allowed-change/flag checks, including source preservation |

**Behavior/errors:** validates the source, applies the four preparation/flagging functions in order, verifies allowed changes and compares the baseline. Raises on invalid input, overwritten/inconsistent flags or unapproved changes. Does not load, display or save data.

### `clean_order_items(df) -> (cleaned, details)`

**Purpose:** execute the complete validated workflow on the original full input.

**Returns:** the eligible 25-column table and the seven inspection diagnostics above, extended with five more DataFrames:

| Key | Contents / grain |
|---|---|
| `filter_reasons` | Independent affected-order/item counts for each reason; groups may overlap |
| `filter_flow` | Input row plus sequential newly removed and remaining order/item counts |
| `excluded_orders` | One row per removed order, item count and all reason flags |
| `dataset_summary` | Input/retained/removed item rows, orders and columns |
| `output_checks` | Successful final-cohort/value/schema checks and source preservation |

**Behavior/errors:** calls inspection, filters entire orders, removes flags and validates the exact retained result independently against source. Keeps original index labels and relative row order. Does not load, display or save data. Do not pass the 28-column inspection result or a previously filtered subset as the source for reproducing the full workflow.

### `save_cleaned_data(df, output_dir=None) -> Path`

**Purpose:** save the final 25-column result under `CLEANED_DATASET_NAME`.

**Inputs:** final DataFrame; optional string/Path output directory, defaulting to `PROCESSED_DIR`.

**Returns/behavior:** actual successful Parquet or CSV path. Delegates structural validation, atomic writes, fallback and error guidance to the shared saver. This validates the clean schema; the source-based cohort check remains the responsibility of `clean_order_items()` or `validate_cleaning_output()`. A custom directory supports isolated runs. No metadata sidecar or report is written.

## 5. Transformations and eligibility

### 5.1. Text and measurements

Sources: [`text.py`](../../src/cleaning/text.py), [`measurements.py`](../../src/cleaning/measurements.py).

| Signature | Inputs / purpose | Returns and behavior |
|---|---|---|
| `inspect_text_fields(df)` | Required text fields plus item keys | Dict: `summary`, `issues`; nine checked columns; separate missing/invalid-observed counts; every invalid cell includes position, index, key and original `repr` |
| `normalize_city_whitespace(df)` | City fields with their entity IDs, string dtypes | `(copy, summary)`; trims edges/collapses whitespace only; per-field item/entity counts; whitespace checks can overlap |
| `clean_physical_measurements(df)` | Physical fields and product/order IDs | `(copy, summary)`; observed values `<= 0` become missing; bool `has_invalid_physical_measurement` records any changed measurement per item |

Text inspection does not coerce, change case, infer codes or repair values. `validate_cleaning_input()` rejects invalid observed formats/domains; optional missing attributes are counted separately. Measurement preparation preserves pre-existing nulls, other values, dtypes, index and order. Statistical tails are not invalid-measurement rules.

### 5.2. Timeline and deadlines

Source: [`timeline.py`](../../src/cleaning/timeline.py).

| Signature | Inputs / purpose | Returns and behavior |
|---|---|---|
| `flag_timeline_issues(df)` | Item table with core datetime fields and consistent entities | `(copy, timeline_summary, core_summary)`; adjacent comparisons use validated distinct orders; any confirmed reversal flags all its items |
| `flag_deadline_issues(df)` | Full item input with original shipping deadlines/purchase times | `(copy, deadline_summary, flagged_records)`; adds `needs_deadline_review`; preserves original timestamps |

Equality is allowed. Missing endpoints remain unknown in summaries and do not alone set the bool flags. Deadline checks are `shipping_limit_date < order_purchase_timestamp` at full precision and deadline year greater than the full-input latest purchase year. Without any observed purchase, the year condition is unknown for every item. The latter condition is a review signal, not an SLA or proof of an invalid date; the current strict eligibility policy excludes observed review flags.

No timestamp replacement, order-level deadline selection, business-day durations, lateness labels or time-slot features occur. Calls must not silently overwrite existing quality flags.

### 5.3. Whole-order eligibility

Source: [`filtering.py`](../../src/cleaning/filtering.py).

`filter_eligible_orders(df) -> (retained, exclusions)` requires `FLAGGED_ORDER_ITEMS_SCHEMA` with quality flags already calculated on the full input. It groups reason flags by order using any-item logic, then removes every item of an ineligible order.

An order fails for non-delivered/missing status, missing required timestamp, or any of the three quality flags. Optional missing attributes remain allowed. The retained frame preserves the 28 inspection columns, dtypes, index labels and relative row order. Final column selection occurs in `clean_order_items()`.

| `exclusions` key | Contract |
|---|---|
| `reasons` | `reason`, `affected_orders`, `affected_item_rows`; includes all items of affected orders; reasons overlap |
| `flow` | `step`, `removed_orders`, `removed_item_rows`, `remaining_orders`, `remaining_item_rows`; disjoint sequential removals |
| `excluded_orders` | Order ID, `item_rows` and every applicable reason flag |

Supports empty inspected input or an empty retained result. Rejects an incompatible inspection schema. It does not recalculate flags, display, select final columns or save.

## 6. Validation reference

Source: [`validation.py`](../../src/cleaning/validation.py).

| Signature | Purpose and inputs | Returns / failure behavior |
|---|---|---|
| `capture_input_baseline(df)` | Record shape, columns, dtypes and indexed row hashes | Dict with `shape`, `columns`, `dtypes`, `row_hashes`; rejects non-DataFrame or repeated columns |
| `validate_cleaning_input(df)` | Nonempty full merge, schema/key/non-null checks, exact text patterns/domains and entity consistency | Dict of `key_profile`, `entity_consistency`, `attribute_consistency`, `conflicts`, `text_formats`; raises for invalid observed text/IDs or conflicting attributes |
| `assert_input_unchanged(df, baseline)` | Compare source with a captured baseline | `None`; raises on shape, column order, dtype, indexed value or row-order changes |
| `validate_inspected_output(source, inspected)` | Verify original items/index/order/dtypes, permitted preparation and all three flags | Passed-check DataFrame; only city whitespace changes and original nonpositive physical cells becoming null are allowed |
| `validate_cleaning_output(source, cleaned)` | Derive eligibility independently from original observations and full-input year reference | Passed-check DataFrame; rejects incomplete orders, wrong retained cohort/order/schema/values; only retained city whitespace may differ |
| `assert_cleaned_roundtrip(expected, restored)` | Compare persisted clean result with its in-memory version | `None`; exact values, columns, dtypes and row order required; ignores only pandas index labels |

Allowed-change/final validators compare against the supplied original source; separately check that source against its earlier baseline. In-memory validation preserves original index labels, but the storage contract persists item keys and row order rather than pandas labels. Assertions and validation exceptions are failures to investigate, not instructions to cast or discard data until checks pass.

## 7. Presentation and visual review

### 7.1. Display formatting

Source: [`presentation.py`](../../src/cleaning/presentation.py).

| Signature | Purpose / inputs | Returns / notes |
|---|---|---|
| `format_cleaning_preview(df, *, rows=10)` | Bound a DataFrame preview and apply readable labels/formats | New string-valued DataFrame with RangeIndex; positive integer row limit; rejects repeated columns; never use this copy for computation/storage |
| `_format_number(value)` | Internal real-number formatting | Thousands separators; preserves useful noninteger precision as text |
| `_format_city(value)` | Internal city display casing | Capitalized words with selected Portuguese connectors in lowercase; no invented accents |
| `_format_value(value, column)` | Internal scalar formatting by field | Missing, Yes/No, dates, BRL amounts, percentages and known field labels as strings |

`COLUMN_LABELS`, `EXCLUSION_LABELS`, `DISPLAY_ORDER`, `CURRENCY_COLUMNS`, `DATETIME_COLUMNS` and `PERCENT_COLUMNS` control display only. Canonical dataset names/values remain unchanged. Source table fields are grouped for readable previews; diagnostic columns not in the display ordering retain their relative order.

### 7.2. Figure builder

Source: [`plots.py`](../../src/cleaning/plots.py).

`build_cleaning_visuals(source, cleaned, details) -> {"figures": ..., "tables": ...}` first validates the exact final cohort against source and checks that `details["filter_flow"]` matches source/final order totals. It profiles the retained data at its appropriate grain and returns six Matplotlib Figure objects without display, closing, export or mutation. Temporary style settings use `rc_context`.

| Figure key | Panels and counting unit |
|---|---|
| `filtering` | Retained/excluded orders and disjoint rule counts |
| `missingness` | Before/after item missing percentages and counts |
| `numeric` | Price/freight per item and weight per product; full and p01–p99 views |
| `calendar` | Purchase months with zero-count calendar gaps; all 24 approval hours per order |
| `categories` | Product-category frequencies, top ten plus special groups |
| `geography` | Separate customer-record/seller state frequencies, top ten plus Other |

The `tables` mapping contains `missingness`, `missingness_display`, `numeric`, `purchase_months`, `approval_hours`, `categories`, `category_display`, `customer_states`, `customer_state_display`, `seller_states`, `seller_state_display`. Numeric tables include all six measurements even though the figure plots three. Complete frequency tables remain available beside compact displays.

### 7.3. Plot/profile interfaces

| Signature | Inputs / purpose | Returns / constraints |
|---|---|---|
| `plot_filtering_flow(flow)` | Sequential order counts including input row | Figure; nonnegative counts and total reconciliation required |
| `profile_missingness_comparison(source, cleaned)` | Matching-column source/final frames | `(full, compact)` DataFrames; before/after use their own item denominators; physical fields group only for identical count pairs; grouped count is per field |
| `plot_missingness_comparison(table, before_rows, after_rows)` | Compact missingness profile and denominator sizes | Figure; empty retained percentages shown as undefined |
| `plot_retained_numeric(cleaned, products, numeric)` | Final item data, validated distinct-product view and six-field numeric profile | Figure with full/window histograms; windows do not filter data |
| `profile_retained_calendar(orders)` | Distinct retained orders with purchase/approval datetimes | `(months, hours)` DataFrames; full month span and all 24 hours; supports empty retained input |
| `plot_retained_calendar(months, hours)` | Tables from the calendar profiler | Figure of order volume; no outcome labels or timezone conversion |
| `_plot_frequency_review(full, title, ax=None)` | Internal full frequency table, title and optional Axes | `(Figure, display_table)`; top ten plus explicit special groups/count labels; reuses EDA accounting |

Color constants `BEFORE_COLOR`, `AFTER_COLOR`, `REMOVED_COLOR` distinguish source, retained and excluded counts. Callers own figure lifetime: the notebook closes pyplot registrations before explicitly displaying the returned Figure objects. No image or Markdown file is written by these functions.

## 8. Storage and artifacts

The shared `load_data(name, stage="raw", **kwargs) -> DataFrame` dispatches processed reads through the registered `CLEANED_ORDER_ITEMS_SCHEMA`. Saved-stage options are `file_format` and optional `schema`; unsupported parser options and conflicting registered-schema overrides are rejected. See the [Merge storage reference](../Merge/Technical_Reference.md#io-reference) for shared codec/atomic implementation details.

| Artifact | Location / producer |
|---|---|
| Full source | `data/interim/merged_order_items.parquet` or `.csv`; produced by merge |
| Final clean data | `data/processed/cleaned_order_items.parquet` or `.csv`; explicit save only |
| Intermediate flags and diagnostics | In-memory return values; not persisted as metadata sidecars |
| Report | `reports/Cleaning/Cleaning_Report.md`; separately maintained |
| Report images | `reports/figures/clean/`; exported separately |
| Usage and technical docs | `docs/Cleaning/`; separately maintained |

Reads default to Parquet with CSV fallback. Explicit CSV announces the selection and reads CSV directly, propagating its failure. Writes attempt Parquet first then CSV; failure of both attempts reports both causes. The successful saver path determines which format to read for a roundtrip; default reads do not select by modification time or metadata. Files of the other format may still exist.

For an isolated output directory:

```python
from src.cleaning.schema import CLEANED_DATASET_NAME, CLEANED_ORDER_ITEMS_SCHEMA
from src.pipeline.io.dataset_io import read_dataset

restored = read_dataset(
    saved_path.parent,
    CLEANED_ORDER_ITEMS_SCHEMA,
    dataset_name=CLEANED_DATASET_NAME,
    file_format=saved_path.suffix.lstrip("."),
)
```

This snippet assumes `saved_path` came from `save_cleaned_data()`. The processed loader always uses its configured directory. CSV encoding restores literal text, null masks, ZIP strings and declared timestamp/numeric dtypes. Index labels are not persisted.

The notebook's explicit save/reload cell can replace the clean dataset; it never writes report text, guide/reference content or report images.

## 9. Testing and maintenance

| Test module | Coverage |
|---|---|
| [`test_cleaning.py`](../../tests/cleaning/test_cleaning.py) | Input/text validation, whitespace, physical cells, chronology/deadlines, whole-order rules, overlapping reasons, exact retained values, empty results, typed storage/failures, notebook execution and no exports |
| [`test_cleaning_plots.py`](../../tests/cleaning/test_cleaning_plots.py) | Correct grains/denominators, missingness grouping, exclusion reconciliation, empty outputs, input/style preservation and six-figure rendering |
| [`tests/conftest.py`](../../tests/conftest.py) | Shared source frames, typed full-table fixture and plot-state cleanup |

`cleaning_full_frame` in `test_cleaning.py` and `visual_source` in `test_cleaning_plots.py` adapt the shared EDA fixture to valid source ID patterns. `eda_plot_cleanup` manages Matplotlib state. Parameterized tests exercise one-item failures, complete-order removal, missing endpoints, optional nulls and format fallback independently of production row totals.

Run from the repository root:

```powershell
python -m pytest tests/cleaning -q
```

To verify shared merge/EDA/cleaning behavior together:

```powershell
python -m pytest tests -q
```

When extending a rule, update its calculation, schema declarations if required, independent source-based validator, diagnostics, notebook explanation and focused tests. Preserve the separation between full-input preparation and retained-data summaries. Do not recompute the year reference after filtering, mutate original dates, remove partial orders or use formatted preview strings as analytical data.
