# EDA User Guide

This guide covers running pre-cleaning inspections on the full merged dataset and reading their tables and figures. It assumes the repository and project environment are already available. The [EDA notebook](../../notebooks/01_exploratory_data_analysis.ipynb) contains the complete workflow. For function contracts and dependencies, see the [EDA Technical Reference](Technical_Reference.md). Observed results are documented separately in the [EDA Report](../../reports/EDA_Report.md).

## Contents

1. [Inputs and outputs](#1-inputs-and-outputs)
2. [Run the notebook](#2-run-the-notebook)
3. [Shared setup](#3-shared-setup)
4. [Workflow by task](#4-workflow-by-task)
5. [Read and verify outputs](#5-read-and-verify-outputs)
6. [Troubleshooting](#6-troubleshooting)

## 1. Inputs and outputs

| Property | Reference input |
|---|---|
| Dataset | `merged_order_items` in `data/interim` |
| Shape | **112,650 items × 25 columns across 98,666 orders** |
| Item key | `(order_id, order_item_id)` |
| Observed entities | 98,666 customer records; 32,951 products; 3,095 sellers |
| Notebook | [`01_exploratory_data_analysis.ipynb`](../../notebooks/01_exploratory_data_analysis.ipynb) |
| Result | In-memory profiles, diagnostic views, tables and figures |
| File writes | None in the EDA workflow or notebook |

Reference counts describe the current merged input. They are not cleaning targets. All statuses, missing values and original values remain available during EDA. Customer records identified by `customer_id` do not count unique people across purchases.

## 2. Run the notebook

Open the EDA notebook with the project Python kernel, restart the kernel and run all cells from top to bottom. Its seven sections are:

| Section | Purpose |
|---|---|
| 1. Input and execution context | Load the full merge and record a preservation baseline |
| 2. Structure and completeness | Inspect dimensions, columns, keys, repeated attributes and source-grain views |
| 3. Distributions | Numeric, categorical, ZIP and calendar distributions |
| 4. Missingness in context | Exact patterns, directional pairs and grouped missingness |
| 5. Domain consistency | Text formats, timestamp sequence, measurements and order composition |
| 6. Detailed record investigations | Examine the records, precision and overlap behind detected patterns |
| 7. Input preservation | Verify that the loaded input has not changed |

Later sections consume earlier results. After replacing `df` or changing the selected source, rerun the notebook from the loading cell. The final preservation assertions should pass. Tables and plots appear in notebook output; the report is maintained separately.

## 3. Shared setup

Run this block once per Python or notebook session from inside the repository. It loads existing merged data and prepares the basic profiles used in subsequent examples.

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

import pandas as pd
from IPython.display import display
from src.data_loader import load_data
from src.eda.checks import build_entity_views, profile_entity_consistency, profile_keys
from src.eda.profiling import get_column_roles, profile_columns, profile_dataset
from src.eda import workflow as eda

# Default: Parquet first, with CSV fallback.
df = load_data("merged_order_items", stage="interim")
input_baseline = {
    "shape": df.shape,
    "columns": df.columns.copy(),
    "dtypes": df.dtypes.copy(),
    "row_hashes": pd.util.hash_pandas_object(df, index=True).copy(),
}
column_roles = get_column_roles()
column_profile = profile_columns(df)
entity_views = build_entity_views(df)
```

To choose CSV directly, replace the loading line with:

```python
df = load_data("merged_order_items", stage="interim", file_format="csv")
```

Then rebuild the baseline, roles, column profile and views from that same `df`. Use the schema-aware reader to preserve ZIP prefixes, timestamps, nulls and literal text values. The loader does not rerun merging. If no saved full dataset exists, follow the [Merge User Guide](../Merge/User_Guide.md).

## 4. Workflow by task

The following blocks share initialization from section 3. Retain their returned objects for detailed investigation. The workflow helpers display tables and figures; lower-level profiling helpers return data without displaying it.

<a id="41-structure-and-repeated-entities"></a>
### Structure and repeated entities

```python
display(profile_dataset(df))
display(column_profile)
display(profile_keys(df))
entity_consistency = profile_entity_consistency(df)
display(entity_consistency["summary"])
display(entity_consistency["conflicts"])
```

Repeated order, product, customer and seller IDs are expected at item level. `build_entity_views()` refuses conflicting source attributes or invalid keys instead of selecting an arbitrary record.

<a id="42-numeric-category-and-time-distributions"></a>
### Numeric, category and time distributions

```python
numeric_profiles = eda.inspect_numeric_distributions(entity_views, column_roles)
category_distributions = eda.inspect_category_distributions(entity_views, column_roles)
temporal_distributions = eda.inspect_temporal_distributions(entity_views, column_roles)
```

Price and freight are profiled per item; physical measurements per product. Order lifecycle timestamps use orders, while shipping deadlines use items. Quantile windows and IQR whiskers affect inspection only.

<a id="43-missingness-and-its-context"></a>
### Missingness and its context

```python
missing_patterns = eda.inspect_missing_patterns(df, entity_views, column_profile)
missing_pair_profiles = eda.inspect_missing_pairs(
    missing_patterns["missingness_frames"],
    missing_patterns["native_column_profiles"],
)
missing_context = eda.inspect_conditional_missingness(column_roles, df, entity_views)
status_missing_rates = eda.inspect_status_missingness(
    missing_context["condition_profiles"],
    missing_context["order_timestamp_columns"],
)
```

These objects retain the full tables even where a notebook preview is bounded. A directional missingness heatmap uses the rows missing the conditioning column as its denominator. The status heatmap instead uses all orders within each status.

<a id="44-domain-checks"></a>
### Domain checks

Run the distribution and missingness sections first because these checks reuse their results.

```python
text_format_profiles = eda.inspect_text_formats(
    column_roles, missing_patterns["missingness_frames"],
)
order_timeline = eda.inspect_order_timeline(entity_views)
measurement_checks = eda.inspect_measurement_plausibility(numeric_profiles, df)
order_structure = eda.inspect_order_structure(df)
```

<a id="45-investigate-records-behind-the-summaries"></a>
### Investigate records behind the summaries

After the distribution, missingness and domain-check sections, use these calls to reproduce notebook section 6. Each result retains full supporting tables when the display is bounded.

```python
timeline_details = eda.inspect_timeline_details(
    entity_views, order_timeline["timeline_profile"], missing_context["order_context"]
)
delivered_completeness = eda.inspect_delivered_completeness(
    entity_views["orders"], missing_context["core_timestamp_columns"],
    timeline_details["any_ordering_flag"], missing_context["order_context"]
)
product_completeness = eda.inspect_product_completeness(
    entity_views, missing_context["product_measurement_columns"], df, entity_views["orders"]
)
numeric_tails = eda.inspect_numeric_tails(numeric_profiles, df, entity_views)
deadline_calendar = eda.inspect_deadline_calendar(
    entity_views["orders"], df, missing_context["order_timestamp_columns"]
)
seller_city_spacing = eda.inspect_seller_city_spacing(entity_views, df, entity_views["orders"])
structure_overlap = eda.inspect_structure_overlap(
    order_structure["order_composition"], entity_views["orders"], df
)
```

| Finding | Records and tables |
|---|---|
| Timestamp reversals | `timeline_details["orders"]` and aligned `timeline_details["flags"]`; `precision_examples` holds bounded examples |
| Delivered orders missing core timestamps | `delivered_completeness["delivered_missing_records"]` |
| Missing or nonpositive measurements | `product_completeness["physical_product_records"]` and `physical_issue_summary` include item and order exposure |
| Zero freight | `product_completeness["zero_freight"]`, aligned to `df`, and `zero_freight_by_status` |
| Numeric tails | `numeric_tails["iqr_group_profiles"]` and bounded `numeric_tail_records` |
| Deadlines beyond the latest purchase year | `deadline_calendar["far_deadline_records"]` |
| Seller-city repeated spaces | `seller_city_spacing["city_space_records"]` and `city_form_counts` |
| Multiple sellers, deadlines or state pairs | `structure_overlap["composition_context"]` and `composition_flags`; `composition_item_examples` is bounded |

```python
print(timeline_details["flags"].columns.tolist())
display(product_completeness["physical_issue_summary"])
display(product_completeness["physical_product_records"])
display(delivered_completeness["delivered_missing_records"])
```

An empty table means the corresponding diagnostic found no matching records. Function details and dependencies are in the [workflow reference](Technical_Reference.md#6-notebook-facing-workflow).

<a id="46-verify-input-preservation"></a>
### Verify input preservation

After the selected inspections, compare the source with the baseline captured in section 3:

```python
assert df.shape == input_baseline["shape"]
pd.testing.assert_index_equal(df.columns, input_baseline["columns"])
pd.testing.assert_series_equal(df.dtypes, input_baseline["dtypes"])
pd.testing.assert_series_equal(
    pd.util.hash_pandas_object(df, index=True), input_baseline["row_hashes"],
)
print("EDA input preserved.")
```

## 5. Read and verify outputs

| Output | Where to inspect | Interpretation |
|---|---|---|
| Dataset and columns | `profile_dataset(df)`, `column_profile` | Item-row denominators, unless stated otherwise |
| Source-grain views | `entity_views` | One row per observed source key after consistency checks |
| Numeric profiles | `numeric_profiles[source]` | Finite-value statistics; missingness still counts all rows |
| Category frequencies | `category_distributions["frequency_tables"]` | Complete tables keyed by `(source, column)` |
| Calendar frequencies | `temporal_distributions["temporal_frequency_tables"]` | Complete tables keyed by `(source, column, frequency)` |
| Missingness patterns | `missing_patterns["missingness_profiles"][view]` | `row_distribution` and exact `patterns` |
| Paired missingness | `missing_pair_profiles[view]` | `pairs` plus directional `conditional_pct` matrix |
| Conditional missingness | `missing_context["condition_profiles"]` | Full grouped tables; unknown evaluations remain separate |
| Timestamp ordering | `order_timeline["timeline_profile"]` | `summary`, nullable `conditions`, bounded `examples` |
| Order composition | `order_structure["order_composition"]` | Order counts, order records and order-seller deadline ranges |

Special frequency groups distinguish Missing, Blank and Other from literal source strings with the same spelling. Top-value displays preserve omitted counts in Other. Undefined percentages are NaN, not zero. Multiple diagnostic conditions may affect the same entity.

Entity views contain only records observed in the full item table. They do not recover orders without items or raw-source entities absent from the merge.

## 6. Troubleshooting

| Symptom | Action |
|---|---|
| Saved input cannot be loaded | Check `data/interim`, the dataset name and actual format; read the loader's Parquet/CSV error details |
| Missing columns or incompatible types | Reload the registered full merge; a custom subset is not the complete notebook input |
| Entity views fail | Inspect key and attribute-conflict tables; resolve the source issue before treating repeated rows as one entity |
| Missing earlier variable | Run initialization and prerequisite sections in order, or restart and run the notebook |
| Reference dimensions differ | Check the input version before comparing with the published report |
| A displayed table is shortened | Inspect its returned full table; preview limits do not filter the dataset |
| Preservation check fails | Reload the source and locate the caller-side change before interpreting results |

EDA records observations without applying eligibility filters, imputing values, creating delivery labels or saving a modified dataset. Review decisions belong in separately maintained documentation. The notebook does not create or update reports, user guides or technical references.
