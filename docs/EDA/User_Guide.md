# EDA User Guide

This guide covers running pre-cleaning inspections on the full merged dataset and reading their tables and figures. It assumes the repository and project environment are already available. For function contracts and dependencies, see the [EDA Technical Reference](Technical_Reference.md). Observed results are documented separately in the [EDA Report](../../reports/EDA/EDA_Report.md).

## 1. Dataset at a glance

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

## Contents

1. [Dataset at a glance](#1-dataset-at-a-glance)
2. [Run the complete notebook](#2-run-the-complete-notebook)
3. [Shared initialization](#3-shared-initialization)
4. [Inspect selected aspects](#4-inspect-selected-aspects)
5. [Read the outputs](#5-read-the-outputs)
6. [Troubleshooting and scope](#6-troubleshooting-and-scope)

## 2. Run the complete notebook

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

## 3. Shared initialization

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

## 4. Inspect selected aspects

The following blocks share initialization from section 3. Retain their returned objects for detailed investigation. The workflow helpers display tables and figures; lower-level profiling helpers return data without displaying it.

### 4.1. Structure and repeated entities

```python
display(profile_dataset(df))
display(column_profile)
display(profile_keys(df))
entity_consistency = profile_entity_consistency(df)
display(entity_consistency["summary"])
display(entity_consistency["conflicts"])
```

Repeated order, product, customer and seller IDs are expected at item level. `build_entity_views()` refuses conflicting source attributes or invalid keys instead of selecting an arbitrary record.

### 4.2. Numeric, category and time distributions

```python
numeric_profiles = eda.inspect_numeric_distributions(entity_views, column_roles)
category_distributions = eda.inspect_category_distributions(entity_views, column_roles)
temporal_distributions = eda.inspect_temporal_distributions(entity_views, column_roles)
```

Price and freight are profiled per item; physical measurements per product. Order lifecycle timestamps use orders, while shipping deadlines use items. Quantile windows and IQR whiskers affect inspection only.

### 4.3. Missingness and its context

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

### 4.4. Domain checks

Run sections 4.2 and 4.3 first because these checks reuse their results.

```python
text_format_profiles = eda.inspect_text_formats(
    column_roles, missing_patterns["missingness_frames"],
)
order_timeline = eda.inspect_order_timeline(entity_views)
measurement_checks = eda.inspect_measurement_plausibility(numeric_profiles, df)
order_structure = eda.inspect_order_structure(df)
```

For the detailed investigations, run notebook section 6. Its dependencies are listed in the [workflow function reference](Technical_Reference.md#6-notebook-facing-workflow). It retains original records behind timestamp reversals, delivered-order gaps, physical issues, numeric tails, future deadlines, city whitespace and multi-seller/deadline/state patterns.

### 4.5. Verify input preservation

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

## 5. Read the outputs

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

## 6. Troubleshooting and scope

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
