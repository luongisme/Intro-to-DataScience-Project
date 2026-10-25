# Cleaning User Guide

This guide covers validating the full merged input, applying the cleaning workflow, inspecting its results and saving or reloading the final dataset. It assumes the repository and project environment are already available. For contracts and implementation details, see the [Cleaning Technical Reference](Technical_Reference.md). Observed results are documented in the [Cleaning Report](../../reports/Cleaning/Cleaning_Report.md).

## 1. Workflow at a glance

| Property | Reference result |
|---|---|
| Input | `merged_order_items` in `data/interim` |
| Input dimensions | 112,650 items × 25 columns; 98,666 orders |
| Final dataset | `cleaned_order_items` in `data/processed` |
| Final dimensions | **108,571 items × 25 columns; 95,075 orders** |
| Unique key | `(order_id, order_item_id)` |
| Exclusions | 3,591 complete orders containing 4,079 items |
| Notebook | [`02_data_cleaning.ipynb`](../../notebooks/02_data_cleaning.ipynb) |
| Storage | Parquet first, CSV fallback; read the actual saved format for verification |

These are reference observations for the current source data. The workflow enforces data contracts and eligibility rules rather than forcing fixed row counts.

## Contents

1. [Workflow at a glance](#1-workflow-at-a-glance)
2. [Run the complete notebook](#2-run-the-complete-notebook)
3. [Shared initialization](#3-shared-initialization)
4. [Clean and inspect](#4-clean-and-inspect)
5. [Save and reload](#5-save-and-reload)
6. [Use inspection without filtering](#6-use-inspection-without-filtering)
7. [Rules and troubleshooting](#7-rules-and-troubleshooting)

## 2. Run the complete notebook

Open the cleaning notebook with the project Python kernel, restart the kernel and run all cells from top to bottom.

| Section | Purpose |
|---|---|
| 1. Input and execution context | Read the full merge and capture the source baseline |
| 2. Input integrity | Inspect text domains, item keys and repeated entity attributes |
| 3. Preparation of the full input | Run cleaning; display city and physical-measurement changes before filtering |
| 4. Temporal quality of the full input | Inspect core chronology and item shipping deadlines |
| 5. Whole-order eligibility | Show independent reasons, sequential exclusions and removed order records |
| 6. Retained dataset | Inspect dimensions, missingness, six figures, output checks and a formatted preview |
| 7. Persistence and source preservation | Save to processed, reload the actual format and verify preservation |

The persistence section writes `cleaned_order_items` and may replace an existing file of the successful format. Earlier inspection functions do not save files. The notebook does not generate or update documentation or export report figures.

## 3. Shared initialization

Run once per session from inside the repository. The examples below reuse these imports.

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
import pandas as pd
from IPython.display import display
from src.data_loader import load_data
from src.cleaning import (
    CLEANED_DATASET_NAME,
    assert_cleaned_roundtrip,
    clean_order_items,
    format_cleaning_preview,
    inspect_order_items,
    save_cleaned_data,
)
from src.cleaning.plots import build_cleaning_visuals
from src.cleaning.validation import (
    assert_input_unchanged,
    capture_input_baseline,
    validate_cleaning_input,
)

df = load_data("merged_order_items", stage="interim")
input_baseline = capture_input_baseline(df)
```

The default reader tries Parquet first, then CSV. For direct CSV input, replace the loading line with:

```python
df = load_data("merged_order_items", stage="interim", file_format="csv")
```

Capture the baseline again whenever you replace `df`. If neither saved input exists, use the [Merge User Guide](../Merge/User_Guide.md) to produce the full merge first.

## 4. Clean and inspect

### 4.1. Run the validated workflow

```python
input_checks = validate_cleaning_input(df)
display(format_cleaning_preview(input_checks["key_profile"]))

cleaned_df, cleaning_details = clean_order_items(df)
display(format_cleaning_preview(cleaning_details["dataset_summary"]))
display(format_cleaning_preview(cleaning_details["output_checks"]))
assert_input_unchanged(df, input_baseline)
```

`clean_order_items()` validates the input internally as well. The explicit first call exposes input profiles for inspection. The source `df` remains unchanged. The returned `cleaned_df` retains canonical column names and calculation dtypes.

### 4.2. Review preparation and exclusions

```python
for name in ("text_changes", "measurements", "timeline", "deadlines",
             "filter_reasons", "filter_flow"):
    print(name)
    display(format_cleaning_preview(cleaning_details[name]))

# Full records remain in memory; the display copy is bounded.
display(format_cleaning_preview(cleaning_details["excluded_orders"], rows=10))
display(format_cleaning_preview(cleaned_df, rows=10))
```

`filter_reasons` counts every order affected by each reason, so its groups may overlap. `filter_flow` counts newly excluded orders in sequence and reconciles with total exclusions. Preparation tables describe the complete input, including orders later removed.

Formatted previews are display-only string tables. Continue calculations and saving with `cleaned_df`, not the formatted copy.

### 4.3. Display the six review figures

```python
cleaning_visuals = build_cleaning_visuals(df, cleaned_df, cleaning_details)
# Avoid automatic inline display before the explicit figure sequence.
for figure in cleaning_visuals["figures"].values():
    plt.close(figure)

for name in ("filtering", "missingness", "numeric", "calendar", "categories", "geography"):
    display(cleaning_visuals["figures"][name])
```

| Figure | Counting unit and interpretation |
|---|---|
| `filtering` | Orders retained/excluded; sequential reasons are disjoint |
| `missingness` | Item nulls; before and after rates use their own item denominators |
| `numeric` | Item price/freight and distinct-product weight; quantile windows affect display only |
| `calendar` | Distinct orders by purchase month and stored approval hour |
| `categories` | Distinct products; top ten, Other and Missing |
| `geography` | Distinct customer records and sellers, with separate denominators |

Full supporting tables are available in `cleaning_visuals["tables"]`. For example:

```python
display(cleaning_visuals["tables"]["numeric"])
display(cleaning_visuals["tables"]["categories"])
```

Physical fields are grouped in the compact missingness table only when their before/after missing counts match. Counts in a grouped row are per field. Optional missing values remain missing.

## 5. Save and reload

### 5.1. Save and verify the actual file

Run this block when the result is ready to persist:

```python
saved_path = save_cleaned_data(cleaned_df)
reloaded_df = load_data(
    CLEANED_DATASET_NAME,
    stage="processed",
    file_format=saved_path.suffix.lstrip("."),
)
assert_cleaned_roundtrip(cleaned_df, reloaded_df)
assert_input_unchanged(df, input_baseline)
print(f"Verified processed output: {saved_path}")
```

The saver validates the clean schema and attempts Parquet, then CSV if needed. It returns the path that was successfully written. Each format uses atomic replacement. If both attempts fail, the error includes their causes and dependency guidance where applicable.

For an isolated check, pass a different directory to `save_cleaned_data(cleaned_df, output_dir=...)`. To reload that directory, use the schema-aware `read_dataset()` interface described in the Technical Reference; `load_data(stage="processed")` always uses the configured processed directory.

### 5.2. Independent reads

```python
# Default: Parquet first, with CSV fallback.
cleaned_df = load_data("cleaned_order_items", stage="processed")
```

```python
# Explicit selection: read CSV directly.
cleaned_df = load_data("cleaned_order_items", stage="processed", file_format="csv")
```

If both files exist, the default still tries Parquet first regardless of modification time. When the latest save returned CSV while an older Parquet file remains, select CSV explicitly. Read encoded CSV through the schema-aware loader to preserve nulls, literal text, ZIP prefixes and timestamps.

## 6. Use inspection without filtering

To retain every input item and status for diagnosis:

```python
inspected_df, inspection_details = inspect_order_items(df)
display(format_cleaning_preview(inspection_details["inspection_checks"]))
```

This returns a 28-column prepared table with three quality flags. It is an intermediate inspection result, not `cleaned_order_items`. The clean saver rejects it because the final contract contains 25 original columns. Call `clean_order_items(df)` on the original full input for the complete validated workflow.

## 7. Rules and troubleshooting

The workflow normalizes city whitespace, converts observed nonpositive physical measurements to null and records quality flags before excluding complete orders. An order is excluded for any of these reasons, in sequence:

1. Status is not `delivered`, including missing status.
2. A required timestamp is missing: Purchase, Approved, Carrier Handoff, Customer Delivery or Estimated Delivery.
3. An adjacent core timestamp is reversed at full precision.
4. At least one item has an observed nonpositive physical measurement.
5. A shipping deadline is before purchase or its year exceeds the latest purchase year in the full input, under the selected strict review policy.

The future-year condition is an observation-based review signal, not proof of an invalid deadline. Its reference is computed before filtering. Missing optional product attributes remain allowed. Price, freight, zero freight and statistical tails are preserved. Quality flags are removed from the final 25-column dataset. Delivery labels, business-day durations, time slots and route features are outside this workflow.

| Symptom | Action |
|---|---|
| Input validation fails | Inspect reported columns, text issues, keys or entity conflicts before applying transformations |
| No orders remain | Review `filter_reasons` and `filter_flow`; an empty retained dataset is supported |
| Flags or values do not match validation | Use the original full input and the shared workflow; do not recalculate flags on a filtered subset |
| Saving rejects 28 columns | Save `cleaned_df`, not the intermediate inspection table |
| Default read returns an older result | Check the successful save extension and select that format explicitly |
| Roundtrip comparison fails | Verify the loaded format, schema and saved dataset; do not cast away the difference |
| Preview contains strings or readable labels | It is display-only; return to `cleaned_df` for computation |

Report text and report figures are maintained separately from notebook execution.
