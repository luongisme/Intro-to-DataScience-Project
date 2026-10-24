# Merge User Guide

This guide covers selecting source tables, merging or reloading datasets, and interpreting the results before cleaning, preprocessing, and analysis. It assumes the repository and project environment are already available. For implementation details, see the [Merge Technical Reference](Technical_Reference.md).

<a id="full"></a>
## 1. Full dataset at a glance

| Property | Reference result |
|---|---|
| Resolved tables | `order_items → orders → customers → products → sellers` |
| Grain | One order item per row |
| Shape | **112,650 rows × 25 columns across 98,666 orders** |
| Unique key | The **composite key** `(order_id, order_item_id)` |
| Dataset name | `merged_order_items` |
| Storage | `../../data/interim`; Parquet first, CSV fallback if needed |

Key considerations:

- An order can contain multiple items, so repeated `order_id` values are expected. Do not remove rows solely because `order_id` repeats.
- The merged dataset retains order statuses and missing attributes. Cleaning rules and delivery-time features have not been applied.
- The raw orders table contains 99,441 orders, while order_items contains 98,666 distinct orders. The full item-level dataset does not create placeholder rows for the 775 orders without items. The `orders + customers` combination retains all 99,441 orders.

**Navigation**

| Task | Reference |
|---|---|
| Initialize a Python or notebook session | [Shared initialization](#start) |
| Merge the full dataset, inspect, save, or reload | [Shared operations](#operations) |
| Select a pair, triple, or four-table combination | [26-combination lookup](#lookup) |
| Review source data, schema, and downstream considerations | [Data notes](#data-notes) |
| Select columns, rename fields, or configure join keys | [Custom joins](#custom) |
| Run a complete example with imports and error handling | [Standalone full workflow](#complete) |
| Diagnose a failure | [Troubleshooting](#errors) |

**Standard workflow:** run section 2 once, select a combination in section 4, then inspect or save it using section 3. Reloading an existing file only requires section 2 and section 3.3. Repeat initialization after restarting the kernel or Python session.

<a id="start"></a>
## 2. Shared initialization — once per session

This block locates the project root, imports shared functions, and sets the full dataset as the default selection. It does not merge or save data. Run it inside the repository or a subdirectory such as `../../notebooks`. Merging requires the source CSV files in `../../data/raw`; reloading interim data only requires the saved dataset.

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
from src.paths import INTERIM_DIR
from src.data_loader import load_data
from src.pipeline.api import merge_data
from src.pipeline.io.dataset_io import save_dataset
from src.pipeline.merging.reporting import build_merge_report_tables

# Default selection; a combination below replaces this name.
dataset_name = "merged_order_items"
```

Subsequent snippets use these imports. Each combination assigns its own `dataset_name` and `result`, so it does not depend on running another combination first. Imports required only by custom joins remain with those examples.

<a id="operations"></a>
## 3. Shared operations

### 3.1. Merge and inspect — full dataset example

After initialization, create the full result in memory:

```python
dataset_name = "merged_order_items"
result = merge_data(tables=["order_items", "orders", "customers", "products", "sellers"])
tables = build_merge_report_tables(result)

with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    for section, table in tables.items():
        print(section.replace("_", " ").title())
        display(table)
```

For another combination, use its snippet in section 4. To inspect all eight report tables afterward, reuse the `with ...` block above; each combination snippet already creates `tables`.

| Object or table | Purpose |
|---|---|
| `result.data` | Output DataFrame for subsequent processing |
| `result.schema` | Column names/order, dtypes, unique key, and non-null constraints |
| `result.report` | In-memory join details and recorded checks |
| `tables["overview"]` | Requested/resolved tables, bridges, grain, shape, and key |
| `tables["sources"]` | Shapes of the sources used |
| `tables["joins"]` | Join order, keys, type, and cardinality |
| `tables["row_counts"]` | Input/output counts and expected counts where applicable |
| `tables["key_checks"]` | Duplicate/invalid keys and matched/unmatched left rows |
| `tables["column_selection"]` | Right-side column selection and renames |
| `tables["validation"]` | Recorded validation statuses |
| `tables["schema"]` | Expected/actual dtypes, key components, and required columns |

### 3.2. Save the current result

Use the `dataset_name` and `result` from the same combination. Do not pair a name from one selection with another selection's data.

```python
saved_path = save_dataset(
    result.data,
    INTERIM_DIR,
    result.schema,
    dataset_name=dataset_name,
)
print(saved_path)
```

The saver attempts Parquet first, reports any failure, then tries CSV. `saved_path` identifies the actual saved file. Dataset names exclude the extension. Merge output belongs in `../../data/interim`; `../../data/processed` is reserved for cleaned data.

To reload the exact format just saved and compare it precisely:

```python
reloaded = load_data(
    dataset_name,
    stage="interim",
    schema=result.schema,
    file_format=saved_path.suffix[1:],
)
pd.testing.assert_frame_equal(result.data, reloaded, check_exact=True)
```

### 3.3. Read an existing dataset

Initialization selects the full dataset by default. This block preserves any subsequent `dataset_name` selection. To choose another file, assign its name before reading, for example `dataset_name = "merged_orders__customers"`.

```python
df = load_data(dataset_name, stage="interim")
print(df.shape)
display(df.head(3))
```

Omitting `file_format` means Parquet first with CSV fallback. Explicit `file_format="parquet"` behaves identically. To read CSV directly:

```python
df = load_data(dataset_name, stage="interim", file_format="csv")
print(df.shape)
display(df.head(3))
```

Use the dataset name listed for the required combination; its merge snippet does not need to run first. All 26 standard names have registered schemas. Custom names require an explicit schema, as shown in section 6.1.

**Format considerations:**

- When both formats exist, the default still tries Parquet first, regardless of modification time. Select CSV explicitly if the latest save returned CSV while an older Parquet file remains.
- Read encoded interim CSV through `load_data()` or `read_dataset()` with its schema to preserve ZIP codes, timestamps, numeric types, missing values, and literals such as `NA`/`NULL`. A conventional CSV reader with inferred types does not implement this workflow's encoding.
- Interim reading returns a DataFrame. Per-step join reports are not persisted with it; rerun `merge_data()` from the sources to obtain them.
- Row, column, and order counts in this guide refer to the reference raw dataset. If the inputs change, investigate the difference and review the reference counts rather than altering data to force a match.

<a id="lookup"></a>
## 4. Select a combination

Click a combination to inspect its resolved sources, grain, key, schema, dataset name, and snippet. Initialize using [section 2](#start) before running any combination snippet.

| Combination | Bridge tables | Reference rows | Reference columns |
|---|---|---:|---:|
| [Full: I + O + C + P + S](#full) | None | 112,650 | 25 |
| [order_items + orders](#pair-i-o) | None | 112,650 | 14 |
| [order_items + customers](#pair-i-c) | orders | 112,650 | 17 |
| [order_items + products](#pair-i-p) | None | 112,650 | 12 |
| [order_items + sellers](#pair-i-s) | None | 112,650 | 10 |
| [orders + customers](#pair-o-c) | None | 99,441 | 11 |
| [orders + products](#pair-o-p) | order_items | 112,650 | 19 |
| [orders + sellers](#pair-o-s) | order_items | 112,650 | 17 |
| [customers + products](#pair-c-p) | order_items, orders | 112,650 | 22 |
| [customers + sellers](#pair-c-s) | order_items, orders | 112,650 | 20 |
| [products + sellers](#pair-p-s) | order_items | 112,650 | 15 |
| [order_items + orders + customers](#triple-i-o-c) | None | 112,650 | 17 |
| [order_items + orders + products](#triple-i-o-p) | None | 112,650 | 19 |
| [order_items + orders + sellers](#triple-i-o-s) | None | 112,650 | 17 |
| [order_items + customers + products](#triple-i-c-p) | orders | 112,650 | 22 |
| [order_items + customers + sellers](#triple-i-c-s) | orders | 112,650 | 20 |
| [order_items + products + sellers](#triple-i-p-s) | None | 112,650 | 15 |
| [orders + customers + products](#triple-o-c-p) | order_items | 112,650 | 22 |
| [orders + customers + sellers](#triple-o-c-s) | order_items | 112,650 | 20 |
| [orders + products + sellers](#triple-o-p-s) | order_items | 112,650 | 22 |
| [customers + products + sellers](#triple-c-p-s) | order_items, orders | 112,650 | 25 |
| [order_items + orders + customers + products](#quad-i-o-c-p) | None | 112,650 | 22 |
| [order_items + orders + customers + sellers](#quad-i-o-c-s) | None | 112,650 | 20 |
| [order_items + orders + products + sellers](#quad-i-o-p-s) | None | 112,650 | 22 |
| [order_items + customers + products + sellers](#quad-i-c-p-s) | orders | 112,650 | 25 |
| [orders + customers + products + sellers](#quad-o-c-p-s) | order_items | 112,650 | 25 |

Standard combinations use strict `LEFT m:1` joins. Bridge tables and their columns remain in the result. Dataset names reflect the **requested tables**, rather than every automatically included source.

The I/O/C/P/S schema groups are defined in [section 5.4](#schema). Only `orders + customers` has order-level grain; all other combinations resolve to an order_items base and item-level grain. Each snippet displays overview and schema. Use section 3.1 for all eight tables, section 3.2 to save, or its dataset name with section 3.3 to reload without merging.

<a id="pairs"></a>
### 4.1. Two tables

<a id="pair-i-o"></a>
#### order_items + orders

No bridges; item base, key `(order_id, order_item_id)`. Join `orders(order_id)`. Reference: **112,650 rows, 14 columns, 98,666 orders**; schema **I → O**.

```python
dataset_name = "merged_order_items__orders"
result = merge_data(tables=["order_items", "orders"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-i-c"></a>
#### order_items + customers

Adds `orders`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id)`. Reference: **112,650 rows, 17 columns, 98,666 orders**; schema **I → O → C**.

```python
dataset_name = "merged_order_items__customers"
result = merge_data(tables=["order_items", "customers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-i-p"></a>
#### order_items + products

No bridges; item base, key `(order_id, order_item_id)`. Join `products(product_id)`. Reference: **112,650 rows, 12 columns, 98,666 orders**; schema **I → P**.

```python
dataset_name = "merged_order_items__products"
result = merge_data(tables=["order_items", "products"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-i-s"></a>
#### order_items + sellers

No bridges; item base, key `(order_id, order_item_id)`. Join `sellers(seller_id)`. Reference: **112,650 rows, 10 columns, 98,666 orders**; schema **I → S**.

```python
dataset_name = "merged_order_items__sellers"
result = merge_data(tables=["order_items", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-o-c"></a>
#### orders + customers

No bridges; orders base, key `order_id`. Join `customers(customer_id)`. Reference: **99,441 rows, 11 columns, 99,441 orders**; schema **order_id → O → C**.

```python
dataset_name = "merged_orders__customers"
result = merge_data(tables=["orders", "customers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-o-p"></a>
#### orders + products

Adds `order_items`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → products(product_id)`. Reference: **112,650 rows, 19 columns, 98,666 orders**; schema **I → O → P**.

```python
dataset_name = "merged_orders__products"
result = merge_data(tables=["orders", "products"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-o-s"></a>
#### orders + sellers

Adds `order_items`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → sellers(seller_id)`. Reference: **112,650 rows, 17 columns, 98,666 orders**; schema **I → O → S**.

```python
dataset_name = "merged_orders__sellers"
result = merge_data(tables=["orders", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-c-p"></a>
#### customers + products

Adds `order_items` and `orders`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → products(product_id)`. Reference: **112,650 rows, 22 columns, 98,666 orders**; schema **I → O → C → P**.

```python
dataset_name = "merged_customers__products"
result = merge_data(tables=["customers", "products"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-c-s"></a>
#### customers + sellers

Adds `order_items` and `orders`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → sellers(seller_id)`. Reference: **112,650 rows, 20 columns, 98,666 orders**; schema **I → O → C → S**.

```python
dataset_name = "merged_customers__sellers"
result = merge_data(tables=["customers", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="pair-p-s"></a>
#### products + sellers

Adds `order_items`; item base, key `(order_id, order_item_id)`. Joins `products(product_id) → sellers(seller_id)`. Reference: **112,650 rows, 15 columns, 98,666 orders**; schema **I → P → S**.

```python
dataset_name = "merged_products__sellers"
result = merge_data(tables=["products", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triples"></a>
### 4.2. Three tables

<a id="triple-i-o-c"></a>
#### order_items + orders + customers

No bridges; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id)`. Reference: **112,650 rows, 17 columns, 98,666 orders**; schema **I → O → C**.

```python
dataset_name = "merged_order_items__orders__customers"
result = merge_data(tables=["order_items", "orders", "customers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-i-o-p"></a>
#### order_items + orders + products

No bridges; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → products(product_id)`. Reference: **112,650 rows, 19 columns, 98,666 orders**; schema **I → O → P**.

```python
dataset_name = "merged_order_items__orders__products"
result = merge_data(tables=["order_items", "orders", "products"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-i-o-s"></a>
#### order_items + orders + sellers

No bridges; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → sellers(seller_id)`. Reference: **112,650 rows, 17 columns, 98,666 orders**; schema **I → O → S**.

```python
dataset_name = "merged_order_items__orders__sellers"
result = merge_data(tables=["order_items", "orders", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-i-c-p"></a>
#### order_items + customers + products

Adds `orders`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → products(product_id)`. Reference: **112,650 rows, 22 columns, 98,666 orders**; schema **I → O → C → P**.

```python
dataset_name = "merged_order_items__customers__products"
result = merge_data(tables=["order_items", "customers", "products"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-i-c-s"></a>
#### order_items + customers + sellers

Adds `orders`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → sellers(seller_id)`. Reference: **112,650 rows, 20 columns, 98,666 orders**; schema **I → O → C → S**.

```python
dataset_name = "merged_order_items__customers__sellers"
result = merge_data(tables=["order_items", "customers", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-i-p-s"></a>
#### order_items + products + sellers

No bridges; item base, key `(order_id, order_item_id)`. Joins `products(product_id) → sellers(seller_id)`. Reference: **112,650 rows, 15 columns, 98,666 orders**; schema **I → P → S**.

```python
dataset_name = "merged_order_items__products__sellers"
result = merge_data(tables=["order_items", "products", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-o-c-p"></a>
#### orders + customers + products

Adds `order_items`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → products(product_id)`. Reference: **112,650 rows, 22 columns, 98,666 orders**; schema **I → O → C → P**.

```python
dataset_name = "merged_orders__customers__products"
result = merge_data(tables=["orders", "customers", "products"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-o-c-s"></a>
#### orders + customers + sellers

Adds `order_items`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → sellers(seller_id)`. Reference: **112,650 rows, 20 columns, 98,666 orders**; schema **I → O → C → S**.

```python
dataset_name = "merged_orders__customers__sellers"
result = merge_data(tables=["orders", "customers", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-o-p-s"></a>
#### orders + products + sellers

Adds `order_items`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → products(product_id) → sellers(seller_id)`. Reference: **112,650 rows, 22 columns, 98,666 orders**; schema **I → O → P → S**.

```python
dataset_name = "merged_orders__products__sellers"
result = merge_data(tables=["orders", "products", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="triple-c-p-s"></a>
#### customers + products + sellers

Adds `order_items` and `orders`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → products(product_id) → sellers(seller_id)`. Reference: **112,650 rows, 25 columns, 98,666 orders**; schema **I → O → C → P → S**.

```python
dataset_name = "merged_customers__products__sellers"
result = merge_data(tables=["customers", "products", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="quadruples"></a>
### 4.3. Four tables

<a id="quad-i-o-c-p"></a>
#### order_items + orders + customers + products

No bridges; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → products(product_id)`. Reference: **112,650 rows, 22 columns, 98,666 orders**; schema **I → O → C → P**.

```python
dataset_name = "merged_order_items__orders__customers__products"
result = merge_data(tables=["order_items", "orders", "customers", "products"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="quad-i-o-c-s"></a>
#### order_items + orders + customers + sellers

No bridges; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → sellers(seller_id)`. Reference: **112,650 rows, 20 columns, 98,666 orders**; schema **I → O → C → S**.

```python
dataset_name = "merged_order_items__orders__customers__sellers"
result = merge_data(tables=["order_items", "orders", "customers", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="quad-i-o-p-s"></a>
#### order_items + orders + products + sellers

No bridges; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → products(product_id) → sellers(seller_id)`. Reference: **112,650 rows, 22 columns, 98,666 orders**; schema **I → O → P → S**.

```python
dataset_name = "merged_order_items__orders__products__sellers"
result = merge_data(tables=["order_items", "orders", "products", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="quad-i-c-p-s"></a>
#### order_items + customers + products + sellers

Adds `orders`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → products(product_id) → sellers(seller_id)`. Reference: **112,650 rows, 25 columns, 98,666 orders**; schema **I → O → C → P → S**.

```python
dataset_name = "merged_order_items__customers__products__sellers"
result = merge_data(tables=["order_items", "customers", "products", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="quad-o-c-p-s"></a>
#### orders + customers + products + sellers

Adds `order_items`; item base, key `(order_id, order_item_id)`. Joins `orders(order_id) → customers(customer_id) → products(product_id) → sellers(seller_id)`. Reference: **112,650 rows, 25 columns, 98,666 orders**; schema **I → O → C → P → S**.

```python
dataset_name = "merged_orders__customers__products__sellers"
result = merge_data(tables=["orders", "customers", "products", "sellers"])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

[↑ Lookup](#lookup)

<a id="data-notes"></a>
## 5. Data considerations before cleaning and preprocessing

### 5.1. Grain and missing values

- Row counts and order counts measure different things at item grain. Define how order-level attributes are aggregated by `order_id`; do not repeatedly sum or count the same order information across its items.
- An unmatched key is a relationship issue. A matched product can still lack weight or dimensions; missing attributes do not imply a failed join.
- Timestamps use `datetime64[us]` without a declared timezone. Merge does not convert timezones, calculate lead times, or handle holidays.
- Preserve string ZIP/ID dtypes when reloading. Converting ZIP codes to numbers discards leading zeros that cannot reliably be reconstructed.
- `customer_unique_id` is excluded from this schema. Do not assume `customer_id` identifies the same person across multiple orders.
- Available columns depend on the **resolved sources**, including bridges. For example, `order_items + customers` includes orders timestamps; `order_items + products` does not contain customer state or delivery timestamps.
- Rules for retaining delivered orders, excluding missing/negative time segments, identifying lateness, or classifying routes belong to the relevant downstream stage. Merge does not apply them.

### 5.2. Interpreting validation information

`Current rows`, `Current columns`, and actual dtypes describe the DataFrame currently inspected. `Recorded status` reflects checks at merge time. If the DataFrame is subsequently modified, revalidate it using `validate_table_contract()` where appropriate; the report does not certify the modified data automatically.

`Matched left rows` counts input left rows, not expanded output rows. `Row count checked = Not applicable` means that row-preservation rule does not apply, rather than a failed check. `Unique key component` identifies a component of one composite key, not an individual uniqueness constraint.

### 5.3. Selected sources

| API name | File under `../../data/raw` | Raw rows | Selected columns |
|---|---|---:|---:|
| `order_items` | `olist_order_items_dataset.csv` | 112,650 | 7 |
| `orders` | `olist_orders_dataset.csv` | 99,441 | 8 |
| `customers` | `olist_customers_dataset.csv` | 99,441 | 4 |
| `products` | `olist_products_dataset.csv` | 32,951 | 6 |
| `sellers` | `olist_sellers_dataset.csv` | 3,095 | 4 |

Column counts refer to the merge contract, not every column in the source CSV. Payments, reviews, geolocation, and category translation are outside the five-source standard registry.

<a id="schema"></a>
### 5.4. Canonical schema and column groups

Combination descriptions use the following schema groups. Concatenating groups in the stated order gives the **exact output names and column order**, without duplicated join keys.

| Position | Column | Canonical dtype | Group |
|---:|---|---|---|
| 1 | `order_id` | `string` | I |
| 2 | `order_item_id` | `int64` | I |
| 3 | `product_id` | `string` | I |
| 4 | `seller_id` | `string` | I |
| 5 | `shipping_limit_date` | `datetime64[us]` | I |
| 6 | `price` | `float64` | I |
| 7 | `freight_value` | `float64` | I |
| 8 | `customer_id` | `string` | O |
| 9 | `order_status` | `string` | O |
| 10 | `order_purchase_timestamp` | `datetime64[us]` | O |
| 11 | `order_approved_at` | `datetime64[us]` | O |
| 12 | `order_delivered_carrier_date` | `datetime64[us]` | O |
| 13 | `order_delivered_customer_date` | `datetime64[us]` | O |
| 14 | `order_estimated_delivery_date` | `datetime64[us]` | O |
| 15 | `customer_zip_code_prefix` | `string` | C |
| 16 | `customer_city` | `string` | C |
| 17 | `customer_state` | `string` | C |
| 18 | `product_category_name` | `string` | P |
| 19 | `product_weight_g` | `float64` | P |
| 20 | `product_length_cm` | `float64` | P |
| 21 | `product_height_cm` | `float64` | P |
| 22 | `product_width_cm` | `float64` | P |
| 23 | `seller_zip_code_prefix` | `string` | S |
| 24 | `seller_city` | `string` | S |
| 25 | `seller_state` | `string` | S |

- **I:** the first seven columns, in order.
- **O:** seven additional orders columns, from customer_id through order_estimated_delivery_date.
- **C:** three additional customer columns, ZIP through state.
- **P:** five additional product columns, category through width.
- **S:** three additional seller columns, ZIP through state.

For example, `I → O → P` has 19 columns. The order-level `orders + customers` schema is `order_id → O → C`, with 11 columns. Full is `I → O → C → P → S`.

Full and every item-based combination use **one composite key**, `(order_id, order_item_id)`; individual components need not be unique. The orders + customers key is order_id. Full requires non-missing order_id, order_item_id, product_id, seller_id, and customer_id. Other attributes may be missing. String ZIP codes preserve leading zeros; timestamps retain microsecond resolution.

<a id="custom"></a>
## 6. Custom joins

Use `tables=[...]` for standard combinations with automatic bridges, or `base=...` and `joins=[JoinSpec(...)]` for an explicit plan. Do not mix modes. Explicit plans preserve the specified order and do not add bridges.

| JoinSpec parameter | Meaning |
|---|---|
| `table` | Right-side table |
| `on` | One key or a list of same-name keys |
| `left_on`, `right_on` | Corresponding differently named keys, instead of on |
| `how` | left, inner, right, outer; default left |
| `validate` | 1:1, 1:m, m:1, m:m; default m:1 |
| `columns` | Selected right-side columns; join keys are retained |
| `rename` | Renames of selected non-key right columns |
| `require_match` | Default True: every left key must exist on the right |

Null/blank keys never match each other. Match policy is evaluated from the left before joining. Many-to-many must be explicitly declared and can expand rows. In explicit mode, `grain=None` means grain is undeclared; the unique key can be empty when uniqueness is not guaranteed.

### 6.1. Select product dimensions and rename weight

After shared initialization, import JoinSpec for this example. It uses an item base and strict LEFT m:1 on product_id. Reference: **112,650 rows, 11 columns, 98,666 orders**.

```python
from src.pipeline.contracts.schema import JoinSpec

dataset_name = "custom_items_dimensions"
join_spec = JoinSpec(
    table="products", on="product_id", how="left", validate="m:1",
    columns=["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
    rename={"product_weight_g": "weight_g"}, require_match=True,
)
result = merge_data(base="order_items", joins=[join_spec])
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["schema"])
```

Use the [shared save block](#operations): this example assigns `custom_items_dimensions` and retains its schema in result.

**Reload in another session:** run shared initialization, then reconstruct the schema from declarations below. This block does not require raw data or execute a merge; the example's output file must already exist.

```python
from src.pipeline.contracts.schema import ORDER_ITEMS_SPEC, PRODUCTS_SPEC, JoinSpec, build_source_schema
from src.pipeline.contracts.schema_derivation import derive_join_schema

join_spec = JoinSpec(
    table="products", on="product_id", how="left", validate="m:1",
    columns=["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
    rename={"product_weight_g": "weight_g"}, require_match=True,
)
custom_schema = derive_join_schema(build_source_schema(ORDER_ITEMS_SPEC), build_source_schema(PRODUCTS_SPEC), join_spec)
dataset_name = "custom_items_dimensions"
df = load_data(dataset_name, stage="interim", schema=custom_schema, file_format="parquet")
print(df.shape)
display(df.head(3))
```

Choose file_format="csv" for direct CSV reading. The schema must match the save configuration. A registered standard name cannot be read with a different schema.

### 6.2. Differently named composite keys with unmatched rows

This memory-mode example joins `(order_id, parcel_no)` to `(reference_id, sequence)`. Additional imports stay with the example. Supplied DataFrames must already satisfy their schemas and dtypes.

```python
from src.pipeline.contracts.schema import JoinSpec, TableSchema

dataset_name = "custom_shipments_tracking"
shipments = pd.DataFrame({
    "order_id": pd.Series(["O1", "O2"], dtype="string"),
    "parcel_no": pd.Series([1, 1], dtype="int64"),
})
tracking = pd.DataFrame({
    "reference_id": pd.Series(["O1"], dtype="string"),
    "sequence": pd.Series([1], dtype="int64"),
    "carrier": pd.Series(["Carrier A"], dtype="string"),
})
shipment_schema = TableSchema(
    columns=["order_id", "parcel_no"],
    dtypes={"order_id": "string", "parcel_no": "int64"},
    unique_key=["order_id", "parcel_no"], non_nullable_columns=["order_id", "parcel_no"],
)
tracking_schema = TableSchema(
    columns=["reference_id", "sequence", "carrier"],
    dtypes={"reference_id": "string", "sequence": "int64", "carrier": "string"},
    unique_key=["reference_id", "sequence"], non_nullable_columns=["reference_id", "sequence"],
)
result = merge_data(
    base="shipments",
    joins=[JoinSpec(
        table="tracking", left_on=["order_id", "parcel_no"], right_on=["reference_id", "sequence"],
        how="left", validate="m:1", columns=["carrier"], rename={"carrier": "carrier_name"},
        require_match=False,
    )],
    source_tables={"shipments": shipments, "tracking": tracking},
    source_schemas={"shipments": shipment_schema, "tracking": tracking_schema},
)
tables = build_merge_report_tables(result)
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
    display(tables["overview"], tables["key_checks"], tables["schema"], result.data)
```

Reference result: **2 rows, 5 columns**, with one matched and one unmatched left row. The result key remains `(order_id, parcel_no)`; right-side sequence becomes nullable Int64. Missing attributes are not filled. The assigned name custom_shipments_tracking allows the shared save block to be used directly; reloading requires the matching custom schema.

An empty `joins=[]` is valid: it returns a checked copy of the base and a report without inferring grain. Use section 3.1 to inspect all report sections.

<a id="complete"></a>
## 7. Standalone full workflow

This complete block includes imports and error handling and can run without section 2. The shared workflow avoids repeated imports in routine use; this independent example intentionally includes its own imports.

By default it merges and displays results in memory. Set SAVE_OUTPUT=True to save, reload the actual saved format, and compare exactly. Errors are printed and re-raised. Its reference counts apply to the current reference raw dataset.

<details>
<summary>Expand the standalone full workflow</summary>

```python
SAVE_OUTPUT = False
result = None
saved_path = None
reloaded = None

try:
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
    from src.paths import INTERIM_DIR
    from src.data_loader import load_data
    from src.pipeline.api import merge_data
    from src.pipeline.io.dataset_io import save_dataset
    from src.pipeline.merging.reporting import build_merge_report_tables

    dataset_name = "merged_order_items"
    result = merge_data(tables=["order_items", "orders", "customers", "products", "sellers"])
    actual = (*result.data.shape, result.data["order_id"].nunique())
    if actual != (112650, 25, 98666):
        raise ValueError(f"Reference counts differ: {actual}. Check the raw dataset version.")
    tables = build_merge_report_tables(result)
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None):
        for section, table in tables.items():
            print(section.replace("_", " ").title())
            display(table)

    if SAVE_OUTPUT:
        saved_path = save_dataset(result.data, INTERIM_DIR, result.schema, dataset_name=dataset_name)
        print(f"Output: {saved_path}")
        reloaded = load_data(dataset_name, stage="interim", file_format=saved_path.suffix[1:])
        pd.testing.assert_frame_equal(result.data, reloaded, check_exact=True)
        print("Merge, save and exact reload checks passed.")
    else:
        print("Merge completed in memory.")
except Exception as error:
    print(f"Merge workflow failed ({type(error).__name__}): {error}")
    raise
```

</details>

<a id="errors"></a>
## 8. Troubleshooting

| Symptom | Action |
|---|---|
| Root or src cannot be located | Run inside the repository or a subdirectory with the project interpreter |
| Missing raw file | Check the filename and data/raw; reloading saved interim does not require raw |
| Missing columns, incorrect dtypes, or malformed timestamps | Compare schema/configuration with the source; merge does not repair values |
| Duplicate/null/blank source key | Review the key's meaning and source data; do not remove duplicates solely to bypass validation |
| Orphan key error | Check relationships; use require_match=False in a custom plan only when unmatched rows should be retained |
| Cardinality violation | Review keys and validate; use m:m only for an intended many-to-many relationship |
| Column collision | Configure columns/rename for non-key columns |
| Parquet failure | Read the specific cause; CSV fallback is available. Follow engine installation guidance only for an engine failure; handle permissions or corruption as reported |
| Both formats fail | Review each format's error and check file/schema/path/read or write permissions |
| An older Parquet file was read | Select file_format="csv" to read the intended CSV output |
| Custom name absent from registry | Supply its matching schema; it can be reconstructed as in section 6.1 |
| Full baseline differs | Check the raw version and resolved sources rather than forcing rows to match |

These workflows process data and display results. Documentation is maintained separately and is not updated by the pipeline or notebook. Test commands and coverage are documented in the [technical reference](Technical_Reference.md#testing).

[↑ Full dataset](#full) · [↑ Shared operations](#operations) · [↑ Lookup](#lookup)
