# Olist Delivery Performance

**Delivery-delay analysis, visualization, and predictive modeling using Brazilian e-commerce data.**

This project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) to help a **Supply Chain Director** identify delivery periods and routes worth investigating.

**Research question:** Which delivery time slots have the most late orders?

A *time slot* is a six-hour interval based on **order approval time**, not a scheduled delivery window. The analysis reports observed associations, **not causal effects**.

## Key findings

An order is late when its actual delivery date is later than its estimated delivery date. Each eligible delivered order is counted once.

| Approval time | Delivered orders | Late orders | Late rate |
|---|---:|---:|---:|
| 00:00–06:00 | 19,910 | 1,455 | **7.31%** |
| 06:00–12:00 | 18,017 | 1,143 | 6.34% |
| 12:00–18:00 | 30,448 | **2,085** | 6.85% |
| 18:00–24:00 | 26,700 | 1,826 | 6.84% |
| **Total** | **95,075** | **6,509** | **6.85%** |

- **Most late cases:** 12:00–18:00 (2,085 orders).
- **Highest late rate:** 00:00–06:00 (7.31%).
- **Route for further investigation:** SP → RJ, particularly afternoon-approved orders purchased in November–December 2017 and February–March 2018.

See the [analysis notebook](notebooks/03_analysis.ipynb) and [report](reports/Analysis_Report.md) for detailed evidence and limitations.

## Quick start

**1. Clone and install** (preferably in a dedicated Python environment):

```bash
git clone https://github.com/luongisme/Olist-Delivery-Performance.git
cd Olist-Delivery-Performance
python -m pip install -r requirements.txt
```

**2. Download the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)** and place these five files in `data/raw/`:

```text
olist_order_items_dataset.csv
olist_orders_dataset.csv
olist_customers_dataset.csv
olist_products_dataset.csv
olist_sellers_dataset.csv
```

**3. Launch Jupyter** from the repository root:

```bash
jupyter lab
```

Run notebook cells from top to bottom in a fresh kernel. Raw and generated datasets are not committed to Git.

## Notebook workflow

| Notebook | Purpose | Main output |
|---|---|---|
| [00 · Merging](notebooks/00_data_merging.ipynb) | Merge and validate five source tables | `data/interim/` |
| [01 · EDA](notebooks/01_exploratory_data_analysis.ipynb) | Profile merged data | In-memory diagnostics |
| [02 · Cleaning](notebooks/02_data_cleaning.ipynb) | Prepare eligible delivered orders | `data/processed/` |
| [03 · Analysis](notebooks/03_analysis.ipynb) | Compare approval slots, routes, and delivery stages | Tables and figures in notebook |
| [04 · Visualisation](notebooks/04_visualisation.ipynb) | Create four revised analytical charts | `outputs/revised_charts/` |
| [05 · Modelling](notebooks/05_modelling.ipynb) | Evaluate late-delivery prediction models | Test metrics and diagnostic figures |

**Execution:** For the complete descriptive analysis, run **00 → 01 → 02 → 03**. Once `02` has produced the cleaned dataset, `04` and `05` can run independently of `03`.

Notebook `05` compares a dummy baseline, **Logistic Regression, Random Forest, and XGBoost** using chronological train/validation/test cohorts. It includes feature ablation, threshold evaluation, and a final held-out test. Its additional plots (feature importance, cumulative gains, and calibration) are written to `outputs/figures/Modelling/` when the relevant cells run. These outputs are not published model benchmarks.

## Reproducibility

- Dependencies are pinned in [`requirements.txt`](requirements.txt); the exact Python interpreter and platform are **not** locked.
- Notebook `05` uses `RANDOM_SEED = 42` from [`src/config.py`](src/config.py). Temporal splits use fixed dates rather than random sampling.
- The pipeline includes input checks, data validation, and automated tests:

```bash
python -m pytest -q
```

**Reference data checks:** 112,650 merged item rows → 108,571 cleaned item rows → 95,075 eligible delivered orders, including 6,509 late orders.

**Verification status:** Reference descriptive results are documented, but independent end-to-end reproduction of all six notebooks has **not** been confirmed. Notebook `04` assumes SP → RJ is the leading state pair for the reference data; notebook `05` also contains dataset-specific assertions. Review those checks when using different inputs.

## Documentation and data use

Detailed procedures are available in the [Merge](docs/Merge/User_Guide.md), [EDA](docs/EDA/User_Guide.md), [Cleaning](docs/Cleaning/User_Guide.md), and [Analysis](docs/Analysis/User_Guide.md) user guides. Implementation details for visualization and modeling are currently in notebooks `04` and `05`.

The Olist dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is used for non-commercial academic analysis. Results concern historical eligible delivered orders; they should not be interpreted as causal evidence or validated deployment performance.
