# IT135IU Data Science Project

Course project exploring order fulfillment and delivery performance in Brazilian e-commerce data. The repository currently provides a reproducible merge, full-table exploratory data analysis, and an item-level cleaning workflow. Analysis and final visualisation are the next project phases.

## Selected decision question

> Which delivery time slots cause the most late orders?

The intended decision-maker is the **Supply Chain Director** in the Supply Chain Department. The project will compare late-order outcomes across defined delivery time slots to inform operational scheduling decisions. The time-slot definition and late-order measurement will be specified and evaluated in the analysis phase. Because the question uses causal wording, conclusions should distinguish observed associations from evidence of causation.

## Dataset and download

The project uses the [Brazilian E-Commerce Public Dataset by Olist on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). It contains Brazilian marketplace order data from 2016–2018 in separate CSV tables. The Kaggle dataset page lists the license as **CC BY-NC-SA 4.0**.

To prepare the merge input, open the Kaggle dataset page, download and extract the archive, then copy these five source files into `data/raw/` while preserving their filenames:

- `olist_order_items_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`

The repository does not include the raw files. The merge notebook combines these tables and saves the item-level working dataset in `data/interim/`.

## Project status

| Phase | Status | Current output |
|---|---|---|
| Data merge | Complete | `merged_order_items`: 112,650 order items, 25 columns and 98,666 orders |
| Exploratory data analysis | Complete | Pre-cleaning profiles, diagnostic tables and figures |
| Data cleaning | Complete | `cleaned_order_items`: 108,571 items across 95,075 delivered orders |
| Data analysis | Not started | Notebook placeholder; analysis content has not yet been implemented |
| Final visualisation | Not started | Notebook placeholder; final project visualisations have not yet been implemented |
| Findings and recommendations | Pending | To be developed from the completed analysis |

A possible later extension is to explore whether historical order information can support late-delivery risk prediction. The target definition, features, leakage checks and evaluation have not yet been implemented.

## Workflow

```text
Raw source tables
    → Merge and validate
    → Explore the full merged data
    → Clean eligible orders
    → Analysis
    → Final visualisation
    → Findings and recommendations
```

The merge and EDA stages inspect the full item-level dataset. Cleaning evaluates data quality at the appropriate order, item and product grains, then retains or excludes each complete order. The processed table remains at item grain; quality flags used during inspection are not part of its final schema.

## Repository structure

```text
.
├── data/
│   ├── raw/             # Downloaded source files; excluded from version control
│   ├── interim/         # Reproducible merged datasets
│   └── processed/      # Cleaned datasets
├── docs/
│   ├── Merge/           # Merge user guide and technical reference
│   ├── EDA/             # EDA user guide and technical reference
│   └── Cleaning/        # Cleaning user guide and technical reference
├── notebooks/
│   ├── 00_data_merging.ipynb
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_visualisation.ipynb
├── reports/
│   ├── EDA/             # Exploratory data analysis report
│   ├── Cleaning/        # Cleaning report
│   └── figures/         # Report figures, grouped by phase
├── src/
│   ├── pipeline/        # Shared schemas, merge logic and dataset I/O
│   ├── eda/             # Reusable profiling, checks and plots
│   └── cleaning/        # Cleaning rules, validation, presentation and plots
├── tests/               # Merge, EDA and cleaning tests
└── requirements.txt
```

## Getting started

1. Clone the repository and install its pinned Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. [Download the Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), extract it, and place the five source CSV files listed above in `data/raw/`. Raw and generated data files are excluded from version control.
3. Start Jupyter from the repository root:

   ```bash
   jupyter lab
   ```

4. Run notebooks `00`, `01` and `02` in order. The merge notebook writes its result to `data/interim/`; EDA reads that full merge; Cleaning writes the validated output to `data/processed/`.

For the exact source-table combinations, loading options, cleaning rules and implementation details, use the phase documentation linked below.

## Documentation and reports

| Phase | Run and usage instructions | Implementation details | Results |
|---|---|---|---|
| Merge | [User Guide](docs/Merge/User_Guide.md) | [Technical Reference](docs/Merge/Technical_Reference.md) | Merge diagnostics are displayed in the notebook |
| EDA | [User Guide](docs/EDA/User_Guide.md) | [Technical Reference](docs/EDA/Technical_Reference.md) | [EDA Report](reports/EDA/EDA_Report.md) |
| Cleaning | [User Guide](docs/Cleaning/User_Guide.md) | [Technical Reference](docs/Cleaning/Technical_Reference.md) | [Cleaning Report](reports/Cleaning/Cleaning_Report.md) |

The guides explain how to run each phase. The technical references describe reusable modules and their interfaces. Reports record observed results. Notebooks do not generate or update report documents.

## Tests

Run the project test suite from the repository root:

```bash
python -m pytest tests -q
```

Tests use synthetic data to check contracts, transformations, storage and workflow behavior. Project source datasets are not required for the unit-test fixtures.

## Data and reproducibility

`data/raw/`, `data/interim/` and `data/processed/` are local working directories. Source data is obtained separately; interim and processed datasets are regenerated by the notebooks. Paths are resolved from the repository, and saved datasets use schema-aware Parquet with CSV fallback.

Use the corresponding phase guide when running one stage independently or selecting an explicit file format. Reference dimensions in reports describe the dataset version used for those results; a changed source dataset may produce different counts.
