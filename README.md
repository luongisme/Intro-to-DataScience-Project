# Olist Delivery Performance

This project examines delivery performance in Brazilian e-commerce using Olist data. It helps a Supply Chain Director choose which orders to review when investigating late deliveries.

## Research question

> Which delivery time slots cause the most late orders?

Each order is counted once and assigned to a six-hour slot based on its approval time. An order is late when its actual delivery date is later than its estimated delivery date. The analysis describes observed associations; evaluating the effect of changing approval times requires additional operational evidence.

## Key findings

The table covers delivered orders retained after cleaning. Each rate uses the orders in its slot.

| Approval time | Delivered orders | Late orders | Late rate |
|---|---:|---:|---:|
| 00:00–06:00 | 19,910 | 1,455 | **7.31%** |
| 06:00–12:00 | 18,017 | 1,143 | 6.34% |
| 12:00–18:00 | 30,448 | **2,085** | 6.85% |
| 18:00–24:00 | 26,700 | 1,826 | 6.84% |

Afternoon approvals accounted for the most late deliveries, with 630 more cases than overnight approvals, making them the first priority when reviewing the largest number of late cases. For an initial operational review, the São Paulo → Rio de Janeiro route offers a specific starting point: orders purchased in November–December 2017 and February–March 2018 accounted for 282 of its 392 late afternoon deliveries.

[Explore the analysis notebook](notebooks/03_analysis.ipynb)

## Data and approach

The [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) is published by Olist under the CC BY-NC-SA 4.0 license. Five source tables are merged, profiled and cleaned before analysis at order level. The cleaned dataset contains 108,571 item rows belonging to 95,075 delivered orders.

The analysis compares approval slots using a Chi-square test, then examines routes, purchase months and delivery stages.

## Reproduce the analysis

From the repository root, install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and extract the [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), then place these five files in `data/raw/`:

- `olist_order_items_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`

Start Jupyter from the repository root:

```bash
jupyter lab
```

Run notebooks `00`, `01`, `02` and `03` in order. Notebook `00` saves the merged table to `data/interim/`; notebook `02` saves the cleaned table to `data/processed/`. Notebook `03` reads both tables to compare outcomes before and after cleaning. It analyzes all eligible orders without random sampling. Data files are created locally.

## Explore the project

| Phase | Output | Links |
|---|---|---|
| Merge | Combined item-level dataset and join checks | [Notebook](notebooks/00_data_merging.ipynb) · [User guide](docs/Merge/User_Guide.md) · [Technical reference](docs/Merge/Technical_Reference.md) · [Report](reports/Merge_Report.md) |
| EDA | Profiles and data-quality observations | [Notebook](notebooks/01_exploratory_data_analysis.ipynb) · [User guide](docs/EDA/User_Guide.md) · [Technical reference](docs/EDA/Technical_Reference.md) · [Report](reports/EDA_Report.md) |
| Cleaning | Cleaned item-level dataset of eligible delivered orders | [Notebook](notebooks/02_data_cleaning.ipynb) · [User guide](docs/Cleaning/User_Guide.md) · [Technical reference](docs/Cleaning/Technical_Reference.md) · [Report](reports/Cleaning_Report.md) |
| Analysis | Order-level delivery outcomes and comparisons | [Notebook](notebooks/03_analysis.ipynb) · [User guide](docs/Analysis/User_Guide.md) · [Technical reference](docs/Analysis/Technical_Reference.md) · [Report](reports/Analysis_Report.md) |

## Future work

A predictive extension is under consideration to estimate late-delivery risk for individual orders, with the prediction point and available features still being defined.
