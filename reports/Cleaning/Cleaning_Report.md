# Data Cleaning Report

## Dataset scope

This report documents the cleaning of the full `merged_order_items` dataset in `data/interim` and the resulting `cleaned_order_items` dataset in `data/processed`. Each row represents one order item, identified by `(order_id, order_item_id)`. Figures use the same data and plotting functions as `notebooks/02_data_cleaning.ipynb`.

Counts use commas as thousands separators and periods for decimals. Percentages use the denominator stated in each table or figure. Item, order, product, customer-record and seller counts refer to distinct units of observation. Diagnostic groups may overlap.

## Contents

1. [Dataset dimensions](#1-dataset-dimensions)
2. [Order filtering](#2-order-filtering)
3. [Value changes](#3-value-changes)
4. [Remaining missing values](#4-remaining-missing-values)
5. [Numeric distributions](#5-numeric-distributions)
6. [Temporal coverage and approval hours](#6-temporal-coverage-and-approval-hours)
7. [Product categories and geography](#7-product-categories-and-geography)

## 1. Dataset dimensions

| Unit of observation | Full input | After cleaning |
|---|---:|---:|
| Items — `(order_id, order_item_id)` | 112,650 | 108,571 |
| Orders — `order_id` | 98,666 | 95,075 |
| Products — `product_id` | 32,951 | 31,875 |
| Sellers — `seller_id` | 3,095 | 2,959 |
| Customer records — `customer_id` | 98,666 | 95,075 |
| Columns | 25 | 25 |

Cleaning excludes **3,591 orders containing 4,079 items**. Every retained order preserves all of its input items. The final dataset contains only `delivered` orders and retains price, freight and optional product attributes.

The intermediate inspection table preserves all input rows and adds three quality flags, bringing its column count to 28. The final dataset contains the 25 original columns. Preparation diagnostics describe the full input; distributions after filtering describe the retained records at the stated level of observation.

`customer_id` identifies a customer record linked to a purchase. It does not represent a unique person across purchases.

## 2. Order filtering

![Figure 1 — Order retention and sequential exclusions](../figures/clean/figure_01_filtering.png)

*Figure 1. Of 98,666 input orders, 95,075 are retained (96.36%). The lower panel counts newly excluded orders at each rule. Each excluded order contributes once to the total.*

| Step | Newly excluded orders | Newly excluded items | Remaining orders | Remaining items |
|---|---:|---:|---:|---:|
| Full input | 0 | 0 | 98,666 | 112,650 |
| Status other than delivered | 2,188 | 2,453 | 96,478 | 110,197 |
| Missing required timestamp | 23 | 24 | 96,455 | 110,173 |
| Core timestamp reversal | 1,373 | 1,592 | 95,082 | 108,581 |
| Invalid physical measurement | 6 | 8 | 95,076 | 108,573 |
| Shipping deadline flagged for review | 1 | 2 | 95,075 | 108,571 |

The five required timestamps are Purchase, Approved, Carrier Handoff, Customer Delivery and Estimated Delivery. Exclusions apply to the entire order, including cases where only one item triggers a rule.

When reasons are counted independently on the full input, **2,205 orders lack a required timestamp**, **1,382 have a core timestamp reversal** and **3 have a shipping deadline flagged for review**. These groups overlap with other exclusion conditions. Their newly excluded counts in the sequential table are therefore 23, 1,373 and 1, respectively. Independent diagnostic counts cannot be summed to obtain total exclusions.

In the full input, 1,359 orders have Carrier Handoff before Approved and 23 have Customer Delivery before Carrier Handoff. No order has Approved before Purchase. Four items across three orders have shipping deadlines in 2020, later than the latest observed purchase year, 2018. After earlier filtering steps, one of these orders, containing two items, remains for exclusion at the deadline step. Original timestamp values remain available in the diagnostic data.

The status rule selects completed deliveries. The deadline rule excludes orders flagged by the selected review policy for deadlines beyond the observed purchase years.

## 3. Value changes

### 3.1. City names

Before filtering, repeated internal whitespace is collapsed in **56 `seller_city` cells across 3 sellers**. Of these, **54 changed cells** remain in the final dataset. No `customer_city` values change.

| Source value | Final value | Retained items with a change | Corresponding orders |
|---|---|---:|---:|
| `sao  paulo` | `sao paulo` | 50 | 48 |
| `sao  jose dos pinhais` | `sao jose dos pinhais` | 2 | 2 |
| `ferraz de  vasconcelos` | `ferraz de vasconcelos` | 2 | 2 |

Comparing retained items with their source rows, only these 54 `seller_city` cells differ. **Values in the other 24 columns remain unchanged.** Date, currency and text formatting in notebook previews affects display only.

### 3.2. Physical measurements

Preparation replaces **8 nonpositive weight values** with nulls, affecting **4 products across 6 orders**. All six orders are subsequently excluded. Weight has 18 nulls in the source, 26 after preparation and **18 in the final dataset**.

Length, height and width have no nonpositive values requiring replacement; each retains 18 source nulls. The final dataset has no observed physical measurements `<= 0`. Missing measurements are not imputed.

### 3.3. Price and freight

Price and freight values remain unchanged for every retained item. The final dataset contains **370 items with zero freight**, compared with 383 in the full input. The difference of 13 items results from order exclusions. Zero freight and large observed values on retained items are preserved.

## 4. Remaining missing values

![Figure 2 — Missing values before and after cleaning](../figures/clean/figure_02_missingness.png)

*Figure 2. Percentages use separate denominators: 112,650 input items and 108,571 retained items. Labels show both the missing count and percentage. The four physical measurements share the same missing counts and are grouped into one row; 18 is the count for each column, not their combined total.*

Approved, Carrier Handoff and Customer Delivery have no nulls after filtering. Purchase, Estimated Delivery and Shipping Limit have no nulls in either the full input or the final dataset.

| Attribute | Input items with nulls | Retained items with nulls | Corresponding retained entities |
|---|---:|---:|---|
| Product category | 1,603 | 1,526 | 582 products |
| Each of weight, length, height and width | 18 | 18 | 2 products per column |

Category is missing for **1.4055% of retained items** and **1.8259% of retained products**. The denominators differ because one product can appear in multiple item rows.

Each physical measurement is missing for **0.0166% of retained items**. Its missing count remains 18, while its rate increases from 0.0160% because fewer items remain after filtering. These are retained source nulls, with no imputation. All other columns in the final dataset have no nulls.

## 5. Numeric distributions

![Figure 3 — Numeric distributions in the retained dataset](../figures/clean/figure_03_numeric.png)

*Figure 3. Price and freight use 108,571 items. Weight uses 31,875 products, of which 31,873 have finite values and 2 have missing values. The upper panels show the full range; the lower panels show the 1st–99th percentile window. Observations outside these windows remain in the dataset and the full-range panels.*

| Measurement | Unit of observation | Unit | Min | Median | Max |
|---|---|---|---:|---:|---:|
| Price | Item | BRL | 0.85 | 74.90 | 6,735 |
| Freight | Item | BRL | 0 | 16.25 | 409.68 |
| Weight | Product | g | 2 | 700 | 40,425 |
| Length | Product | cm | 7 | 25 | 105 |
| Height | Product | cm | 2 | 13 | 105 |
| Width | Product | cm | 6 | 20 | 118 |

Product statistics use one record per product, avoiding repeated counts based on the number of items sold. The display windows span **9.99–889 BRL** for price, **4.45–83.83 BRL** for freight and approximately **75–22,507 g** for weight. These windows affect the plots only; no observations are removed or clipped by percentile.

## 6. Temporal coverage and approval hours

![Figure 4 — Purchase months and approval hours after cleaning](../figures/clean/figure_04_calendar.png)

*Figure 4. Both panels use 95,075 distinct orders. The month axis includes months with zero orders. Approval hours are extracted from the stored `order_approved_at` timestamps, without timezone conversion or work-shift grouping.*

Purchase months span **September 2016 through August 2018**. **November 2016 has no orders**. **November 2017 has 7,287 orders**, the largest monthly count in the retained dataset. The most frequent approval hour is **02:00–02:59, with 6,855 orders**.

The panels describe order counts by recorded purchase month and approval hour.

## 7. Product categories and geography

### 7.1. Product categories

![Figure 5 — Category frequencies at product level](../figures/clean/figure_05_categories.png)

*Figure 5. The denominator is all 31,875 retained products. The ten most frequent non-missing categories are shown individually; remaining categories are grouped as Other, and missing categories are shown separately. Other combines multiple source categories.*

`cama_mesa_banho` is the most frequent individual category, with **2,948 products (9.2486%)**. Other contains **11,181 products**, and Missing contains **582 products**. Categories outside the top ten remain in the dataset and the notebook's complete frequency table.

### 7.2. Customer and seller states

![Figure 6 — State frequencies at customer-record and seller levels](../figures/clean/figure_06_geography.png)

*Figure 6. The left panel uses 95,075 customer records identified by `customer_id`; the right panel uses 2,959 sellers. Customer records do not represent unique people. Each panel shows the ten most frequent states and groups the remaining states as Other.*

SP accounts for **39,900 customer records (41.9669%)** and **1,760 sellers (59.4796%)** using the respective denominators. The panels use separate units of observation and count scales. States outside the top ten remain in the dataset.
