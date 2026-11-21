# Merge Report

This report describes the full item-level merge of five Olist sources. See the [notebook](../notebooks/00_data_merging.ipynb), [User Guide](../docs/Merge/User_Guide.md) and [Technical Reference](../docs/Merge/Technical_Reference.md).

## Contents

1. [Scope and key result](#1-scope-and-key-result)
2. [Data and definitions](#2-data-and-definitions)
3. [Findings by workflow stage](#3-findings-by-workflow-stage)
4. [Conclusions and limitations](#4-conclusions-and-limitations)

## 1. Scope and key result

The full merge contains **112,650 order items, 25 columns and 98,666 distinct orders**. Each row represents an order item and the composite key `(order_id, order_item_id)` is unique. Four left joins retain all item rows; each has zero unmatched left rows in this reference dataset.

## 2. Data and definitions

The selected source tables are order items, orders, customers, products and sellers. The item table is the base; joining an order-level or product-level attribute repeats it for every applicable item. Customer identifiers represent purchase-linked customer records rather than a count of distinct people. Counts refer to the supplied reference data and can change with the input.

| Source | Rows | Selected columns |
|---|---:|---:|
| Order items | 112,650 | 7 |
| Orders | 99,441 | 8 |
| Customers | 99,441 | 4 |
| Products | 32,951 | 6 |
| Sellers | 3,095 | 4 |

## 3. Findings by workflow stage

Each join uses a left many-to-one relationship. The recorded unmatched count refers to left input rows with no matching right-side key.

| Stage | Right source and join key | Input item rows | Output item rows | Output columns | Unmatched left rows |
|---|---|---:|---:|---:|---:|
| Base | Order items | 112,650 | 112,650 | 7 | — |
| 1 | Orders by `order_id` | 112,650 | 112,650 | 14 | 0 |
| 2 | Customers by `customer_id` | 112,650 | 112,650 | 17 | 0 |
| 3 | Products by `product_id` | 112,650 | 112,650 | 22 | 0 |
| 4 | Sellers by `seller_id` | 112,650 | 112,650 | 25 | 0 |

All 112,650 base items retain their original row count through every stage. Required links were found for each item, and the output item key is complete and unique. Matching a product key does not imply that every product attribute is present.

The orders source contains **99,441 orders**. The full item-based merge contains **98,666 orders**, because **775 source orders have no item record**. The `orders + customers` combination uses the order-level base and retains all **99,441 orders**. These combinations cover different populations.

## 4. Conclusions and limitations

The full merge provides an item-level input for EDA and cleaning. It retains observed order statuses and missing source attributes. EDA examines those values at their appropriate item, order, product, customer-record and seller grains before cleaning applies its own rules.

The saved dataset is `merged_order_items` in `data/interim/`, with Parquet preferred and CSV fallback. The join report is held in memory during execution. The values in this report describe the reference input; use the User Guide to reproduce the merge or inspect other table combinations.
