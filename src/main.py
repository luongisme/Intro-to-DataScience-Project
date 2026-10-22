import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

orders = pd.read_csv('src/data/olist_orders_dataset.csv', parse_dates = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
orders = orders[orders['order_status'] == 'delivered']
main_columns = ['order_id', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
orders = orders.dropna(subset = main_columns)

orders["is_late"] = orders["order_delivered_customer_date"].dt.normalize() > orders["order_estimated_delivery_date"]
orders["delay_days"] = (orders[orders["is_late"]]["order_delivered_customer_date"].dt.normalize() - orders[orders["is_late"]]["order_estimated_delivery_date"]).dt.days
