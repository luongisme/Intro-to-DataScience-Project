import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import orders

items = pd.read_csv('src/data/olist_order_items_dataset.csv', parse_dates = ['shipping_limit_date'])
orders_deadline = items.groupby('order_id')['shipping_limit_date'].max()

orders = orders.merge(orders_deadline, on = 'order_id')

orders['seller_delay_date'] = (orders['order_delivered_carrier_date'].dt.normalize() - orders['shipping_limit_date'].dt.normalize()).dt.days


plt.figure(figsize=(8, 5))
sns.scatterplot(data = orders, x ='seller_delay_date', y ='delay_days', alpha=0.5)
plt.xlabel('Seller delay date')
plt.ylabel('Delay days')
plt.title('Seller delay vs Customer late delivery')
plt.show()