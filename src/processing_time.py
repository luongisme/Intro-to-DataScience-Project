import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import orders

orders["processing_days"] = (orders[orders["is_late"]]["order_delivered_carrier_date"].dt.normalize() - orders[orders["is_late"]]["order_approved_at"].dt.normalize()).dt.days
late = orders[orders["processing_days"] >= 0]

plt.figure(figsize=(8, 5))
sns.scatterplot(data=late, x='processing_days', y='delay_days', alpha=0.2)
plt.xlabel('Processing time')
plt.ylabel('Delay days')
plt.title('Processing time vs delivery delay')
plt.show()