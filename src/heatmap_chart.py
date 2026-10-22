import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main import orders

orders["week_day"] = orders["order_approved_at"].dt.dayofweek

def time_slot(hours):
    if 8 <= hours < 12:
        return "8h-12h"
    elif 12 <= hours < 18:
        return "12h-18h"
    elif 18 <= hours < 24:
        return "18h-24h"
    else:
        return "0h-8h"

orders["time_slot"] = orders["order_approved_at"].dt.hour.apply(time_slot)
table = orders.pivot_table(index = 'week_day', columns = 'time_slot', values = "is_late", aggfunc = 'mean') * 100

table = table[['8h-12h', '12h-18h', '18h-24h', '0h-8h']]
table.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.figure(figsize=(8, 5))
sns.heatmap(table, annot = True, fmt = ".1f", cmap = "Blues")
plt.xlabel('Time slot (order_approved_at)')
plt.ylabel('Day of week')
plt.title('Late order rate (%) by time slot and day of week')
plt.show()
