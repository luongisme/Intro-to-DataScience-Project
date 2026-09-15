import pandas as pd


def grouped_mean(df, group_col, value_col):
    """Compute mean of value_col by group_col."""
    return df.groupby(group_col, dropna=False)[value_col].mean().reset_index(name="mean_value")


def correlation_pair(df, col_x, col_y):
    """Return Pearson correlation between two numeric columns."""
    return df[[col_x, col_y]].corr().iloc[0, 1]
