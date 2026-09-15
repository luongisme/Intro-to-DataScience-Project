import pandas as pd


def duplicate_summary(df):
    """Return duplicate row counts."""
    duplicate_count = int(df.duplicated().sum())
    return {
        "row_count": int(len(df)),
        "duplicate_count": duplicate_count,
        "duplicate_percent": (duplicate_count / len(df) * 100) if len(df) else 0.0,
    }


def missing_summary(df):
    """Return missing-value counts and percentages by column."""
    missing_count = df.isna().sum()
    return pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percent": (missing_count.values / len(df) * 100) if len(df) else 0.0,
        }
    )
