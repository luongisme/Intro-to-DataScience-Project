import pandas as pd


def load_csv(path, **kwargs):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path, **kwargs)


def basic_schema_summary(df):
    """Return a basic schema summary for a DataFrame."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": df.isna().sum().values,
            "non_null_count": df.notna().sum().values,
        }
    )
