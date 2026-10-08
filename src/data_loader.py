import pandas as pd

from src.paths import RAW_DIR


DATASETS = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load_data(name):
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available datasets: {list(DATASETS.keys())}"
        )

    path = RAW_DIR / DATASETS[name]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: '{DATASETS[name]}' searched in folder '{RAW_DIR}'.\n"
            "Please place the required file in data/raw/."
        )

    return pd.read_csv(path)

def load_all():
    return {
        name: load_data(name)
        for name in DATASETS
    }