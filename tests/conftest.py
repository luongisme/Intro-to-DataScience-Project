"""Pytest fixtures with small synthetic DataFrames for unit testing."""

import pandas as pd
import pytest


@pytest.fixture
def synthetic_order_items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2"],
            "order_item_id": [1, 2, 1],
            "product_id": ["P1", "P2", "P1"],
            "seller_id": ["S1", "S1", "S2"],
            "shipping_limit_date": pd.to_datetime(
                ["2017-02-05 10:00:00", "2017-02-05 10:00:00", "2017-03-01 12:00:00"]
            ),
            "price": [10.5, 20.0, 15.75],
            "freight_value": [5.0, 5.0, 4.2],
        }
    )


@pytest.fixture
def synthetic_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2"],
            "customer_id": ["C1", "C2"],
            "order_status": ["delivered", "shipped"],
            "order_purchase_timestamp": pd.to_datetime(["2017-02-01 08:00:00", "2017-02-25 09:00:00"]),
            "order_approved_at": pd.to_datetime(["2017-02-01 08:30:00", "2017-02-25 09:30:00"]),
            "order_delivered_carrier_date": pd.to_datetime(["2017-02-02 14:00:00", "2017-02-26 15:00:00"]),
            "order_delivered_customer_date": pd.to_datetime(["2017-02-04 18:00:00", pd.NaT]),
            "order_estimated_delivery_date": pd.to_datetime(["2017-02-10 00:00:00", "2017-03-05 00:00:00"]),
        }
    )


@pytest.fixture
def synthetic_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "customer_zip_code_prefix": ["01310", "04571"],
            "customer_city": ["sao paulo", "sao paulo"],
            "customer_state": ["SP", "SP"],
        }
    )


@pytest.fixture
def synthetic_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "product_category_name": ["beleza_saude", "informatica_acessorios"],
            "product_weight_g": [500.0, 250.0],
            "product_length_cm": [20.0, 15.0],
            "product_height_cm": [10.0, 5.0],
            "product_width_cm": [15.0, 10.0],
        }
    )


@pytest.fixture
def synthetic_sellers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seller_id": ["S1", "S2"],
            "seller_zip_code_prefix": ["02011", "08420"],
            "seller_city": ["sao paulo", "sao paulo"],
            "seller_state": ["SP", "SP"],
        }
    )