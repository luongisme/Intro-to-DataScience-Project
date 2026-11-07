"""Pytest fixtures with small synthetic DataFrames for unit testing."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DATASETS


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


@pytest.fixture
def synthetic_pipeline_dirs(
    tmp_path: Path,
    monkeypatch,
    synthetic_order_items,
    synthetic_orders,
    synthetic_customers,
    synthetic_products,
    synthetic_sellers,
) -> dict[str, Path]:
    """Provide isolated raw sources and interim output for full pipeline tests.

    Reuse the small DataFrame fixtures to write UTF-8 raw CSV files. Route
    production loading/saving to pytest's temporary directory. Leave the
    interim directory absent so tests can verify failures happen before I/O.
    The monkeypatch fixture restores all three paths after each test.
    """
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    raw_dir.mkdir()
    sources = {
        "order_items": synthetic_order_items,
        "orders": synthetic_orders,
        "customers": synthetic_customers,
        "products": synthetic_products,
        "sellers": synthetic_sellers,
    }
    for name, frame in sources.items():
        frame.to_csv(raw_dir / DATASETS[name], index=False, encoding="utf-8")

    monkeypatch.setattr("src.data_loader.RAW_DIR", raw_dir)
    monkeypatch.setattr("src.data_loader.INTERIM_DIR", interim_dir)
    monkeypatch.setattr("src.pipeline.run_merge.INTERIM_DIR", interim_dir)
    return {"raw": raw_dir, "interim": interim_dir}


@pytest.fixture
def eda_full_frame() -> pd.DataFrame:
    """Provide three canonical items with repeated indexes and two valid orders.

    O1 has two sellers, two products, two deadlines and mixed state comparisons.
    O2 repeats P1/S1 with consistent attributes. Values are specified independently
    of merge_data so EDA expectations do not depend on the production merger.
    """
    from src.pipeline.contracts.schema import FULL_MERGE_SCHEMA

    values = {
        "order_id": ["O1", "O1", "O2"], "order_item_id": [1, 2, 1],
        "product_id": ["P1", "P2", "P1"], "seller_id": ["S1", "S2", "S1"],
        "shipping_limit_date": ["2017-01-02 12:00", "2017-01-03 12:00", "2017-02-02 12:00"],
        "price": [10.0, 20.0, 30.0], "freight_value": [2.0, 0.0, 4.0],
        "customer_id": ["C1", "C1", "C2"], "order_status": ["delivered"] * 3,
        "order_purchase_timestamp": ["2017-01-01 10:00", "2017-01-01 10:00", "2017-02-01 10:00"],
        "order_approved_at": ["2017-01-01 11:00", "2017-01-01 11:00", "2017-02-01 11:00"],
        "order_delivered_carrier_date": ["2017-01-02 10:00", "2017-01-02 10:00", "2017-02-02 10:00"],
        "order_delivered_customer_date": ["2017-01-03 10:00", "2017-01-03 10:00", "2017-02-03 10:00"],
        "order_estimated_delivery_date": ["2017-01-05", "2017-01-05", "2017-02-05"],
        "customer_zip_code_prefix": ["00123", "00123", "00456"],
        "customer_city": ["city a", "city a", "city b"], "customer_state": ["SP", "SP", "RJ"],
        "product_category_name": ["cat1", "cat2", "cat1"], "product_weight_g": [100., 200., 100.],
        "product_length_cm": [10., 20., 10.], "product_height_cm": [5., 6., 5.], "product_width_cm": [4., 8., 4.],
        "seller_zip_code_prefix": ["00123", "00789", "00123"],
        "seller_city": ["city a", "city b", "city a"], "seller_state": ["SP", "RJ", "SP"],
    }
    frame = pd.DataFrame({
        column: pd.Series(values[column], dtype=FULL_MERGE_SCHEMA.dtypes[column])
        for column in FULL_MERGE_SCHEMA.columns
    })
    frame.index = pd.Index([7, 7, 2], name="source_index")
    return frame


@pytest.fixture
def eda_plot_cleanup():
    """Close test figures without changing the production plotting defaults."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
