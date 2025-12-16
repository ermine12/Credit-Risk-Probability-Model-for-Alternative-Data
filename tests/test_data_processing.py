import numpy as np
import pandas as pd
import pytest

from src.data_processing import TransactionDatetimeFeatures, calculate_rfm, make_preprocess_pipeline


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerId": ["A", "A", "B", "C"],
            "TransactionStartTime": pd.to_datetime(
                ["2023-01-01 10:00", "2023-01-02 11:00", "2023-01-01 12:00", "2023-01-03 13:00"]
            ),
            "Amount": [100, -50, 200, 1000],
            "Value": [100, 50, 200, 1000],
            "ProviderId": ["P1", "P1", "P2", "P1"],
            "ProductCategory": ["cat1", "cat2", "cat1", "cat3"],
            "ChannelId": ["C1", "C1", "C2", "C3"],
            "some_other_numeric": [1, 2, 3, 4],
            "some_other_object": ["x", "y", "z", "w"],
        }
    )


def test_make_preprocess_pipeline_runs(sample_df: pd.DataFrame) -> None:
    pipe = make_preprocess_pipeline(use_woe=False)
    X_transformed = pipe.fit_transform(sample_df)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(sample_df)
    assert not np.isnan(X_transformed).any()


def test_transaction_datetime_features(sample_df: pd.DataFrame) -> None:
    transformer = TransactionDatetimeFeatures()
    df_transformed = transformer.fit_transform(sample_df)

    expected_cols = ['tx_hour', 'tx_day', 'tx_month', 'tx_year', 'tx_dayofweek', 'tx_is_weekend']
    for col in expected_cols:
        assert col in df_transformed.columns
    assert 'TransactionStartTime' not in df_transformed.columns
    assert df_transformed['tx_is_weekend'].isin([0, 1]).all()


def test_calculate_rfm(sample_df: pd.DataFrame) -> None:
    rfm_df = calculate_rfm(sample_df)

    assert all(col in rfm_df.columns for col in ['Recency', 'Frequency', 'Monetary'])
    assert rfm_df.index.name == 'CustomerId'
    assert rfm_df.loc['A', 'Recency'] == 2  # 2023-01-04 (snapshot) - 2023-01-02
    assert rfm_df.loc['A', 'Frequency'] == 2
    assert rfm_df.loc['A', 'Monetary'] == 150 # 100 + 50
