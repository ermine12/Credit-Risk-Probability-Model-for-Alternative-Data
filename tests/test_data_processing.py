import numpy as np
import pandas as pd
import pytest

from src.data_processing import make_preprocess_pipeline


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
