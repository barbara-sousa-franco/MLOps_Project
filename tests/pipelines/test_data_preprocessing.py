"""Pytest tests for preprocessing (runs on the sample, NOT to be confused with data_unit_tests)."""

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_PATH = Path(__file__).parent / "sample" / "sample.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """~200 representative rows from portugal_listings for fast tests."""
    return pd.read_csv(SAMPLE_PATH)


def test_clean_data_no_nulls_after_imputation(sample_df):
    """clean_data must not leave nulls in the imputed columns.

    TODO:
      - call clean_data(sample_df, parameters)
      - assert result[imputed_columns].isna().sum() == 0
    """
    pytest.skip("TODO: implement when clean_data is ready")


def test_encoder_not_fit_on_test(sample_df):
    """The encoder is fitted on training data only — anti-leakage.

    TODO:
      - fit encoder on training data; ensure that transform on the batch does NOT
        modify the encoder (same categories/statistics)
    """
    pytest.skip("TODO: implement when encode_features is ready")
