"""
This is a boilerplate test file for pipeline 'split_train_pipeline'
generated using Kedro 1.3.1.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
"""Pytest tests for split_train (train/validation split)."""

import numpy as np
import pandas as pd

from kedro_temp_mlops.pipelines.split_train_pipeline.nodes import split_train

PARAMS = {
    "target_col": "Price_log",
    "random_state": 2021,
    "test_size": 0.2,
    "stratify_bins": None,
}


def _df(n=20):
    return pd.DataFrame({
        "Feature1": np.arange(n),
        "Feature2": np.arange(n, 2 * n),
        "Price_log": np.linspace(10, 14, n),
    })


def test_split_train_shapes():
    X_train, X_val, y_train, y_val, cols = split_train(_df(10), PARAMS)
    assert X_train.shape == (8, 2)      # 2 features, target removed
    assert X_val.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_val.shape == (2,)
    assert "Price_log" not in cols


def test_split_is_reproducible():
    X1, Xv1, _, _, _ = split_train(_df(), PARAMS)
    X2, Xv2, _, _, _ = split_train(_df(), PARAMS)
    pd.testing.assert_frame_equal(X1, X2)      # same seed -> identical split
    pd.testing.assert_frame_equal(Xv1, Xv2)


def test_no_overlap_train_val():
    X_train, X_val, _, _, _ = split_train(_df(), PARAMS)
    assert set(X_train.index) & set(X_val.index) == set()