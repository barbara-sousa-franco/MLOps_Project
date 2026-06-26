"""Pytest tests for split_data (out-of-sample ref/ana carve-off)."""

import pandas as pd
import numpy as np

from kedro_temp_mlops.pipelines.split_data.nodes import split_out_of_sample

BASE_PARAMS = {
    "strategy": "random",
    "ref_frac": 0.8,
    "random_state": 42,
}


def _df(n=100):
    return pd.DataFrame({
        "District": (["Lisboa"] * 30) + (["Porto"] * 40) + (["Faro"] * 30),
        "Price": np.arange(n) * 1000.0,
    })


def test_random_proportions():
    learning, test = split_out_of_sample(_df(100), BASE_PARAMS)
    assert len(learning) == 80          # ref_frac 0.8
    assert len(test) == 20              # the remaining 20%


def test_no_overlap_learning_test():
    learning, test = split_out_of_sample(_df(100), BASE_PARAMS)
    assert set(learning.index) & set(test.index) == set()


def test_random_is_reproducible():
    l1, t1 = split_out_of_sample(_df(100), BASE_PARAMS)
    l2, t2 = split_out_of_sample(_df(100), BASE_PARAMS)
    pd.testing.assert_frame_equal(l1, l2)
    pd.testing.assert_frame_equal(t1, t2)