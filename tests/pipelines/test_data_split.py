"""Pytest tests for split_data."""

import pytest


def test_split_shapes():
    """X_train/X_test/y_train/y_test with consistent shapes and test_size respected.

    TODO: call split_data and validate proportions.
    """
    pytest.skip("TODO: implement when split_data is ready")


def test_no_overlap_train_test():
    """There must be no index overlap between train and test.

    TODO: assert set(X_train.index) & set(X_test.index) == set()
    """
    pytest.skip("TODO: implement when split_data is ready")


def test_split_is_reproducible():
    """Same seed -> same split.

    TODO: run split_data twice with the same seed and compare indices.
    """
    pytest.skip("TODO: implement when split_data is ready")
