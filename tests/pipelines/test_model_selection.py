"""Pytest tests for model_selection."""

import pytest


def test_returns_trained_model_and_valid_metric():
    """model_selection returns a trained model and a valid metric.

    TODO:
      - run model_selection on the sample (few n_trials)
      - assert the model has .predict and that RMSE/R² is finite and in the expected domain
    """
    pytest.skip("TODO: implement when model_selection is ready")
