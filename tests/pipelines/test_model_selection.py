"""Pytest tests for model_selection."""
import pytest


def test_returns_trained_model_and_valid_metric():
    """model_selection returns a trained model and a valid metric.

    TODO:
      - run model_selection on the sample (few n_trials)
      - assert the model has .predict and that RMSE/R² is finite and in the expected domain
    """
    pytest.skip("TODO: implement when model_selection is ready")

"""Pytest tests for model_selection (trains models — marked slow)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


@pytest.mark.slow
def test_model_selection_returns_fitted_model():
    from kedro_temp_mlops.pipelines.model_selection.nodes import model_selection

    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.random((60, 4)), columns=["f1", "f2", "f3", "f4"])
    X_val   = pd.DataFrame(rng.random((20, 4)), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(rng.random(60) + 10)      # Price_log-like positive values
    y_val   = pd.Series(rng.random(20) + 10)

    parameters = {
        "random_state": 42,
        "n_trials": 3,                            # tiny — fast
        "use_feature_selection": False,
        "candidates": ["RandomForestRegressor", "GradientBoostingRegressor"],
        "search_spaces": {
            "RandomForestRegressor": {
                "n_estimators": {"low": 10, "high": 30},
                "max_depth": {"low": 2, "high": 5},
            },
            "GradientBoostingRegressor": {
                "n_estimators": {"low": 10, "high": 30},
                "learning_rate": {"low": 0.01, "high": 0.3, "log": True},
            },
        },
    }

    model = model_selection(X_train, X_val, y_train, y_val, parameters)

    # returns a fitted regressor of one of the candidate types
    assert isinstance(model, (RandomForestRegressor, GradientBoostingRegressor))
    # it's fitted -> can predict, and predict returns the right shape
    preds = model.predict(X_val)
    assert preds.shape == (20,)


@pytest.mark.slow
def test_model_selection_keeps_better_champion():
    """If an existing champion has a better (lower) RMSE, it's kept."""
    from kedro_temp_mlops.pipelines.model_selection.nodes import model_selection

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.random((40, 3)), columns=["a", "b", "c"])
    X_val   = pd.DataFrame(rng.random((15, 3)), columns=["a", "b", "c"])
    y_train = pd.Series(rng.random(40) + 10)
    y_val   = pd.Series(rng.random(15) + 10)

    parameters = {
        "random_state": 42, "n_trials": 2, "use_feature_selection": False,
        "candidates": ["RandomForestRegressor"],
        "search_spaces": {"RandomForestRegressor": {"n_estimators": {"low": 10, "high": 20}}},
    }

    sentinel = RandomForestRegressor().fit(X_train, y_train)   # a stand-in champion
    champion_dict = {"val_rmse_log": -1.0}    # impossibly good -> champion must win

    result = model_selection(X_train, X_val, y_train, y_val, parameters,
                             champion_dict=champion_dict, champion_model=sentinel)
    assert result is sentinel                 # champion kept, not the challenger