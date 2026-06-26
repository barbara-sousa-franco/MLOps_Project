"""Pytest tests for model_selection."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


@pytest.mark.slow
def test_model_selection_returns_fitted_model_and_valid_metric():
    from kedro_temp_mlops.pipelines.model_selection.nodes import model_selection, _evaluate

    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.random((60, 4)), columns=["f1", "f2", "f3", "f4"])
    X_val   = pd.DataFrame(rng.random((20, 4)), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(rng.random(60) + 10)
    y_val   = pd.Series(rng.random(20) + 10)

    parameters = {
        "random_state": 42, "n_trials": 3, "use_feature_selection": False,
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

    # 1. returns a fitted model with .predict
    assert isinstance(model, (RandomForestRegressor, GradientBoostingRegressor))
    preds = model.predict(X_val)
    assert preds.shape == (20,)

    # 2. the metric is finite and in a valid domain
    metrics = _evaluate(model, X_val, y_val)
    assert np.isfinite(metrics["rmse_log"]) and metrics["rmse_log"] >= 0
    assert metrics["r2"] <= 1.0                      # R² can be negative on random data, but never > 1


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