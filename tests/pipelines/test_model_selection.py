"""Pytest tests for model_selection (compare_models and tune_model nodes)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


PARAMS = {
    "random_state": 42,
    "n_trials": 3,
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


@pytest.mark.slow
def test_compare_models_returns_fitted_model():
    """compare_models returns the best fitted model from the candidates."""
    from kedro_temp_mlops.pipelines.model_selection.nodes import compare_models

    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.random((60, 4)), columns=["f1", "f2", "f3", "f4"])
    X_val   = pd.DataFrame(rng.random((20, 4)), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(rng.random(60) + 10)
    y_val   = pd.Series(rng.random(20) + 10)

    model = compare_models(X_train, X_val, y_train, y_val, PARAMS)

    assert isinstance(model, (RandomForestRegressor, GradientBoostingRegressor))
    preds = model.predict(X_val)
    assert preds.shape == (20,)


@pytest.mark.slow
def test_tune_model_returns_fitted_model_with_selected_features():
    """tune_model returns a fitted model trained only on best_columns."""
    from kedro_temp_mlops.pipelines.model_selection.nodes import compare_models, tune_model, _evaluate

    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.random((60, 4)), columns=["f1", "f2", "f3", "f4"])
    X_val   = pd.DataFrame(rng.random((20, 4)), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(rng.random(60) + 10)
    y_val   = pd.Series(rng.random(20) + 10)
    best_columns = ["f1", "f3"]  # simulated RFE output

    best_model = compare_models(X_train, X_val, y_train, y_val, PARAMS)
    selected = tune_model(X_train, X_val, y_train, y_val, best_model, best_columns, PARAMS)

    assert isinstance(selected, (RandomForestRegressor, GradientBoostingRegressor))
    preds = selected.predict(X_val[best_columns])
    assert preds.shape == (20,)

    metrics = _evaluate(selected, X_val[best_columns], y_val)
    assert np.isfinite(metrics["rmse_log"]) and metrics["rmse_log"] >= 0
    assert metrics["r2"] <= 1.0
