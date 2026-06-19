"""Nodes for the `model_selection` pipeline.

Adapts the professor's pattern for REGRESSION + Optuna (instead of GridSearchCV).
Regression metric (lower RMSE = better!), never accuracy.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name: str) -> str:
    """Return the MLflow experiment id, creating it if it does not exist (same as the professor).

    TODO _get_or_create_experiment_id:
      - mlflow.get_experiment_by_name(name); if None -> mlflow.create_experiment(name)
      - return experiment_id
    """
    # TODO: implement
    raise NotImplementedError


def model_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: dict,
    champion_dict: dict | None = None,
    champion_model=None,
):
    """Compare challengers, tune with Optuna and compare against the current champion.

    Args:
        X_train, X_test, y_train, y_test: data from the split.
        parameters: candidates + Optuna search spaces (parameters_model_selection.yml).
        champion_dict: metrics of the current champion (state from a previous run; None on
            the first execution). Optional and NOT wired in the graph to avoid cycles —
            load internally from the registry/artifact when available.
        champion_model: current champion model (same).

    Returns:
        selected_model — the best model (tuned challenger OR existing champion),
        passed to `model_train`.

    TODO model_selection:
      STEP 1 — compare model types (challengers):
        - candidates: RandomForestRegressor, GradientBoostingRegressor,
          (XGBoost/LightGBM optional)
        - mlflow.sklearn.autolog; train each one; metric = RMSE or R² on test
        - choose the best type
      STEP 2 — tuning with OPTUNA:
        - def objective(trial): suggest hyperparameters from the search space in parameters
        - optuna.create_study(direction="minimize" for RMSE), n_trials from parameters
        - each trial in a nested mlflow run
      STEP 3 — compare against current champion:
        - if new score BETTER than champion_dict (lower RMSE is better!) -> return new model
        - otherwise -> return existing champion
      - return selected_model
    """
    # TODO: implement
    raise NotImplementedError
