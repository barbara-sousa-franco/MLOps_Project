"""Nodes for the `model_train` pipeline.

REGRESSION metrics (RMSE, MAE, R²), never accuracy. Always compare against a baseline
(mean of Price) — the skill requires a baseline.

=============================================================================
WORKFLOW (two-pass):
  Pass 1 — kedro run --pipeline training  (use_feature_selection: false)
      model_selection + model_train on all features → champion saved
  Pass 2 — kedro run --pipeline feature_selection
      RFE on champion → best_columns saved to data/06_models/best_cols.pkl
  Pass 3 — kedro run --pipeline training  (use_feature_selection: true)
      model_selection + model_train on best_columns → final champion
  Pass 4 — kedro run --pipeline explainability
      SHAP on final champion → artifacts logged to MLflow

SPLIT SEMANTICS:
  - X_train / X_val: from split_train (val is VALIDATION, not test)
  - test_data: true out-of-sample, evaluated in Phase 3 only
=============================================================================
"""

import logging
import math
import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

_BEST_COLUMNS_PATH = os.path.join("data", "06_models", "best_cols.pkl")


def _regression_metrics(y_true, y_pred) -> dict:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _load_best_columns() -> list | None:
    """Load best_columns from disk if feature_selection has already run."""
    try:
        with open(_BEST_COLUMNS_PATH, "rb") as f:
            cols = pickle.load(f)
        logger.info("Loaded %d best_columns from %s.", len(cols), _BEST_COLUMNS_PATH)
        return cols
    except FileNotFoundError:
        return None


def model_train(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    parameters: dict,
    selected_model=None,
):
    """Train the champion, evaluate on the validation set, return model + metrics.

    If use_feature_selection=true and best_cols.pkl exists (from feature_selection
    pipeline), restricts features to those columns before training.

    Returns:
        Tuple (production_model, production_columns, production_model_metrics).
    """
    use_fs = parameters.get("use_feature_selection", False)
    if use_fs:
        best_columns = _load_best_columns()
        if best_columns:
            X_train = X_train[best_columns]
            X_val = X_val[best_columns]
            logger.info("Using %d selected features.", len(best_columns))
        else:
            logger.warning("use_feature_selection=true but best_cols.pkl not found — using all features.")

    production_columns = list(X_train.columns)

    # --- candidate model ---
    if selected_model is not None:
        candidate = selected_model
    else:
        candidate = RandomForestRegressor(**parameters["baseline_model_params"])

    candidate.fit(X_train, y_train)

    # --- baseline (mean predictor) ---
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    candidate_metrics = _regression_metrics(y_val, candidate.predict(X_val))
    baseline_metrics = _regression_metrics(y_val, baseline.predict(X_val))

    logger.info(
        "Candidate   → RMSE=%.4f  MAE=%.4f  R²=%.4f",
        candidate_metrics["rmse"], candidate_metrics["mae"], candidate_metrics["r2"],
    )
    logger.info(
        "Baseline    → RMSE=%.4f  MAE=%.4f  R²=%.4f",
        baseline_metrics["rmse"], baseline_metrics["mae"], baseline_metrics["r2"],
    )

    use_mlflow = mlflow.active_run() is not None
    if use_mlflow:
        mlflow.log_metrics({f"val_{k}": v for k, v in candidate_metrics.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_param("use_feature_selection", use_fs)

    production_model_metrics = {
        **{f"val_{k}": v for k, v in candidate_metrics.items()},
        **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
    }

    return candidate, production_columns, production_model_metrics


def register_model(model, metrics: dict, parameters: dict):
    """Register the trained model in the MLflow Model Registry with champion/challenger logic.

    - New model always enters as 'challenger'.
    - If it beats the current 'champion' (lower val_rmse) → promoted to 'champion'.
    - If no champion exists yet → promoted directly.
    """
    registry_cfg = parameters.get("registry", {})
    model_name = registry_cfg.get("model_name", "house_price_model")
    challenger_alias = registry_cfg.get("challenger_alias", "challenger")
    champion_alias = registry_cfg.get("champion_alias", "champion")

    client = MlflowClient()

    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)
        logger.info("Created registered model '%s'", model_name)

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("register_model must be called inside an active MLflow run.")

    run_id = active_run.info.run_id
    mlflow.sklearn.log_model(model, artifact_path="model")
    model_uri = f"runs:/{run_id}/model"

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registered '%s' version %s", model_name, mv.version)

    client.set_registered_model_alias(model_name, challenger_alias, mv.version)
    logger.info("Tagged version %s as '%s'", mv.version, challenger_alias)

    new_rmse = metrics.get("val_rmse")
    promoted = False

    try:
        champion_mv = client.get_model_version_by_alias(model_name, champion_alias)
        champion_run = client.get_run(champion_mv.run_id)
        champion_rmse = champion_run.data.metrics.get("val_rmse")

        if champion_rmse is not None and new_rmse is not None:
            logger.info("Champion RMSE=%.4f  vs  Challenger RMSE=%.4f", champion_rmse, new_rmse)
            if new_rmse < champion_rmse:
                client.set_registered_model_alias(model_name, champion_alias, mv.version)
                logger.info("Challenger promoted to champion (version %s)", mv.version)
                promoted = True
            else:
                logger.info("Champion retained (version %s)", champion_mv.version)
        else:
            logger.warning("Could not compare RMSEs — promoting challenger by default.")
            client.set_registered_model_alias(model_name, champion_alias, mv.version)
            promoted = True

    except mlflow.exceptions.MlflowException:
        client.set_registered_model_alias(model_name, champion_alias, mv.version)
        logger.info("No existing champion — version %s promoted directly.", mv.version)
        promoted = True

    return {
        "model_name": model_name,
        "version": mv.version,
        "promoted_to_champion": promoted,
        "val_rmse": new_rmse,
    }
