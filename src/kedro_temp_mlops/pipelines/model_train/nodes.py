"""Nodes for the `model_train` pipeline.

REGRESSION metrics (RMSE, MAE, R²), never accuracy. Always compare against a baseline
(mean of Price) — the skill requires a baseline.

Runs as part of the `training` composition:
  model_selection (compare + tune) → feature_selection (RFE) → model_train

SPLIT SEMANTICS:
  - X_train / X_val: from split_train (val is VALIDATION, not test)
  - test_data: true out-of-sample, evaluated in inference only
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
    best_columns: list | None = None,
):
    """Evaluate the tuned model against the baseline and return model + metrics.

    The selected_model comes from tune_model (already tuned on best_columns).
    best_columns are used to restrict X_train/X_val to the same feature set.

    Returns:
        Tuple (production_model, production_columns, production_model_metrics).
    """
    if best_columns:
        X_train = X_train[best_columns]
        X_val = X_val[best_columns]
        logger.info("Using %d selected features.", len(best_columns))

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

    production_model_metrics = {
        **{f"val_{k}": v for k, v in candidate_metrics.items()},
        **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
    }

    return candidate, production_columns, production_model_metrics


def register_model(model, metrics: dict, parameters: dict):
    """Register the tuned model (pass2) in the MLflow Model Registry.

    Logic:
    - pass2 is always registered with tag pass=pass2.
    - If no @champion exists yet → this version becomes @champion.
    - If @champion exists → this version enters as @challenger.
      If its RMSE beats the champion → it is promoted to @champion.
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
    model_type = type(model).__name__

    mlflow.sklearn.log_model(model, artifact_path="pass2_model")
    model_uri = f"runs:/{run_id}/pass2_model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_model_version_tag(model_name, mv.version, "pass", "pass2")
    client.set_model_version_tag(model_name, mv.version, "model_type", model_type)

    new_rmse = metrics.get("val_rmse")
    if new_rmse is not None:
        client.set_model_version_tag(model_name, mv.version, "val_rmse", f"{new_rmse:.6f}")

    logger.info("Registered %s pass2 → version %s (RMSE=%s)", model_type, mv.version, new_rmse)

    promoted = False
    try:
        champion_mv = client.get_model_version_by_alias(model_name, champion_alias)
        champion_rmse_tag = champion_mv.tags.get("val_rmse")
        champion_rmse = float(champion_rmse_tag) if champion_rmse_tag else None

        if champion_rmse is not None and new_rmse is not None:
            logger.info("Champion RMSE=%.4f  vs  Challenger RMSE=%.4f", champion_rmse, new_rmse)
            if new_rmse < champion_rmse:
                client.set_registered_model_alias(model_name, champion_alias, mv.version)
                client.set_registered_model_alias(model_name, challenger_alias, champion_mv.version)
                logger.info("New model promoted to @champion (version %s)", mv.version)
                promoted = True
            else:
                client.set_registered_model_alias(model_name, challenger_alias, mv.version)
                logger.info("@champion retained (version %s) — new model is @challenger", champion_mv.version)
        else:
            client.set_registered_model_alias(model_name, champion_alias, mv.version)
            promoted = True

    except mlflow.exceptions.MlflowException:
        # no champion yet — first pass2 becomes champion
        client.set_registered_model_alias(model_name, champion_alias, mv.version)
        logger.info("No existing @champion — version %s promoted directly.", mv.version)
        promoted = True

    return {
        "model_name": model_name,
        "version": mv.version,
        "promoted_to_champion": promoted,
        "val_rmse": new_rmse,
    }
