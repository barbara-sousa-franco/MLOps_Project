"""Nodes for the `model_train` pipeline.

REGRESSION metrics (RMSE, MAE, R²), never accuracy. Always compare against a baseline
(mean of Price) — the skill requires a baseline.

=============================================================================
NOTES FOR IMPLEMENTERS (decisions already taken — please follow):
-----------------------------------------------------------------------------
1. CHAMPION/CHALLENGER + MLflow Model Registry live HERE.
   feature_selection keeps only SHAP -> best_columns.

2. MODEL INPUTS = `X_train_scaled` / `X_val_scaled` (05_model_input layer, after
   impute -> cap -> encode -> scale).

3. SPLIT SEMANTICS (professor's scheme):
   - `X_train` = training data (FIT here).
   - `X_val` = leak-free VALIDATION set -> used for the
     champion-vs-challenger PROMOTION decision.
   - `test_data` (split_data) = true out-of-sample TEST set -> honest final metric
     computed in Phase 3 (preprocessing_batch -> model_predict), NOT here.

4. `selected_model` (from model_selection) is ALREADY tuned and fit on X_train.

5. BASELINE = DummyRegressor(strategy="mean") — naive "predict the mean" baseline.
=============================================================================
"""

import logging
import math

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


def model_train(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    parameters: dict,
    selected_model=None,
    best_columns=None,
):
    """Train the champion, evaluate on the validation set, and return model + metrics.

    Returns:
        Tuple (production_model, production_columns, production_model_metrics).
        Metrics are VALIDATION metrics; the honest test number comes from test_data (Phase 3).
    """
    use_fs = parameters.get("use_feature_selection", False)
    if use_fs and best_columns:
        X_train = X_train[best_columns]
        X_val = X_val[best_columns]

    production_columns = list(X_train.columns)

    # --- build candidate model ---
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
    """Register the trained model in the MLflow Model Registry with champion/challenger logic.

    - New model always enters as 'challenger'.
    - If it beats the current 'champion' (lower val_rmse) it is promoted to 'champion'.
    - If no champion exists yet, the new model is immediately promoted.

    Returns:
        dict with version info and whether the model was promoted.
    """
    registry_cfg = parameters.get("registry", {})
    model_name = registry_cfg.get("model_name", "house_price_model")
    challenger_alias = registry_cfg.get("challenger_alias", "challenger")
    champion_alias = registry_cfg.get("champion_alias", "champion")

    client = MlflowClient()

    # ensure the registered model exists
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)
        logger.info("Created registered model '%s'", model_name)

    # log and register the model under the active run
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("register_model must be called inside an active MLflow run.")

    run_id = active_run.info.run_id
    mlflow.sklearn.log_model(model, artifact_path="model")
    model_uri = f"runs:/{run_id}/model"

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registered '%s' version %s", model_name, mv.version)

    # tag as challenger first
    client.set_registered_model_alias(model_name, challenger_alias, mv.version)
    logger.info("Tagged version %s as '%s'", mv.version, challenger_alias)

    # champion/challenger comparison
    new_rmse = metrics.get("val_rmse")
    promoted = False

    try:
        champion_mv = client.get_model_version_by_alias(model_name, champion_alias)
        champion_run = client.get_run(champion_mv.run_id)
        champion_rmse = champion_run.data.metrics.get("val_rmse")

        if champion_rmse is not None and new_rmse is not None:
            logger.info(
                "Champion RMSE=%.4f  vs  Challenger RMSE=%.4f", champion_rmse, new_rmse
            )
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
        # no champion yet — promote immediately
        client.set_registered_model_alias(model_name, champion_alias, mv.version)
        logger.info("No existing champion — version %s promoted directly.", mv.version)
        promoted = True

    return {
        "model_name": model_name,
        "version": mv.version,
        "promoted_to_champion": promoted,
        "val_rmse": new_rmse,
    }
