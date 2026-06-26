"""Nodes for the `model_selection` pipeline.

Two nodes:
  1. compare_models — train each candidate with default params, pick the best type.
  2. tune_model     — Optuna tuning of the winner on the RFE-selected features.

SPLIT SEMANTICS:
  - X_train / X_val: from split_train (val is VALIDATION, not test).
  - test_data: true out-of-sample, evaluated in inference only.

Target is Price_log (log1p of Price); metrics are on the log scale.
"""

import logging
from datetime import datetime

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

_MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "XGBRegressor": XGBRegressor,
    "LGBMRegressor": LGBMRegressor,
}


def _as_1d(y) -> np.ndarray:
    return np.asarray(y).ravel()


def _build_model(name: str, params: dict, random_state: int):
    if name not in _MODELS:
        raise ValueError(f"Unknown candidate model: {name!r} (known: {list(_MODELS)})")
    kwargs = dict(params, random_state=random_state)
    if name == "RandomForestRegressor":
        kwargs.setdefault("n_jobs", -1)
    return _MODELS[name](**kwargs)


def _suggest(trial: "optuna.Trial", name: str, spec: dict):
    low, high = spec["low"], spec["high"]
    is_int = isinstance(low, int) and isinstance(high, int) and not spec.get("log", False)
    if is_int:
        return trial.suggest_int(name, low, high, step=spec.get("step", 1))
    return trial.suggest_float(name, low, high, log=spec.get("log", False))


def _evaluate(model, X, y_true) -> dict:
    pred_log = model.predict(X)
    y_log = _as_1d(y_true)
    pred_eur, y_eur = np.expm1(pred_log), np.expm1(y_log)
    return {
        "rmse_log": float(root_mean_squared_error(y_log, pred_log)),
        "mae_log": float(mean_absolute_error(y_log, pred_log)),
        "r2": float(r2_score(y_log, pred_log)),
        "rmse_eur": float(root_mean_squared_error(y_eur, pred_eur)),
        "mae_eur": float(mean_absolute_error(y_eur, pred_eur)),
    }


def _register_pass1(model, name: str, rmse: float, run_id: str, model_name: str):
    """Log and register a pass1 model in the MLflow Model Registry (no alias)."""
    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)

    mlflow.sklearn.log_model(model, artifact_path=f"pass1_{name}")
    model_uri = f"runs:/{run_id}/pass1_{name}"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.set_model_version_tag(model_name, mv.version, "pass", "pass1")
    client.set_model_version_tag(model_name, mv.version, "model_type", name)
    client.set_model_version_tag(model_name, mv.version, "val_rmse_log", f"{rmse:.6f}")
    logger.info("  Registered %s pass1 → version %s (RMSE=%.4f)", name, mv.version, rmse)


def compare_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    parameters: dict,
):
    """Train all candidate models with default params, register all as pass1, return best.

    All 4 models are logged to the MLflow Model Registry with tag pass=pass1 (no alias).
    The best model (lowest val RMSE) is returned to flow into feature_selection → tune_model.

    Returns:
        best_model — fitted model of the winning type (default params, all features).
    """
    random_state = parameters.get("random_state", 42)
    candidates = parameters["candidates"]
    registry_cfg = parameters.get("registry", {})
    model_name = registry_cfg.get("model_name", "house_price_model")

    y_train_arr = _as_1d(y_train)
    y_val_arr = _as_1d(y_val)
    active_run = mlflow.active_run()
    use_mlflow = active_run is not None

    logger.info("Comparing %d candidate model type(s) with default params...", len(candidates))
    val_rmse_by_candidate: dict[str, float] = {}
    fitted_models: dict[str, object] = {}

    for name in candidates:
        model = _build_model(name, {}, random_state)
        model.fit(X_train, y_train_arr)
        rmse = float(root_mean_squared_error(y_val_arr, model.predict(X_val)))
        val_rmse_by_candidate[name] = rmse
        fitted_models[name] = model
        logger.info("  %s: validation RMSE(log) = %.4f", name, rmse)
        if use_mlflow:
            mlflow.log_metric(f"candidate_val_rmse_{name}", rmse)
            _register_pass1(model, name, rmse, active_run.info.run_id, model_name)

    best_name = min(val_rmse_by_candidate, key=val_rmse_by_candidate.get)
    logger.info("Best candidate: %s (RMSE=%.4f)", best_name, val_rmse_by_candidate[best_name])

    if use_mlflow:
        mlflow.log_param("best_model_type", best_name)

    return fitted_models[best_name]


def tune_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    best_model,
    best_columns: list,
    parameters: dict,
):
    """Optuna tuning of the champion model type on the RFE-selected features.

    Args:
        best_model:   fitted model from compare_models (used to get the model type).
        best_columns: feature list from feature_selection (RFE output).

    Returns:
        selected_model — tuned model refit on X_train[best_columns].
    """
    best_name = type(best_model).__name__
    random_state = parameters.get("random_state", 42)
    n_trials = parameters.get("n_trials", 30)
    search_spaces = parameters["search_spaces"]

    X_train_fs = X_train[best_columns]
    X_val_fs = X_val[best_columns]
    y_train_arr = _as_1d(y_train)
    y_val_arr = _as_1d(y_val)

    logger.info(
        "Optuna tuning of %s on %d selected features (%d trials)...",
        best_name, len(best_columns), n_trials,
    )

    use_mlflow = mlflow.active_run() is not None
    space = search_spaces.get(best_name, {})

    def objective(trial: "optuna.Trial") -> float:
        trial_params = {p: _suggest(trial, p, spec) for p, spec in space.items()}
        model = _build_model(best_name, trial_params, random_state)
        model.fit(X_train_fs, y_train_arr)
        rmse = float(root_mean_squared_error(y_val_arr, model.predict(X_val_fs)))
        if use_mlflow:
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                mlflow.log_params(trial_params)
                mlflow.log_metric("val_rmse_log", rmse)
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)

    default_rmse = float(root_mean_squared_error(y_val_arr, best_model.predict(X_val) if hasattr(best_model, "predict") else [0]*len(y_val_arr)))
    if study.best_value <= default_rmse:
        best_params = study.best_params
        logger.info("Best params: %s (RMSE=%.4f)", best_params, study.best_value)
    else:
        best_params = {}
        logger.info("Tuning did not beat default — keeping default params.")

    selected_model = _build_model(best_name, best_params, random_state)
    selected_model.fit(X_train_fs, y_train_arr)
    metrics = _evaluate(selected_model, X_val_fs, y_val_arr)

    logger.info(
        "Tuned %s | val RMSE(log)=%.4f, R2=%.4f | RMSE=%.0f EUR, MAE=%.0f EUR",
        best_name, metrics["rmse_log"], metrics["r2"], metrics["rmse_eur"], metrics["mae_eur"],
    )

    if use_mlflow:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        mlflow.set_tag("mlflow.runName", f"{best_name}_tuned_{timestamp}")
        mlflow.log_param("selected_model_type", best_name)
        mlflow.log_param("n_features_selected", len(best_columns))
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

    return selected_model
