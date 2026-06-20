"""Nodes for the `model_selection` pipeline.

Regression model selection in two steps:
  1. Challenger comparison — train each candidate model type with default params and
     compare RMSE on an internal validation split.
  2. Optuna tuning — tune the winning model type (direction="minimize" on RMSE), logging
     each trial as a nested MLflow run.

The target is `Price_log` (log1p of Price), so metrics are computed on the log scale and
also inverted with expm1 to report RMSE/MAE in euros. The sacred test set (`X_test`) is
only used for the final evaluation, never during the search.
"""

import logging

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Candidate model registry: name (from parameters) -> sklearn estimator class.
_MODELS = {
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}


def _as_1d(y) -> np.ndarray:
    """Coerce a target (DataFrame/Series/array) into a 1-D numpy array."""
    return np.asarray(y).ravel()


def _build_model(name: str, params: dict, random_state: int):
    """Instantiate a candidate model by name with the given hyperparameters.

    Adds `random_state` for reproducibility and `n_jobs=-1` where supported (RandomForest).
    """
    if name not in _MODELS:
        raise ValueError(f"Unknown candidate model: {name!r} (known: {list(_MODELS)})")
    kwargs = dict(params, random_state=random_state)
    if name == "RandomForestRegressor":
        kwargs.setdefault("n_jobs", -1)
    return _MODELS[name](**kwargs)


def _suggest(trial: "optuna.Trial", name: str, spec: dict):
    """Suggest one hyperparameter from a search-space spec.

    Integer spec -> suggest_int (honours `step`); otherwise -> suggest_float (honours `log`).
    """
    low, high = spec["low"], spec["high"]
    is_int = isinstance(low, int) and isinstance(high, int) and not spec.get("log", False)
    if is_int:
        return trial.suggest_int(name, low, high, step=spec.get("step", 1))
    return trial.suggest_float(name, low, high, log=spec.get("log", False))


def _evaluate(model, X, y_true) -> dict:
    """Compute regression metrics on the log scale and inverted to euros."""
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


def model_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    parameters: dict,
    champion_dict: dict | None = None,
    champion_model=None,
):
    """Compare challengers, tune the best one with Optuna and return the selected model.

    Args:
        X_train, X_test, y_train, y_test: data from the split (target is `Price_log`).
        parameters: candidates + Optuna search spaces (parameters_model_selection.yml).
        champion_dict: metrics of the current champion (state from a previous run; None on
            the first execution). Optional and NOT wired in the graph to avoid cycles.
        champion_model: current champion model (same).

    Returns:
        selected_model — the tuned best model, refit on the full training set (or the
        existing champion if it is still better).
    """
    random_state = parameters.get("random_state", 42)
    val_size = parameters.get("val_size", 0.2)
    n_trials = parameters.get("n_trials", 30)
    candidates = parameters["candidates"]
    search_spaces = parameters["search_spaces"]

    y_train = _as_1d(y_train)
    y_test = _as_1d(y_test)
    use_mlflow = mlflow.active_run() is not None

    # Internal validation holdout — keeps X_test untouched during the search.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    # ---- STEP 1: compare candidate model types (default params) ----------------
    logger.info("STEP 1 — comparing %d candidate model type(s)...", len(candidates))
    val_rmse_by_candidate: dict[str, float] = {}
    for name in candidates:
        model = _build_model(name, {}, random_state)
        model.fit(X_tr, y_tr)
        rmse = float(root_mean_squared_error(y_val, model.predict(X_val)))
        val_rmse_by_candidate[name] = rmse
        logger.info("  %s: validation RMSE(log) = %.4f", name, rmse)
        if use_mlflow:
            mlflow.log_metric(f"candidate_val_rmse_{name}", rmse)

    best_name = min(val_rmse_by_candidate, key=val_rmse_by_candidate.get)
    logger.info("Best candidate type: %s", best_name)

    # ---- STEP 2: Optuna tuning of the winning model type -----------------------
    logger.info("STEP 2 — Optuna tuning of %s (%d trials)...", best_name, n_trials)
    space = search_spaces[best_name]

    def objective(trial: "optuna.Trial") -> float:
        trial_params = {p: _suggest(trial, p, spec) for p, spec in space.items()}
        model = _build_model(best_name, trial_params, random_state)
        model.fit(X_tr, y_tr)
        rmse = float(root_mean_squared_error(y_val, model.predict(X_val)))
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

    # Guard: never ship a tuned model that is worse than the default baseline.
    default_val_rmse = val_rmse_by_candidate[best_name]
    if study.best_value <= default_val_rmse:
        best_params = study.best_params
        logger.info("Best params: %s (validation RMSE(log) = %.4f)", best_params, study.best_value)
    else:
        best_params = {}
        logger.info(
            "Tuning did not beat the default (%.4f vs default %.4f) — keeping default params. "
            "Consider more n_trials or a wider search space.",
            study.best_value, default_val_rmse,
        )

    # ---- STEP 3: refit best model on full train, evaluate on the test set ------
    selected_model = _build_model(best_name, best_params, random_state)
    selected_model.fit(X_train, y_train)
    metrics = _evaluate(selected_model, X_test, y_test)
    logger.info(
        "Selected %s | test RMSE(log)=%.4f, R2=%.4f | test RMSE=%.0f EUR, MAE=%.0f EUR",
        best_name, metrics["rmse_log"], metrics["r2"], metrics["rmse_eur"], metrics["mae_eur"],
    )

    if use_mlflow:
        mlflow.log_param("selected_model_type", best_name)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    # ---- Optional: keep the current champion if it is still better -------------
    if champion_model is not None and champion_dict is not None:
        champion_rmse = champion_dict.get("test_rmse_log", float("inf"))
        if champion_rmse <= metrics["rmse_log"]:
            logger.info(
                "Champion RMSE(log)=%.4f <= challenger %.4f — keeping champion.",
                champion_rmse, metrics["rmse_log"],
            )
            return champion_model

    return selected_model
