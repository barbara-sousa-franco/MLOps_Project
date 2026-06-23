"""Nodes for the `feature_selection` pipeline.

SHAP selects features from the champion (also = the required explainability plot),
then an Optuna-tuned challenger of a DIFFERENT model family is trained on the
selected features and promoted only if it beats the champion on the validation set.

SPLIT SEMANTICS (professor's scheme — names kept, roles clarified):
  - `X_train` = training data (FIT here).
  - `X_test` from split_train = the leak-free VALIDATION set (despite the name) — used to
    tune the challenger (holdout) and to decide champion-vs-challenger.
  - `ana_data` = the true out-of-sample TEST set, evaluated later (inference / Phase 3).

Leakage discipline: SHAP + challenger tuning FIT on X_train only; the validation set is
used for tuning and the promotion decision. The honest test (ana_data) is touched once,
downstream.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# 1. SHAP — selection signal + mandatory explainability
# ----------------------------------------------------------------------------
def compute_shap(production_model, X_train: pd.DataFrame, parameters: dict):
    """SHAP values + beeswarm summary for the champion (regression: 2D, no class axis)."""
    explainer = shap.TreeExplainer(production_model)   # champion is tree-based
    shap_values = explainer(X_train)                   # .values: (n_samples, n_features)

    logger.info("SHAP computed: %s", shap_values.values.shape)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return shap_values, fig


def select_features(shap_values, X_train: pd.DataFrame, parameters: dict) -> list:
    """Top-N features by mean(|SHAP|)."""
    cfg = parameters.get("shap", {})
    top_n = cfg.get("top_n", 15)
    threshold = cfg.get("threshold")

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X_train.columns).sort_values(ascending=False)

    best_cols = (importance[importance >= threshold].index.tolist()
                 if threshold is not None
                 else importance.head(top_n).index.tolist())

    logger.info("Selected %d/%d features: %s", len(best_cols), len(importance), best_cols)
    return best_cols


# ----------------------------------------------------------------------------
# 2. Challenger — Optuna-tuned, different family, on the selected features
# ----------------------------------------------------------------------------
def tune_challenger(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
                    best_columns: list, parameters: dict):
    """Optuna-tuned HistGradientBoosting challenger on best_columns.

    Holdout (NOT CV): fit on X_train, evaluate on the leak-free validation set X_val
    (= X_test from split_train), consistent with model_selection.
    """
    cfg = parameters.get("challenger", {})
    n_trials = cfg.get("n_trials", 30)
    seed = parameters["random_state"]

    X_tr, y_tr = X_train[best_columns], np.ravel(y_train)
    X_va, y_va = X_val[best_columns], np.ravel(y_val)

    def objective(trial):
        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_iter":          trial.suggest_int("max_iter", 100, 600),
            "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 15, 255),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 1.0, log=True),
            "random_state": seed,
        }
        model = HistGradientBoostingRegressor(**params).fit(X_tr, y_tr)
        return float(root_mean_squared_error(y_va, model.predict(X_va)))   # minimise RMSE

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {**study.best_params, "random_state": seed}
    challenger = HistGradientBoostingRegressor(**best_params).fit(X_tr, y_tr)

    logger.info("Challenger best validation RMSE=%.4f | params=%s", study.best_value, best_params)
    return challenger, best_params


# ----------------------------------------------------------------------------
# 3. Promote — champion vs challenger on the validation set
# ----------------------------------------------------------------------------
def compare_and_promote(production_model, challenger_model, challenger_params,
                        X_test: pd.DataFrame, y_test, best_columns: list, parameters: dict):
    """Evaluate champion vs challenger on the VALIDATION set; return the better model.

    `X_test`/`y_test` here are the split_train validation set (despite the name). This is
    a SELECTION decision on validation; the honest test (ana_data) is evaluated downstream.
    No MLflow calls here — logging is handled by the catalog (kedro-mlflow datasets).
    """
    y_true = np.ravel(y_test)

    champ_pred = production_model.predict(X_test)                 # full features
    chall_pred = challenger_model.predict(X_test[best_columns])   # selected features

    champion_rmse   = float(np.sqrt(mean_squared_error(y_true, champ_pred)))
    champion_r2     = float(r2_score(y_true, champ_pred))
    challenger_rmse = float(np.sqrt(mean_squared_error(y_true, chall_pred)))
    challenger_r2   = float(r2_score(y_true, chall_pred))

    challenger_wins = challenger_rmse < champion_rmse
    best_model = challenger_model if challenger_wins else production_model
    winner = "challenger" if challenger_wins else "champion"

    # native MLflow metrics -> logged by MlflowMetricsHistoryDataset
    metrics = {
        "champion_rmse":   {"value": champion_rmse,   "step": 0},
        "champion_r2":     {"value": champion_r2,     "step": 0},
        "challenger_rmse": {"value": challenger_rmse, "step": 0},
        "challenger_r2":   {"value": challenger_r2,   "step": 0},
    }

    # full record -> logged as a JSON artifact
    report = {
        "winner": winner,
        "champion_rmse": champion_rmse, "champion_r2": champion_r2,
        "challenger_rmse": challenger_rmse, "challenger_r2": challenger_r2,
        "n_features_champion": int(X_test.shape[1]),
        "n_features_challenger": len(best_columns),
        "challenger_params": challenger_params,
    }

    logger.info("Winner: %s | champion_rmse=%.4f challenger_rmse=%.4f",
                winner, champion_rmse, challenger_rmse)
    return best_model, metrics, report