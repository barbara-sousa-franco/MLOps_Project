"""Nodes for the `feature_selection` pipeline.

RFE (Recursive Feature Elimination) on the production model (champion).
Runs as a standalone pipeline AFTER the first `training` run.

Workflow:
  1. kedro run --pipeline training          → champion saved (all features)
  2. kedro run --pipeline feature_selection → RFE on champion → best_columns
  3. kedro run --pipeline training          → retrain on best_columns (use_feature_selection: true)
  4. kedro run --pipeline explainability    → SHAP on final champion

Output: `best_columns` saved to data/06_models/best_cols.pkl
"""

import logging
from typing import Any, Dict

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

logger = logging.getLogger(__name__)


def feature_selection(
    X_train: pd.DataFrame,
    y_train,
    parameters: Dict[str, Any],
):
    """RFE on the champion model from MLflow Model Registry.
    Falls back to baseline RF if no champion exists yet.
    """
    logger.info("Feature selection starting with %d columns.", len(X_train.columns))

    X_cols = X_train.columns.tolist()
    method = parameters.get("method", "rfe")

    if method == "rfe":
        y_train = np.ravel(y_train)

        try:
            estimator = mlflow.sklearn.load_model("models:/house_price_model@champion")
            logger.info("Loaded champion model from MLflow Model Registry for RFE.")
        except Exception as e:
            logger.warning("Could not load champion from registry (%s) — using baseline RF.", e)
            estimator = RandomForestRegressor(**parameters["baseline_model_params"])

        rfe = RFE(estimator, n_features_to_select=parameters.get("n_features_to_select"))
        rfe.fit(X_train, y_train)
        X_cols = X_train.columns[rfe.get_support(1)].tolist()

    logger.info("Selected %d columns: %s", len(X_cols), X_cols)
    return X_cols
