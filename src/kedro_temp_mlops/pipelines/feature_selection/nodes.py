"""Nodes for the `feature_selection` pipeline.

RFE (Recursive Feature Elimination) on a regression estimator. Loads the
production_model from disk if it exists (from a previous training run), otherwise
falls back to a baseline RandomForestRegressor.

Cannot receive production_model as a Kedro input — that would create a cycle:
  feature_selection → best_columns → model_train → production_model → feature_selection

Instead, loads the artifact from its catalog path directly when available.

SPLIT SEMANTICS:
  - `X_train` = training data; RFE FITS here only.
  - `X_val` = leak-free validation set — used downstream, NOT here.
  - `test_data` = true out-of-sample test set, evaluated in Phase 3.

Output: `best_columns` — the RFE-selected feature list.
"""

import logging
import os
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

logger = logging.getLogger(__name__)

_PRODUCTION_MODEL_PATH = os.path.join("data", "06_models", "production_model.pkl")


def feature_selection(
    X_train: pd.DataFrame,
    y_train,
    parameters: Dict[str, Any],
):
    """RFE feature selection. Uses the production model from a previous run if available,
    otherwise falls back to a baseline RandomForestRegressor."""
    logger.info("Feature selection starting with %d columns", len(X_train.columns))

    X_cols = X_train.columns.tolist()
    method = parameters.get("method", "rfe")

    if method == "rfe":
        y_train = np.ravel(y_train)

        try:
            with open(_PRODUCTION_MODEL_PATH, "rb") as f:
                estimator = pickle.load(f)
            logger.info("Loaded production model from %s for RFE.", _PRODUCTION_MODEL_PATH)
        except FileNotFoundError:
            estimator = RandomForestRegressor(**parameters["baseline_model_params"])
            logger.info("No production model found — using baseline RF for RFE.")

        rfe = RFE(estimator, n_features_to_select=parameters.get("n_features_to_select"))
        rfe.fit(X_train, y_train)
        X_cols = X_train.columns[rfe.get_support(1)].tolist()

    logger.info("Selected %d columns: %s", len(X_cols), X_cols)
    return X_cols
