"""Nodes for the `feature_selection` pipeline.

RFE (Recursive Feature Elimination) on a regression estimator. Uses the champion
model if it exists on disk, otherwise a baseline RandomForestRegressor — mirrors
the professor's bank-example pattern, adapted from classification to regression.

SPLIT SEMANTICS (professor's scheme — names kept, roles clarified):
  - `X_train` = training data; RFE FITS here only.
  - `X_test` from split_train = the leak-free validation set (despite the name) —
    used downstream for tuning / model comparison, NOT here.
  - `ana_data` = the true out-of-sample test set, evaluated later (inference).

Output: `best_columns` — the RFE-selected feature list. Selection only; explainability
(SHAP) is handled separately in model_train, per the project requirements.
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


def feature_selection(X_train: pd.DataFrame, y_train, parameters: Dict[str, Any]):
    """RFE feature selection (regression). Uses the champion if it exists on disk,
    else a baseline RandomForestRegressor — mirrors the professor's pattern."""
    logger.info("We start with: %d columns", len(X_train.columns))

    X_cols = X_train.columns.tolist()

    if parameters["feature_selection"] == "rfe":
        y_train = np.ravel(y_train)

        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                estimator = pickle.load(f)
        except Exception:
            estimator = RandomForestRegressor(**parameters['baseline_model_params'])

        rfe = RFE(estimator, n_features_to_select=parameters.get("n_features_to_select"))
        rfe = rfe.fit(X_train, y_train)
        X_cols = X_train.columns[rfe.get_support(1)].tolist()

    logger.info("Number of best columns is: %d", len(X_cols))
    return X_cols