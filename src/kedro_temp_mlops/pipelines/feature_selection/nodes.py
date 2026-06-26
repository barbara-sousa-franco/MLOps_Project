"""Nodes for the `feature_selection` pipeline.

RFE on the best model from compare_models.
Runs between compare_models and tune_model.

Workflow:
  1. compare_models  → best_model (winning type, default params, all features)
  2. feature_selection → RFE on best_model → best_columns
  3. tune_model      → Optuna on best_model type with best_columns
  4. model_train     → retrain + register
  5. explainability  → SHAP on final champion

Output: `best_columns` list saved via catalog.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE

logger = logging.getLogger(__name__)


def feature_selection(
    X_train: pd.DataFrame,
    y_train,
    best_model,
    parameters: Dict[str, Any],
):
    """RFE on the best model from compare_models.

    Args:
        best_model: fitted model returned by compare_models.
        parameters: feature_selection params (n_features_to_select, method).

    Returns:
        best_columns — list of selected feature names.
    """
    logger.info("Feature selection starting with %d columns.", len(X_train.columns))

    y_train = np.ravel(y_train)
    method = parameters.get("method", "rfe")

    if method == "rfe":
        n_features = parameters.get("n_features_to_select")
        rfe = RFE(best_model, n_features_to_select=n_features)
        rfe.fit(X_train, y_train)
        best_columns = X_train.columns[rfe.get_support(1)].tolist()
    else:
        best_columns = X_train.columns.tolist()

    logger.info("Selected %d columns: %s", len(best_columns), best_columns)
    return best_columns
