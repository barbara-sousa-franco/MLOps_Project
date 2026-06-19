"""Nodes for the `feature_selection` pipeline (SHAP).

WARNING — BUG IN THE EXAMPLE NOT TO COPY: `shap_values[:,:,1]` is a class index
(classification). For REGRESSION there is no class axis — use `shap_values` 2D directly.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap(production_model, X_train: pd.DataFrame, parameters: dict):
    """Compute SHAP values for the champion model and generate the summary plot.

    Args:
        production_model: trained champion (from model_train).
        X_train: training features.
        parameters: SHAP config (parameters_model_train.yml or _model_selection.yml).

    Returns:
        Tuple (shap_values, shap_plot) — shap_plot is saved as an MLflow artifact (png).

    TODO compute_shap:
      - shap.TreeExplainer(model); shap_values = explainer(X_train)
      - WARNING: shap.summary_plot(shap_values, X_train, ...) — 2D directly.
        Do NOT use shap_values[:,:,1] (class index; bug in the example)
      - generate shap_plot.png -> MLflow artifact
    """
    # TODO: implement
    raise NotImplementedError


def select_features(shap_values, X_train: pd.DataFrame, parameters: dict):
    """Select the top-N features by mean |SHAP|.

    Args:
        shap_values: output of `compute_shap`.
        X_train: training features (to map column names).
        parameters: N or selection threshold.

    Returns:
        best_cols — list of columns to save in best_cols.pkl (artifact).

    TODO select_features:
      - top-N features by mean |SHAP| (N or threshold in parameters)
      - save best_cols.pkl (artifact)
      - (professor's tip: feature selection from SHAP; best_cols feeds back into
        model_train via use_feature_selection)
    """
    # TODO: implement
    raise NotImplementedError
