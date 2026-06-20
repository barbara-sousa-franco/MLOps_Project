"""Nodes for the `feature_selection` pipeline (SHAP).

WARNING — BUG IN THE EXAMPLE NOT TO COPY: `shap_values[:,:,1]` is a class index
(classification). For REGRESSION there is no class axis — use `shap_values` 2D directly.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap(production_model, X_train: pd.DataFrame, parameters: dict):
    """Compute SHAP values for the champion model and generate the summary plot.

    Args:
        production_model: trained champion (from model_train).
        X_train: training features.
        parameters: SHAP config (parameters_model_train.yml or _model_selection.yml).

    Returns:
        Tuple (shap_values, shap_plot):
          - shap_values: shap.Explanation object (2D: n_samples x n_features) for
            REGRESSION. There is no class axis here (that only exists for
            classification) — do NOT slice it like `shap_values[:, :, 1]`.
          - shap_plot: matplotlib Figure with the SHAP summary (beeswarm) plot,
            saved as an MLflow artifact (png) via the catalog.
    """
    # TreeExplainer works directly with our candidates (RandomForestRegressor /
    # GradientBoostingRegressor) — much faster than the model-agnostic KernelExplainer.
    explainer = shap.TreeExplainer(production_model)

    # shap_values is a shap.Explanation with .values of shape (n_samples, n_features)
    # for regression (single output) — NO class dimension, unlike classification.
    shap_values = explainer(X_train)

    logger.info(
        "SHAP values computed: shape=%s for %d features.",
        shap_values.values.shape,
        X_train.shape[1],
    )

    # Summary (beeswarm) plot — global view of feature impact on the prediction.
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, show=False)
    fig = plt.gcf()  # shap.summary_plot draws on the current figure
    fig.tight_layout()

    return shap_values, fig


def select_features(shap_values, X_train: pd.DataFrame, parameters: dict) -> list[str]:
    """Select the top-N features by mean |SHAP|.

    Args:
        shap_values: output of `compute_shap` (shap.Explanation, 2D for regression).
        X_train: training features (to map column names — order must match the
            columns used when `compute_shap` built the explainer/values).
        parameters: `model_train` dict; reads `shap.top_n` (and optionally
            `shap.threshold`, both defined in parameters_model_train.yml).

    Returns:
        best_cols — list of column names (length top_n, or fewer if a threshold
        is used and fewer features qualify), ordered by decreasing importance.
        Saved as best_cols.pkl (MLflow artifact) by the catalog.
    """
    shap_cfg = parameters.get("shap", {})
    top_n = shap_cfg.get("top_n", 15)
    threshold = shap_cfg.get("threshold")

    # mean(|SHAP|) per feature = global importance (standard SHAP convention).
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    importance = pd.Series(mean_abs_shap, index=X_train.columns).sort_values(
        ascending=False
    )

    if threshold is not None:
        best_cols = importance[importance >= threshold].index.tolist()
    else:
        best_cols = importance.head(top_n).index.tolist()

    logger.info(
        "Feature selection (SHAP): %d/%d columns kept -> %s",
        len(best_cols),
        len(importance),
        best_cols,
    )

    return best_cols