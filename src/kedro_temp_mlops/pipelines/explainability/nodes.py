"""Nodes for the `explainability` pipeline.

Computes SHAP values on the final champion model and logs them as MLflow artifacts.
Runs AFTER the final `training` run (with feature-selected columns).

Workflow:
  1. kedro run --pipeline training            (1st pass, all features)
  2. kedro run --pipeline feature_selection   (RFE → best_columns)
  3. kedro run --pipeline training            (2nd pass, best_columns)
  4. kedro run --pipeline explainability      ← THIS PIPELINE
"""

import logging
import os
import pickle
import tempfile

import mlflow
import pandas as pd
import shap

logger = logging.getLogger(__name__)

_BEST_COLUMNS_PATH = os.path.join("data", "06_models", "best_cols.pkl")


def compute_shap(
    production_model,
    X_val: pd.DataFrame,
    parameters: dict,
):
    """Compute SHAP values on X_val using the production model and log to MLflow.

    If best_cols.pkl exists, restricts X_val to those columns before computing SHAP
    (must match the columns the model was trained on).
    """
    # align columns with what the model was trained on
    try:
        with open(_BEST_COLUMNS_PATH, "rb") as f:
            best_columns = pickle.load(f)
        X_val = X_val[best_columns]
        logger.info("Restricted X_val to %d selected features for SHAP.", len(best_columns))
    except FileNotFoundError:
        logger.info("No best_cols.pkl — using all features for SHAP.")

    explainer = shap.TreeExplainer(production_model)
    shap_values = explainer(X_val)

    use_mlflow = mlflow.active_run() is not None
    if use_mlflow:
        with tempfile.TemporaryDirectory() as tmp:
            explainer_path = os.path.join(tmp, "shap_explainer.pkl")
            with open(explainer_path, "wb") as f:
                pickle.dump(explainer, f)
            mlflow.log_artifact(explainer_path, artifact_path="shap")

            shap_values_path = os.path.join(tmp, "shap_values.pkl")
            with open(shap_values_path, "wb") as f:
                pickle.dump(shap_values, f)
            mlflow.log_artifact(shap_values_path, artifact_path="shap")

        logger.info("SHAP explainer and values logged to MLflow.")

    # mean absolute SHAP per feature → log as metrics
    shap_importance = pd.DataFrame(
        shap_values.values, columns=X_val.columns
    ).abs().mean().sort_values(ascending=False)

    if use_mlflow:
        for feat, val in shap_importance.items():
            mlflow.log_metric(f"shap_{feat}", float(val))

    logger.info("Top 5 features by SHAP:\n%s", shap_importance.head())

    return shap_values.values, shap_importance.reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    )
