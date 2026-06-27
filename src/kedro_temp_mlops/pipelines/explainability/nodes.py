"""Nodes for the `explainability` pipeline.

Computes SHAP values on the final champion model and logs them as MLflow artifacts.
Runs AFTER the `training` pipeline (compare_models → feature_selection → tune → train).
"""

import logging
import os
import pickle
import tempfile
from datetime import datetime

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

    # sample to keep SHAP computation tractable (TreeExplainer scales poorly with n_samples)
    n_shap = parameters.get("n_shap_samples", 500)
    if len(X_val) > n_shap:
        X_val = X_val.sample(n=n_shap, random_state=42)
        logger.info("Sampled %d rows from X_val for SHAP computation.", n_shap)

    explainer = shap.TreeExplainer(production_model)
    shap_values = explainer(X_val)

    # save explainer to disk so the notebook can load base_values for waterfall plots
    disk_explainer_path = os.path.join("data", "08_reporting", "shap_explainer.pkl")
    with open(disk_explainer_path, "wb") as f:
        pickle.dump(explainer, f)

    use_mlflow = mlflow.active_run() is not None
    if use_mlflow:
        model_type = type(production_model).__name__
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        mlflow.set_tag("mlflow.runName", f"{model_type}_shap_{timestamp}")
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
