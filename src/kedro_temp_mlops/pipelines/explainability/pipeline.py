"""Pipeline `explainability` — SHAP on the final champion."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_shap


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_shap,
            inputs=["production_model", "X_val_scaled", "params:model_train"],
            outputs=["shap_values", "shap_importance"],
            name="compute_shap_node",
        ),
    ])
