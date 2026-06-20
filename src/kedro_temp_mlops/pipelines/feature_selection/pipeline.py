"""Pipeline `feature_selection`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_shap, select_features


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature-selection pipeline (SHAP of the champion).

    Runs AFTER model_train. outputs: `shap_plot`, `best_columns`.
    """
    return pipeline(
        [
            node(
                func=compute_shap,
                inputs=["production_model", "X_train_encoded", "params:model_train"],
                outputs=["shap_values", "shap_plot"],
                name="compute_shap_node",
            ),
            node(
                func=select_features,
                inputs=["shap_values", "X_train_encoded", "params:model_train"],
                outputs="best_columns",
                name="select_features_node",
            ),
        ]
    )
