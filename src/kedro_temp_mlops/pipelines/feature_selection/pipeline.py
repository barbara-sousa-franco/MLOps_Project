"""Pipeline `feature_selection` — RFE on the best model from compare_models."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=feature_selection,
            inputs=["X_train_scaled", "y_train_data", "best_model", "params:feature_selection"],
            outputs="best_columns",
            name="model_feature_selection",
        ),
    ])
