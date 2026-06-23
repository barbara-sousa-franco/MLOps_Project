"""Pipeline `feature_selection` — RFE selection."""


from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=feature_selection,
            inputs=["X_train_scaled", "y_train_data", "params:feature_selection"],
            outputs="best_columns",
            name="model_feature_selection",
        ),
    ])