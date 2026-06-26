"""Pipeline `model_selection` — two nodes: compare then tune."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compare_models, tune_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=compare_models,
                inputs=[
                    "X_train_scaled",
                    "X_val_scaled",
                    "y_train_data",
                    "y_val_data",
                    "params:model_selection",
                ],
                outputs="best_model",
                name="compare_models_node",
            ),
            node(
                func=tune_model,
                inputs=[
                    "X_train_scaled",
                    "X_val_scaled",
                    "y_train_data",
                    "y_val_data",
                    "best_model",
                    "best_columns",
                    "params:model_selection",
                ],
                outputs="selected_model",
                name="tune_model_node",
            ),
        ]
    )
