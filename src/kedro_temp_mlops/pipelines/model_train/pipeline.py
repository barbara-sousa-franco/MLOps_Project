"""Pipeline `model_train`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train, register_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the champion training pipeline + Model Registry registration.

    outputs: `production_model`, `production_columns`, `production_model_metrics`.
    """
    return pipeline(
        [
            node(
                func=model_train,
                # best_columns comes from feature_selection (RFE) — Kedro runs
                # feature_selection first automatically via this data dependency.
                # use_feature_selection in params controls whether they are applied.
                inputs=[
                    "X_train_scaled",
                    "X_val_scaled",
                    "y_train_data",
                    "y_val_data",
                    "params:model_train",
                    "selected_model",
                    "best_columns",
                ],
                outputs=[
                    "production_model",
                    "production_columns",
                    "production_model_metrics",
                ],
                name="model_train_node",
            ),
            node(
                func=register_model,
                inputs=[
                    "production_model",
                    "production_model_metrics",
                    "params:model_train",
                ],
                outputs="registered_model_version",
                name="register_model_node",
            ),
        ]
    )
