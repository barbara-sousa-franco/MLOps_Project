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
                # `best_columns` is optional and NOT wired on the 1st pass (avoids a cycle
                # with feature_selection). 2nd pass: add "best_columns" here + set
                # use_feature_selection=true in parameters_model_train.yml.
                inputs=[
                    "X_train_encoded",
                    "X_test_encoded",
                    "y_train_data",
                    "y_test_data",
                    "params:model_train",
                    "selected_model",
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
