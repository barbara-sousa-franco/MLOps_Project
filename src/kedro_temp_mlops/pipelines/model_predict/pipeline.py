"""Pipeline `model_predict`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de inferência. outputs: `predictions`."""
    return pipeline(
        [
            node(
                func=predict,
                inputs=[
                    "production_model",
                    "preprocessed_batch_data",
                    "best_columns",
                    "params:model_train",
                ],
                outputs="predictions",
                name="predict_node",
            ),
        ]
    )
