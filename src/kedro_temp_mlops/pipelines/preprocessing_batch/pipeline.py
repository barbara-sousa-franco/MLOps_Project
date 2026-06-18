"""Pipeline `preprocessing_batch`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_batch


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de preprocessing do batch.

    Recebe `encoder_transform` (do treino) como input. outputs: `preprocessed_batch_data`.
    """
    return pipeline(
        [
            node(
                func=preprocess_batch,
                inputs=["ana_data", "encoder_transform", "params:preprocessing"],
                outputs="preprocessed_batch_data",
                name="preprocess_batch_node",
            ),
        ]
    )
