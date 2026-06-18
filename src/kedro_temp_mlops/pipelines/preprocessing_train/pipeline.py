"""Pipeline `preprocessing_train`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data, encode_features


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de preprocessing do treino.

    outputs: `preprocessed_training_data`, `encoder_transform`.
    """
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["ingested_data", "params:preprocessing"],
                outputs="clean_training_data",
                name="clean_data_node",
            ),
            node(
                func=encode_features,
                inputs=["clean_training_data", "params:preprocessing"],
                outputs=["preprocessed_training_data", "encoder_transform"],
                name="encode_features_node",
            ),
        ]
    )
