"""Pipeline `preprocessing_batch`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_batch


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_batch,
            inputs=["test_data", "imputer", "capper",
                    "target_encoder", "scaler", "params:preprocessing"],
            outputs="preprocessed_batch_data",
            name="preprocess_batch_node",
        ),
    ])
