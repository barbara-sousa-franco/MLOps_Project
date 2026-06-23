"""Pipeline `preprocessing_train` — pre-split cleaning only."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs=["learning_data", "params:preprocessing"],
            outputs=["cleaned_data", "reporting_data_preprocessing"],
            name="clean_data",
        ),
    ])
