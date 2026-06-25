"""Pipeline `preprocessing_train` — pre-split cleaning only."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data
from kedro_temp_mlops.utils import upload_cleaned_to_fs


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_data,
            inputs=["learning_data", "params:preprocessing"],
            outputs=["cleaned_data", "reporting_data_preprocessing"],
            name="clean_data",
        ),
        node(
            func=upload_cleaned_to_fs,
            inputs=["cleaned_data", "params:ingestion"],
            outputs="cleaned_data_fs_done",
            name="upload_cleaned_to_fs_node",
        ),
    ])
