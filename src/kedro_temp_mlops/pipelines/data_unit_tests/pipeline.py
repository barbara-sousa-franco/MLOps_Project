"""Pipeline for `data_unit_tests`.

Three validation nodes:
  1. unit_test_raw           — validates raw ingested_data
  2. unit_test_cleaned_data  — validates cleaned_data (post-preprocessing, pre-split)
  3. unit_test_model_input   — validates X_train / X_val (ready for modelling)

The raw checkpoint feeds write_traffic_light, which gates downstream pipelines.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    unit_test_raw,
    unit_test_cleaned_data,
    unit_test_model_input,
    write_traffic_light,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=unit_test_raw,
            inputs=["ingested_data", "params:data_unit_tests"],
            outputs="reporting_tests_raw",
            name="unit_test_raw_node",
        ),
        node(
            func=unit_test_cleaned_data,
            inputs=["cleaned_data", "params:data_unit_tests_cleaned"],
            outputs="reporting_tests_cleaned",
            name="unit_test_cleaned_data_node",
        ),
        node(
            func=unit_test_model_input,
            inputs=["X_train_data", "X_val_data", "params:data_unit_tests_model_input"],
            outputs="reporting_tests_model_input",
            name="unit_test_model_input_node",
        ),
        node(
            func = write_traffic_light, 
            inputs = ["reporting_tests_raw", "params:data_unit_tests"],
            outputs = "flag_raw", name="traffic_light_raw",
        ),
        node(
            func = write_traffic_light,
            inputs = ["reporting_tests_cleaned", "params:data_unit_tests_cleaned"],
            outputs = "flag_cleaned", name="traffic_light_cleaned",
        ),
        node(
            func = write_traffic_light,
            inputs = ["reporting_tests_model_input", "params:data_unit_tests_model_input"],
            outputs = "flag_model_input", name="traffic_light_model_input",
        ),
    ])