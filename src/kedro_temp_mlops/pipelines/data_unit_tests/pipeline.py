"""Pipeline for `data_unit_tests`.

Two validation nodes:
  1. unit_test_cleaned_data  — validates cleaned_data (post-preprocessing, pre-split)
  2. unit_test_model_input   — validates X_train (ready for modelling)

Both feed into write_traffic_light which gates downstream pipelines.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import unit_test_cleaned_data, unit_test_model_input, write_traffic_light


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
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
                    func=write_traffic_light,
                    inputs=["reporting_tests_cleaned", "params:data_unit_tests_cleaned"],
                    outputs="traffic_light_flag",
                    name="write_traffic_light_node",
                ),
    ])