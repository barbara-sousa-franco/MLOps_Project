"""Pipeline `data_unit_tests`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import unit_test, write_traffic_light


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de data unit tests.

    Node(unit_test) -> Node(write_traffic_light) -> outputs `reporting_tests`.
    """
    return pipeline(
        [
            node(
                func=unit_test,
                inputs=["ingested_data", "params:data_unit_tests"],
                outputs="reporting_tests",
                name="unit_test_node",
            ),
            node(
                func=write_traffic_light,
                inputs=["reporting_tests", "params:data_unit_tests"],
                outputs="traffic_light_flag",
                name="write_traffic_light_node",
            ),
        ]
    )
