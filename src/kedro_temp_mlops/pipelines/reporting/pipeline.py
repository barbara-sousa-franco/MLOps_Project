"""Pipeline `reporting`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_report


def create_pipeline(**kwargs) -> Pipeline:
    """Consolidate metrics + data quality + drift + SHAP into a Markdown report.

    Runs last (needs the outputs of training, inference, monitoring and data_prep).
    `shap_importance` (from explainability) is read optionally inside the node.
    outputs: `final_report`.
    """
    return pipeline(
        [
            node(
                func=build_report,
                inputs=[
                    "production_model",
                    "production_model_metrics",
                    "production_test_metrics",
                    "drift_result",
                    "reporting_tests_cleaned",
                    "reporting_tests_model_input",
                    "params:reporting",
                ],
                outputs="final_report",
                name="build_report_node",
            ),
        ]
    )
