"""Pipeline `reporting`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_report


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de reporting. outputs: `final_report`."""
    return pipeline(
        [
            node(
                func=build_report,
                inputs=[
                    "production_model_metrics",
                    "shap_plot",
                    "drift_result",
                    "reporting_tests",
                    "params:reporting",
                ],
                outputs="final_report",
                name="build_report_node",
            ),
        ]
    )
