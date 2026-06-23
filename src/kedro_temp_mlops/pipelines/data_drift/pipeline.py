"""Pipeline `data_drift`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_drift


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de drift. outputs: `drift_result`, `drift_report`."""
    return pipeline(
        [
            node(
                func=compute_drift,
                inputs=["learning_data", "test_data", "params:data_drift"],
                outputs=["drift_result", "drift_report"],
                name="compute_drift_node",
            ),
        ]
    )
