"""Pipeline `split_data`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de split. outputs: X_train, X_test, y_train, y_test, columns."""
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_training_data", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test", "all_columns"],
                name="split_data_node",
            ),
        ]
    )
