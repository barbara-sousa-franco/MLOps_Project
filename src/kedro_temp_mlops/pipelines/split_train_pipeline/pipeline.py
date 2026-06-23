"""
This is a boilerplate pipeline 'split_train_pipeline'
generated using Kedro 1.3.1
"""

"""Pipeline `split_train` — train/test split of cleaned data."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_train,
            inputs=["cleaned_data", "params:split_train"],
            outputs=["X_train_data", "X_val_data", "y_train_data", "y_val_data", "all_columns"],
            name="split_train_node",
        ),
    ])
