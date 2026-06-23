"""Pipeline `model_selection`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model-selection pipeline. outputs: `selected_model`.

    Note: `champion_dict`/`champion_model` (state from previous runs) are optional node
    parameters and are NOT wired here — load them from the registry/artifact inside the
    node when they exist. Keeps the graph acyclic (model_selection -> model_train).
    """
    return pipeline(
        [
            node(
                func=model_selection,
                # X_*_scaled = final 05_model_input layer (impute->cap->encode->scale);
                # X_val_* = leak-free VALIDATION set; target = log-y from split_train
                inputs=[
                    "X_train_scaled",
                    "X_val_scaled",
                    "y_train_data",
                    "y_val_data",
                    "params:model_selection",
                    "best_columns",
                ],
                outputs="selected_model",
                name="model_selection_node",
            ),
        ]
    )
