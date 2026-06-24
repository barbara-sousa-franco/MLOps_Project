"""Pipeline `model_predict`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    """Create the inference pipeline. outputs: `predictions`, `production_test_metrics`.

    The champion uses all features (no best_columns needed); it selects its own
    `feature_names_in_` from the batch. `parameters` carries target_col + use_log_target.
    """
    return pipeline(
        [
            node(
                func=predict,
                # pass only the small params needed (not the whole dict — mlflow logs it)
                inputs=["production_model", "preprocessed_batch_data",
                        "params:target_col", "params:use_log_target"],
                outputs=["predictions", "production_test_metrics"],
                name="predict_node",
            ),
        ]
    )
