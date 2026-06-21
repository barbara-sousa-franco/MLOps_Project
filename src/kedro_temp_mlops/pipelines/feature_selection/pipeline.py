"""Pipeline `feature_selection` — SHAP selection + Optuna challenger + promotion."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_shap, select_features, tune_challenger, compare_and_promote


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=compute_shap,
            inputs=["production_model", "X_train_encoded", "params:feature_selection"],
            outputs=["shap_values", "shap_plot"],
            name="compute_shap_node",
        ),
        node(
            func=select_features,
            inputs=["shap_values", "X_train_encoded", "params:feature_selection"],
            outputs="best_columns",
            name="select_features_node",
        ),
        node(
            func=tune_challenger,
            inputs=["X_train_encoded", "y_train_data", "best_columns", "params:feature_selection"],
            outputs=["challenger_model", "challenger_params"],
            name="tune_challenger_node",
        ),
        node(
            func=compare_and_promote,
            inputs=["production_model", "challenger_model", "challenger_params",
                    "X_test_encoded", "y_test_data", "best_columns", "params:feature_selection"],
            outputs=["promoted_model", "champion_challenger_metrics", "champion_challenger_report"],
            name="compare_and_promote_node",
        ),
    ])
