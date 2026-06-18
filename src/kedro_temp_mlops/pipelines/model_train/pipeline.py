"""Pipeline `model_train`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train, register_model


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de treino do champion + registo no Model Registry.

    outputs: `production_model`, `production_columns`, `production_model_metrics`.
    """
    return pipeline(
        [
            node(
                func=model_train,
                # `best_columns` é opcional e NÃO é cabeado no 1º passe (evita ciclo com
                # feature_selection). 2º passe: adicionar "best_columns" aqui + pôr
                # use_feature_selection=true em parameters_model_train.yml.
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "params:model_train",
                    "selected_model",
                ],
                outputs=[
                    "production_model",
                    "production_columns",
                    "production_model_metrics",
                ],
                name="model_train_node",
            ),
            node(
                func=register_model,
                inputs=[
                    "production_model",
                    "production_model_metrics",
                    "params:model_train",
                ],
                outputs="registered_model_version",
                name="register_model_node",
            ),
        ]
    )
