"""Pipeline `model_selection`."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de model selection. outputs: `selected_model`.

    Nota: `champion_dict`/`champion_model` (estado de runs anteriores) são parâmetros
    opcionais do nó e NÃO são cabeados aqui — carregar do registry/artifact dentro do nó
    quando existirem. Mantém o grafo acíclico (model_selection -> model_train).
    """
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "params:model_selection",
                ],
                outputs="selected_model",
                name="model_selection_node",
            ),
        ]
    )
