"""Nodes da pipeline `model_selection`.

Adapta o padrão do prof para REGRESSÃO + Optuna (em vez de GridSearchCV).
Métrica de regressão (RMSE menor = melhor!), nunca accuracy.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name: str) -> str:
    """Devolve o id do experimento MLflow, criando-o se não existir (igual ao prof).

    TODO _get_or_create_experiment_id:
      - mlflow.get_experiment_by_name(name); se None -> mlflow.create_experiment(name)
      - devolver experiment_id
    """
    # TODO: implementar
    raise NotImplementedError


def model_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: dict,
    champion_dict: dict | None = None,
    champion_model=None,
):
    """Compara challengers, afina com Optuna e compara com o champion atual.

    Args:
        X_train, X_test, y_train, y_test: dados do split.
        parameters: candidatos + espaços Optuna (parameters_model_selection.yml).
        champion_dict: métricas do champion atual (estado de um run anterior; None na 1ª
            execução). Opcional e NÃO cabeado no grafo para evitar ciclo — carregar
            internamente do registry/artifact quando existir.
        champion_model: modelo champion atual (idem).

    Returns:
        selected_model — o melhor modelo (challenger afinado OU champion existente),
        entregue ao `model_train`.

    TODO model_selection:
      PASSO 1 — comparar tipos de modelo (challengers):
        - candidatos: RandomForestRegressor, GradientBoostingRegressor,
          (XGBoost/LightGBM opcional)
        - mlflow.sklearn.autolog; treinar cada um; métrica = RMSE ou R² no test
        - escolher melhor tipo
      PASSO 2 — tuning com OPTUNA:
        - def objective(trial): sugerir hiperparâmetros do espaço em parameters
        - optuna.create_study(direction="minimize" p/ RMSE), n_trials de parameters
        - cada trial num mlflow nested run
      PASSO 3 — comparar com champion atual:
        - se novo score MELHOR que champion_dict (RMSE menor é melhor!) -> devolver novo
        - senão -> devolver champion existente
      - devolver selected_model
    """
    # TODO: implementar
    raise NotImplementedError
