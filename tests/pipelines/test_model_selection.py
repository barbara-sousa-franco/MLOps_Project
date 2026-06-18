"""Testes pytest do model_selection."""

import pytest


def test_returns_trained_model_and_valid_metric():
    """model_selection devolve um modelo treinado e uma métrica válida.

    TODO:
      - correr model_selection na sample (poucos n_trials)
      - assert que o modelo tem .predict e que RMSE/R² é finito e no domínio esperado
    """
    pytest.skip("TODO: implementar quando model_selection estiver pronto")
