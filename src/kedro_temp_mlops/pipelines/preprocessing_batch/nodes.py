"""Nodes da pipeline `preprocessing_batch`.

Reutiliza o `encoder_transform` do treino: aplica `transform` (NUNCA `fit`) ao batch
novo. Output alimenta drift e predict.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_batch(ana_data: pd.DataFrame, encoder, parameters: dict) -> pd.DataFrame:
    """Pré-processa um batch novo com o encoder já treinado.

    Args:
        ana_data: batch novo a analisar (02_intermediate/ana_data.csv).
        encoder: `encoder_transform` ajustado no treino.
        parameters: mesma config de limpeza/encoding do treino (parameters_preprocessing.yml).

    Returns:
        preprocessed_batch_data — pronto para drift e inferência.

    TODO preprocess_batch:
      - aplicar a MESMA limpeza/FE do treino (sem refit)
      - encoder.transform(...) — NÃO encoder.fit(...)  (anti-leakage)
      - NÃO aplicar log1p ao target se o batch não tiver target (inferência)
      - output preprocessed_batch_data
    """
    # TODO: implementar
    raise NotImplementedError
