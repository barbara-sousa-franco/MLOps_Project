"""Nodes da pipeline `split_data`.

⚠️ BUG DO EXEMPLO A NÃO COPIAR: `stratify=y` só serve classificação; em regressão
rebenta. Ver TODO abaixo.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_data(preprocessed_training_data: pd.DataFrame, parameters: dict):
    """Separa features/target e faz o train/test split (regressão).

    Args:
        preprocessed_training_data: saída de `preprocessing_train`.
        parameters: test_size, seed, estratégia (parameters_split.yml).

    Returns:
        Tuple (X_train, X_test, y_train, y_test, columns):
          - columns: X_train.columns (mapeia a best_columns no catalog, como o exemplo).

    TODO split_data:
      - assert sem nulls (como o exemplo)
      - separar target (Price) das features; drop da coluna "index"
      - train_test_split com seed e test_size de parameters
      - ⚠️ NÃO usar stratify=y (bug do exemplo — só serve classificação).
        Regressão: ou sem stratify, OU criar bins de Price (pd.qcut) e stratify por
        bin para garantir que o test cobre toda a gama de preços (recomendado).
      - devolver também X_train.columns
      - test set é sagrado: não tocar até avaliação final
    """
    # TODO: implementar
    raise NotImplementedError
