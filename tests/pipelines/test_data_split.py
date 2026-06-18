"""Testes pytest do split_data."""

import pytest


def test_split_shapes():
    """X_train/X_test/y_train/y_test com shapes coerentes e test_size respeitado.

    TODO: chamar split_data e validar proporções.
    """
    pytest.skip("TODO: implementar quando split_data estiver pronto")


def test_no_overlap_train_test():
    """Não pode haver overlap de índices entre train e test.

    TODO: assert set(X_train.index) & set(X_test.index) == set()
    """
    pytest.skip("TODO: implementar quando split_data estiver pronto")


def test_split_is_reproducible():
    """Mesma seed -> mesmo split.

    TODO: correr split_data duas vezes com a mesma seed e comparar índices.
    """
    pytest.skip("TODO: implementar quando split_data estiver pronto")
