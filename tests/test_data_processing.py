import pytest
import numpy as np
from model_pipeline.data_processing import prepare_data


def test_prepare_data():
    x_train, x_test, y_train, y_test = prepare_data()

    assert x_train.shape[0] > 0
    assert x_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

    assert np.isclose(x_train.mean(), 0, atol=1)
    assert np.isclose(x_train.std(), 1, atol=1)
