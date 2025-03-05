import pytest
import numpy as np
from model_pipeline.model_training import train_model
from model_pipeline.model_evaluation import evaluate_model


def test_evaluate_model():
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    x_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)

    model = train_model(x_train, y_train)
    accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)

    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1
