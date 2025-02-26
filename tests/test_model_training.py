import pytest
import numpy as np
from model_pipeline.model_training import (train_model)


def test_train_model():
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)

    model = train_model(x_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")

    preds = model.predict(x_train[:5])
    assert len(preds) == 5
