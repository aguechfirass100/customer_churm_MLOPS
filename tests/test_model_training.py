import pytest
import mlflow
import numpy as np
from model_pipeline.model_training import train_model


def test_train_model():

    if mlflow.active_run():
        mlflow.end_run()

    x_train = np.random.rand(100, 9)
    y_train = np.random.randint(0, 2, 100)

    model = train_model(x_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")

    test_input = np.array(
        [
            [
                -0.23082038,
                0.91509065,
                -0.94449979,
                -0.70174202,
                0.58817274,
                0.80225696,
                -1.55337352,
                0.97725852,
                0.42739449,
            ]
        ]
    )

    preds = model.predict(test_input)

    assert len(preds) == 1
    assert preds[0] in [0, 1]
