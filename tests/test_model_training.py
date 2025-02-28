import pytest
import mlflow
import numpy as np
from model_pipeline.model_training import train_model


def test_train_model():

    if mlflow.active_run():
        mlflow.end_run()

    # Generate synthetic training data
    x_train = np.random.rand(100, 9)  # 100 samples, 9 features
    y_train = np.random.randint(0, 2, 100)  # Binary target (0 or 1)

    # Train the model
    model = train_model(x_train, y_train)

    # Ensure the model was returned and has the 'predict' method
    assert model is not None
    assert hasattr(model, "predict")

    # Test input with the same number of features as x_train (9 features)
    test_input = np.array([[-0.23082038, 0.91509065, -0.94449979, -0.70174202,
                            0.58817274, 0.80225696, -1.55337352, 0.97725852, 0.42739449]])

    # Make predictions
    preds = model.predict(test_input)

    # Assert that the prediction is for one sample
    assert len(preds) == 1  # Ensure we're predicting one value
    assert preds[0] in [0, 1]  # Ensure prediction is a valid class (0 or 1)
