import mlflow
import mlflow.sklearn
import joblib
from xgboost import XGBClassifier
import os

experiment_name = "Churn_Model_Experiment_XGBoost"

if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)

mlflow.set_experiment(experiment_name)


def train_model(x_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_split=2):
    """Trains the XGBoost model and logs parameters with MLflow."""
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=1,
            use_label_encoder=False,
            eval_metric='logloss',
            min_samples_split=min_samples_split
        )
        model.fit(x_train, y_train)
        print("X_train shape", x_train.shape)
        print("y_train shape", y_train.shape)

        mlflow.sklearn.log_model(model, "xgboost_model")
        return model


def save_model(model, model_path="xgboost_model.joblib"):
    """Saves the trained model."""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path="xgboost_model.joblib"):
    """Loads a saved model."""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


# import mlflow
# import mlflow.sklearn
# import joblib
# from xgboost import XGBClassifier

# mlflow.set_experiment("Churn_Model_Experiment_XGBoost")


# def train_model(x_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1):
#     """Trains the XGBoost model and logs parameters with MLflow."""
#     with mlflow.start_run():
#         mlflow.log_param("n_estimators", n_estimators)
#         mlflow.log_param("max_depth", max_depth)
#         mlflow.log_param("learning_rate", learning_rate)

#         model = XGBClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             random_state=1,
#             use_label_encoder=False,
#             eval_metric='logloss'
#         )
#         model.fit(x_train, y_train)

#         mlflow.sklearn.log_model(model, "xgboost_model")
#         return model


# def save_model(model, model_path="xgboost_model.joblib"):
#     """Saves the trained model."""
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")


# def load_model(model_path="xgboost_model.joblib"):
#     """Loads a saved model."""
#     model = joblib.load(model_path)
#     print(f"Model loaded from {model_path}")
#     return model
