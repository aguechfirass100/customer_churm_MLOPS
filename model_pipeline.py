# import pandas as pd
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib
# import mlflow
# import mlflow.sklearn

# mlflow.set_experiment("Churn_Model_Experiment_XGBoost")


# def prepare_data(data_path="Churn_Modelling.csv"):
#     encoder = LabelEncoder()
#     data = pd.read_csv(data_path)
#     data = data.drop(["Surname", "Geography"], axis=1)
#     data["Gender"] = encoder.fit_transform(data["Gender"])
#     data = data.dropna()
#     X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
#     y = data["Exited"]
#     x_train, x_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=1
#     )
#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_test_scaled = scaler.transform(x_test)
#     joblib.dump(scaler, "scaler.joblib")
#     return x_train_scaled, x_test_scaled, y_train, y_test


# def train_model(x_train, y_train, n_estimators=100, max_depth=3, learning_rate=0.1):
#     with mlflow.start_run():
#         # Log XGBoost hyperparameters
#         mlflow.log_param("n_estimators", n_estimators)
#         mlflow.log_param("max_depth", max_depth)
#         mlflow.log_param("learning_rate", learning_rate)

#         print("Training features (x_train):")
#         print(x_train[:5])
#         print("Training labels (y_train):")
#         print(y_train[:5])

#         # Initialize XGBoost classifier
#         model = XGBClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             learning_rate=learning_rate,
#             random_state=1,
#             use_label_encoder=False,  # To avoid warnings
#             eval_metric='logloss'     # Recommended for binary classification
#         )
#         model.fit(x_train, y_train)

#         # Log model
#         mlflow.sklearn.log_model(model, "xgboost_model")
#         return model


# def evaluate_model(model, x_test, y_test):
#     print("Test features (x_test):")
#     print(x_test[:5])
#     print("Test labels (y_test):")
#     print(y_test[:5])

#     y_pred = model.predict(x_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)
#     mlflow.log_metric("f1", f1)

#     print(
#         f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
#     return accuracy, precision, recall, f1


# def save_model(model, model_path="xgboost_model.joblib"):  # Updated default path
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")


# def load_model(model_path="xgboost_model.joblib"):  # Updated default path
#     model = joblib.load(model_path)
#     print(f"Model loaded from {model_path}")
#     return model


# --------------------***********************------------------------


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib
# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# mlflow.set_experiment("Churn_Model_Experiment")


# def prepare_data(data_path="Churn_Modelling.csv"):
#     encoder = LabelEncoder()
#     data = pd.read_csv(data_path)
#     data = data.drop(["Surname", "Geography"], axis=1)
#     data["Gender"] = encoder.fit_transform(data["Gender"])
#     data = data.dropna()
#     X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
#     y = data["Exited"]
#     x_train, x_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=1
#     )
#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_test_scaled = scaler.transform(x_test)
#     joblib.dump(scaler, "scaler.joblib")
#     return x_train_scaled, x_test_scaled, y_train, y_test


# def train_model(x_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2):
#     with mlflow.start_run():

#         mlflow.log_param("n_estimators", n_estimators)
#         mlflow.log_param("max_depth", max_depth)
#         mlflow.log_param("min_samples_split", min_samples_split)

#         print("Training features (x_train):")
#         print(x_train[:5])
#         print("Training labels (y_train):")
#         print(y_train[:5])

#         model = RandomForestClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             random_state=1
#         )
#         model.fit(x_train, y_train)

#         mlflow.sklearn.log_model(model, "random_forest_model")

#         return model


# def evaluate_model(model, x_test, y_test):

#     print("Test features (x_test):")
#     print(x_test[:5])
#     print("Test labels (y_test):")
#     print(y_test[:5])

#     y_pred = model.predict(x_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_metric("precision", precision)
#     mlflow.log_metric("recall", recall)
#     mlflow.log_metric("f1", f1)

#     print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
#     return accuracy, precision, recall, f1


# def save_model(model, model_path="model.joblib"):
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")


# def load_model(model_path="model.joblib"):
#     model = joblib.load(model_path)
#     print(f"Model loaded from {model_path}")
#     return model


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import joblib


# def prepare_data(data_path="Churn_Modelling.csv"):
#     encoder = LabelEncoder()
#     data = pd.read_csv(data_path)
#     data = data.drop(["Surname", "Geography"], axis=1)
#     data["Gender"] = encoder.fit_transform(data["Gender"])
#     data = data.dropna()
#     X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
#     y = data["Exited"]
#     x_train, x_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=1
#     )
#     scaler = StandardScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_test_scaled = scaler.transform(x_test)
#     joblib.dump(scaler, "scaler.joblib")
#     return x_train_scaled, x_test_scaled, y_train, y_test


# def train_model(x_train, y_train):
#     model = RandomForestClassifier(n_estimators=100, random_state=1)
#     model.fit(x_train, y_train)
#     return model


# def evaluate_model(model, x_test, y_test):
#     y_pred = model.predict(x_test)
#     (y_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
#     return accuracy


# def save_model(model, model_path="model.joblib"):
#     joblib.dump(model, model_path)
#     print(f"Model saved to {model_path}")


# def load_model(model_path="model.joblib"):
#     model = joblib.load(model_path)
#     print(f"Model loaded from {model_path}")
#     return model
