from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow


def evaluate_model(model, x_test, y_test):
    """Evaluates the model and logs metrics using MLflow."""
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    print(
        f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    return accuracy, precision, recall, f1
