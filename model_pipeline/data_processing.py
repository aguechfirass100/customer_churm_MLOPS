import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def prepare_data(data_path="Churn_Modelling.csv"):
    """Loads and preprocesses data for training and testing."""
    encoder = LabelEncoder()
    data = pd.read_csv(data_path)
    data = data.drop(["Surname", "Geography"], axis=1)
    data["Gender"] = encoder.fit_transform(data["Gender"])
    data = data.dropna()

    X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = data["Exited"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, "scaler.joblib")
    return x_train_scaled, x_test_scaled, y_train, y_test
