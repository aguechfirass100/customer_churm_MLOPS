import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import mlflow
from model_pipeline import evaluate_model, prepare_data, save_model, train_model

print("Loading the XGBoost model...")
try:
    model = joblib.load("xgboost_model.joblib")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

app = FastAPI()


class PredictionRequest(BaseModel):
    features: list


class RetrainRequest(BaseModel):
    n_estimators: int
    max_depth: int
    min_samples_split: int


@app.post("/predict")
async def predict(request: PredictionRequest):
    print("Predict endpoint called.")
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed")


@app.post("/retrain")
async def retrain(request: RetrainRequest):
    print("Retrain endpoint called.")
    if mlflow.active_run():
        mlflow.end_run()
    try:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train,
                            n_estimators=request.n_estimators,
                            max_depth=request.max_depth,
                            min_samples_split=request.min_samples_split)
        accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)
        save_model(model)
        return {
            "message": "Model re-trained successfully",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    except Exception as e:
        print(f"Retraining failed: {e}")
        raise HTTPException(
            status_code=400, detail=f"Retraining failed: {str(e)}")
