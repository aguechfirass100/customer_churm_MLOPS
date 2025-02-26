from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

from model_pipeline import evaluate_model, prepare_data, save_model, train_model

model = joblib.load("xgboost_model.joblib")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    features: list
    
class RetrainRequest(BaseModel):
    n_estimators: int
    max_depth: int
    min_samples_split: int

@app.post("/predict")
def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)

    try:
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Prediction failed")


@app.post("/retrain")
def retrain(request: RetrainRequest):
    try:
        x_train, x_test, y_train, y_test = prepare_data()

        model = train_model(x_train, y_train, 
                            n_estimators=request.n_estimators, 
                            max_depth=request.max_depth, 
                            min_samples_split=request.min_samples_split)

        accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)

        save_model(model)

        return {"message": "Model re-trained successfully", 
                "accuracy": accuracy, 
                "precision": precision, 
                "recall": recall, 
                "f1": f1}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Retraining failed: {str(e)}")