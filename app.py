from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import mlflow
import numpy as np
import os
from motor.motor_asyncio import AsyncIOMotorClient
from model_pipeline import evaluate_model, prepare_data, save_model, train_model

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

client = AsyncIOMotorClient(MONGO_URI)

db = client["TCCPM"]
predictions_collection = db.predictions


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
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)

    try:
        prediction = model.predict(features)

        # Save prediction to MongoDB
        prediction_data = {"prediction": int(prediction[0])}
        result = await predictions_collection.insert_one(prediction_data)

        # Return the inserted ID
        return {"prediction": int(prediction[0]), "prediction_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Prediction failed")


@app.post("/retrain")
async def retrain(request: RetrainRequest):
    if mlflow.active_run():
        mlflow.end_run()

    try:
        print("Retraining the model")
        x_train, x_test, y_train, y_test = prepare_data()

        model = train_model(x_train, y_train,
                            n_estimators=request.n_estimators,
                            max_depth=request.max_depth,
                            min_samples_split=request.min_samples_split)

        accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)

        print("Saving the retrained model")
        save_model(model)

        # Save retraining metrics to MongoDB
        retraining_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        result = await predictions_collection.insert_one(retraining_metrics)

        return {"message": "Model re-trained successfully",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                # Return the inserted ID
                "retraining_id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Retraining failed: {str(e)}")


# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import joblib
# import mlflow
# import numpy as np

# from model_pipeline import evaluate_model, prepare_data, save_model, train_model

# model = joblib.load("xgboost_model.joblib")

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class PredictionRequest(BaseModel):
#     features: list


# class RetrainRequest(BaseModel):
#     n_estimators: int
#     max_depth: int
#     min_samples_split: int


# @app.post("/predict")
# def predict(request: PredictionRequest):
#     features = np.array(request.features).reshape(1, -1)

#     try:
#         prediction = model.predict(features)
#         return {"prediction": int(prediction[0])}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Prediction failed")


# @app.post("/retrain")
# def retrain(request: RetrainRequest):

#     if mlflow.active_run():
#         mlflow.end_run()

#     try:
#         print("Retraining the model")
#         x_train, x_test, y_train, y_test = prepare_data()

#         model = train_model(x_train, y_train,
#                             n_estimators=request.n_estimators,
#                             max_depth=request.max_depth,
#                             min_samples_split=request.min_samples_split)

#         accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)

#         print("Saving the retrained model")
#         save_model(model)

#         return {"message": "Model re-trained successfully",
#                 "accuracy": accuracy,
#                 "precision": precision,
#                 "recall": recall,
#                 "f1": f1}
#     except Exception as e:
#         raise HTTPException(
#             status_code=400, detail=f"Retraining failed: {str(e)}")
