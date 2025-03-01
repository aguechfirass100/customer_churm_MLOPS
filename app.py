# from dotenv import load_dotenv
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import mlflow
import numpy as np
import os
from motor.motor_asyncio import AsyncIOMotorClient
from model_pipeline import evaluate_model, prepare_data, save_model, train_model

# load_dotenv()

# MONGO_URI = os.getenv("MONGO_URI")

MONGO_URI = "mongodb+srv://aguechfirass100:RBfGTcdnZ3Q8JASy@tccpm.xmctk.mongodb.net/?retryWrites=true&w=majority&appName=TCCPM"

print("Attempting to connect to MongoDB...")
client = AsyncIOMotorClient(MONGO_URI)

db = client["TCCPM"]
predictions_collection = db.predictions


async def test_connection():
    try:
        print("Testing MongoDB connection...")
        await client.server_info()
        print("MongoDB connection successful!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")

# asyncio.run(test_connection())

print("Loading the XGBoost model...")
try:
    model = joblib.load("xgboost_model.joblib")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

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


@app.get("/")
def read_root():
    print("Root endpoint called.")
    return {"message": "Hello, World!"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    print("Predict endpoint called.")
    print(f"Received features: {request.features}")

    try:
        features = np.array(request.features).reshape(1, -1)
        print(f"Reshaped features: {features}")

        print("Making prediction...")
        prediction = model.predict(features)
        print(f"Prediction result: {prediction}")

        prediction_data = {"prediction": int(prediction[0])}
        print(f"Prediction data to insert: {prediction_data}")

        print("Inserting prediction into MongoDB...")
        result = await predictions_collection.insert_one(prediction_data)
        print(f"Insertion result: {result.inserted_id}")

        return {"prediction": int(prediction[0]), "prediction_id": str(result.inserted_id)}
    except Exception as e:
        print(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail="Prediction failed")


@app.post("/retrain")
async def retrain(request: RetrainRequest):
    print("Retrain endpoint called.")
    print(
        f"Retrain request parameters: n_estimators={request.n_estimators}, max_depth={request.max_depth}, min_samples_split={request.min_samples_split}")

    if mlflow.active_run():
        print("Ending existing MLflow run...")
        mlflow.end_run()

    try:
        print("Preparing data for retraining...")
        x_train, x_test, y_train, y_test = prepare_data()
        print(
            f"Data prepared: x_train={x_train.shape}, x_test={x_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

        print("Training the model...")
        model = train_model(x_train, y_train,
                            n_estimators=request.n_estimators,
                            max_depth=request.max_depth,
                            min_samples_split=request.min_samples_split)
        print("Model training completed.")

        print("Evaluating the model...")
        accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)
        print(
            f"Evaluation metrics: accuracy={accuracy}, precision={precision}, recall={recall}, f1={f1}")

        print("Saving the retrained model...")
        save_model(model)
        print("Model saved successfully.")

        retraining_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        print(f"Retraining metrics to insert: {retraining_metrics}")

        print("Inserting retraining metrics into MongoDB...")
        result = await predictions_collection.insert_one(retraining_metrics)
        print(f"Insertion result: {result.inserted_id}")

        return {"message": "Model re-trained successfully",
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "retraining_id": str(result.inserted_id)}
    except Exception as e:
        print(f"Retraining failed: {e}")
        raise HTTPException(
            status_code=400, detail=f"Retraining failed: {str(e)}")


@app.get("/health")
def health_check():
    print("Health check endpoint called.")
    return {"status": "ok"}


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
