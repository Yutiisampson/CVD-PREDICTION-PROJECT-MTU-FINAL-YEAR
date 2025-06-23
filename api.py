from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import logging
import os
from preprocessing import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "https://cvdprediction-mtu.streamlit.app"),
        "http://localhost:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input validation
class PredictionInput(BaseModel):
    General_Health: str
    Checkup: str
    Exercise: str
    Skin_Cancer: str
    Other_Cancer: str
    Depression: str
    Diabetes: str
    Arthritis: str
    Gender: str
    Age_Category: str
    Height_cm: float
    Weight_kg: float
    BMI: float
    Smoking_History: str
    Alcohol_Consumption: int
    Fruit_Consumption: int
    Green_Vegetables_Consumption: int
    FriedPotato_Consumption: int

# Load model and scaler
try:
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    selected_features = joblib.load("selected_features.joblib")
    logger.info("Model, scaler, and selected features loaded successfully")
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    raise RuntimeError(f"Failed to load model or scaler: {e}")

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput) -> Dict[str, Any]:
    logger.info(f"Received prediction request with input: {data.dict()}")
    try:
        input_dict = data.dict()
        processed_input = preprocess_input(input_dict, selected_features, scaler, is_training=False)
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]
        logger.info(f"Prediction: {prediction}, Probabilities: {prediction_proba.tolist()}")
        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": {
                "Yes": float(prediction_proba[1]),
                "No": float(prediction_proba[0]),
            }
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
