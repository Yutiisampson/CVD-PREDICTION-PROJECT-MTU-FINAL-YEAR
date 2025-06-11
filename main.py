from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

from preprocessing import preprocess_input 


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cvdprediction-mtu.streamlit.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InputData(BaseModel):
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

random_forest_model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")
selected_features = joblib.load("selected_features.joblib")
label_encoder  = joblib.load("label_encoder.joblib")

# --- Health check endpoint ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PredictionInput) -> Dict[str, Any]:
    try:
        input_dict = data.dict()
        processed_input = preprocess_input(input_dict, selected_features, scaler, is_training=False)
        
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": {
                "Yes": float(prediction_proba[1]),
                "No": float(prediction_proba[0]),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

