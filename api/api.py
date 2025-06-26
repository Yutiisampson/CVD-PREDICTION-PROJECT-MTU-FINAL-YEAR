from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from ..scripts.preprocessing import preprocess_input

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cvdprediction-mtu.streamlit.app", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


try:
    random_forest_model = joblib.load("../models/random_forest_model.joblib")
    scaler = joblib.load("../models/scaler.joblib")
    selected_features = joblib.load("../models/selected_features.joblib")
    print("Selected features:", selected_features)
    print("Feature importances:", dict(zip(selected_features, random_forest_model.feature_importances_)))
except FileNotFoundError as e:
    raise RuntimeError(f"Failed to load model or artifacts: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput) -> Dict[str, Any]:
    try:
        input_dict = data.dict()
        print("Received input:", input_dict)
        processed_input = preprocess_input(input_dict, selected_features, scaler, is_training=False)
        print("Processed input columns:", processed_input.columns.tolist())
        print("Raw processed values:", processed_input.values)
        print("Scaled input values:", processed_input.to_numpy())
        processed_input_array = processed_input.to_numpy()
        prediction = random_forest_model.predict(processed_input_array)[0]
        prediction_proba = random_forest_model.predict_proba(processed_input_array)[0]
        response = {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": {
                "Yes": float(prediction_proba[1]),
                "No": float(prediction_proba[0]),
            }
        }
        print("Prediction response:", response)
        return response
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
