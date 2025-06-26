from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os
from scripts.preprocessing import preprocess_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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

# Load artifacts
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.joblib"))

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        processed = preprocess_input(input_df, selected_features, scaler)
        prediction = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0]
        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": {"Yes": float(prob[1]), "No": float(prob[0])}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
