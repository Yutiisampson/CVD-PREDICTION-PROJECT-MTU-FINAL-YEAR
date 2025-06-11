from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from preprocessing import preprocess_input

app = FastAPI()

# Load saved objects
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    selected_features = joblib.load('selected_features.joblib')
except FileNotFoundError as e:
    raise Exception(f"Missing file: {e}")

# Input model
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

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        input_processed = preprocess_input(data.dict(), selected_features, scaler)
        prediction = model.predict(input_processed)
        prediction_proba = model.predict_proba(input_processed)[0]
        prediction_label = label_encoder.inverse_transform(prediction)[0]
        return {
            "prediction": prediction_label,
            "probability": {
                "No": float(prediction_proba[0]),
                "Yes": float(prediction_proba[1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")