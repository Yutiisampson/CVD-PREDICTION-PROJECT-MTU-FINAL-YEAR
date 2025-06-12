from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from preprocessing import preprocess_input

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cvdprediction-mtu.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Load models
required_files = {
    "model": "random_forest_model.joblib",
    "scaler": "scaler.joblib",
    "label_encoder": "label_encoder.joblib",
    "selected_features": "selected_features.joblib"
}

loaded_objects = {}
missing = []

for key, path in required_files.items():
    if os.path.exists(path):
        loaded_objects[key] = joblib.load(path)
    else:
        missing.append(path)

if missing:
    raise RuntimeError(f"Missing files: {', '.join(missing)}")

model = loaded_objects["model"]
scaler = loaded_objects["scaler"]
label_encoder = loaded_objects["label_encoder"]
selected_features = loaded_objects["selected_features"]

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

# Prediction endpoint
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
