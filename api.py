from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cvdprediction-mtu.streamlit.app", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Manual encoding mappings
GENERAL_HEALTH_MAPPING = {
    'Poor': [1, 0, 0, 0],
    'Fair': [0, 1, 0, 0],
    'Good': [0, 0, 1, 0],
    'Very Good': [0, 0, 0, 1],
    'Excellent': [0, 0, 0, 0]  # Reference category
}

CHECKUP_MAPPING = {
    'In the past year': [1, 0, 0, 0],
    'Within the last 2 years': [0, 1, 0, 0],
    'In the last 5 years': [0, 0, 1, 0],
    '5 years or longer ago': [0, 0, 0, 1],
    'Never': [0, 0, 0, 0]  # Reference category
}

GENDER_MAPPING = {
    'Male': [1],
    'Female': [0]  # Reference category
}

AGE_CATEGORY_MAPPING = {
    'youth': [1, 0, 0, 0],
    'young_adults': [0, 1, 0, 0],
    'adults': [0, 0, 1, 0],
    'middle_aged': [0, 0, 0, 1],
    'old': [0, 0, 0, 0]  # Reference category
}

# Preprocessing function with manual encoding
def preprocess_input(data, selected_features, scaler, is_training=False):
    cardio = pd.DataFrame([data] if isinstance(data, dict) else data)

    # Input validation
    required_cols = [
        'General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer',
        'Depression', 'Diabetes', 'Arthritis', 'Gender', 'Age_Category',
        'Height_cm', 'Weight_kg', 'BMI', 'Smoking_History', 'Alcohol_Consumption',
        'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption'
    ]
    missing_cols = [col for col in required_cols if col not in cardio.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Binary encoding
    binary_cols = ['Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']
    for col in binary_cols:
        if not cardio[col].isin(['Yes', 'No']).all():
            raise ValueError(f"Invalid values in {col}: must be 'Yes' or 'No'")
        cardio[col] = cardio[col].map({'No': 0, 'Yes': 1})

    # Diabetes simplified
    diabetes_valid = [
        'No', 'Yes', 'No, pre-diabetes or borderline diabetes',
        'Yes, but female told only during pregnancy'
    ]
    if not cardio['Diabetes'].isin(diabetes_valid).all():
        raise ValueError(f"Invalid values in Diabetes: must be one of {diabetes_valid}")
    cardio['Diabetes'] = cardio['Diabetes'].map({
        'No': 0,
        'Yes': 1,
        'No, pre-diabetes or borderline diabetes': 0,
        'Yes, but female told only during pregnancy': 1
    })

    # Checkup normalization
    checkup_mapping = {
        'Within the past year': 'In the past year',
        'Within the past 2 years': 'Within the last 2 years',
        'Within the past 5 years': 'In the last 5 years',
        '5 or more years ago': '5 years or longer ago',
        'Never': 'Never'
    }
    cardio['Checkup'] = cardio['Checkup'].replace(checkup_mapping)
    if not cardio['Checkup'].isin(CHECKUP_MAPPING.keys()).all():
        raise ValueError(f"Invalid values in Checkup: must be one of {list(CHECKUP_MAPPING.keys())}")

    # Age category grouping
    age_mapping = {
        '18-24': 'youth', '25-29': 'youth',
        '30-34': 'young_adults', '35-39': 'young_adults',
        '40-44': 'adults', '45-49': 'adults',
        '50-54': 'middle_aged', '55-59': 'middle_aged',
        '60-64': 'middle_aged', '65-69': 'middle_aged',
        '70-74': 'old', '75-79': 'old', '80+': 'old'
    }
    cardio['Age_Category'] = cardio['Age_Category'].replace(age_mapping)
    if not cardio['Age_Category'].isin(AGE_CATEGORY_MAPPING.keys()).all():
        raise ValueError(f"Invalid values in Age_Category: must be one of {list(AGE_CATEGORY_MAPPING.keys())}")

    # Numerical columns
    numerical_cols = [
        'Height_cm', 'Weight_kg', 'BMI',
        'Alcohol_Consumption', 'Fruit_Consumption',
        'Green_Vegetables_Consumption', 'FriedPotato_Consumption'
    ]
    for col in numerical_cols:
        cardio[col] = pd.to_numeric(cardio[col], errors='coerce')
        if cardio[col].isna().any():
            raise ValueError(f"Invalid numerical values in {col}")

    # Manual encoding for categorical columns
    general_health_cols = ['General_Health_Poor', 'General_Health_Fair', 'General_Health_Good', 'General_Health_Very Good']
    if not cardio['General_Health'].isin(GENERAL_HEALTH_MAPPING.keys()).all():
        raise ValueError(f"Invalid values in General_Health: must be one of {list(GENERAL_HEALTH_MAPPING.keys())}")
    cardio[general_health_cols] = cardio['General_Health'].apply(
        lambda x: GENERAL_HEALTH_MAPPING[x]
    ).tolist()

    checkup_cols = [
        'Checkup_In the past year', 'Checkup_Within the last 2 years',
        'Checkup_In the last 5 years', 'Checkup_5 years or longer ago'
    ]
    cardio[checkup_cols] = cardio['Checkup'].apply(
        lambda x: CHECKUP_MAPPING[x]
    ).tolist()

    gender_cols = ['Gender_Male']
    if not cardio['Gender'].isin(GENDER_MAPPING.keys()).all():
        raise ValueError(f"Invalid values in Gender: must be one of {list(GENDER_MAPPING.keys())}")
    cardio[gender_cols] = cardio['Gender'].apply(
        lambda x: GENDER_MAPPING[x]
    ).tolist()

    age_cols = ['Age_Category_youth', 'Age_Category_young_adults', 'Age_Category_adults', 'Age_Category_middle_aged']
    cardio[age_cols] = cardio['Age_Category'].apply(
        lambda x: AGE_CATEGORY_MAPPING[x]
    ).tolist()

    # Combine numerical, binary, and encoded categorical columns
    input_processed = pd.concat([
        cardio[numerical_cols + binary_cols + ['Diabetes']],
        cardio[general_health_cols + checkup_cols + gender_cols + age_cols]
    ], axis=1)

    # Ensure all selected features exist
    for col in selected_features:
        if col not in input_processed.columns:
            input_processed[col] = 0

    # Reorder and convert to float
    input_processed = input_processed[selected_features].astype(float)

    # Scale
    if not is_training:
        input_scaled = scaler.transform(input_processed)
        return pd.DataFrame(input_scaled, columns=selected_features)

    return input_processed

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

# Load models and artifacts
try:
    random_forest_model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    selected_features = joblib.load("selected_features.joblib")
except FileNotFoundError as e:
    raise RuntimeError(f"Failed to load model or artifacts: {str(e)}")

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict(data: PredictionInput) -> Dict[str, Any]:
    try:
        input_dict = data.dict()
        # Log input for debugging
        print("Received input:", input_dict)
        processed_input = preprocess_input(input_dict, selected_features, scaler, is_training=False)
        # Log processed input columns and values
        print("Processed input columns:", processed_input.columns.tolist())
        print("Processed input values:", processed_input.values)
        # Convert to NumPy array to remove feature names
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
        # Log response
        print("Prediction response:", response)
        return response
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
