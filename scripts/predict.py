import pandas as pd
import numpy as np
import joblib
import requests
from io import StringIO
from preprocessing import preprocess_input 


GITHUB_RAW_URL = "https://raw.githubusercontent.com/yourusername/CVD_Final_Year_Project_MTU/main/data/CVDdata_mini.csv"  # Replace with your URL
try:
    response = requests.get(GITHUB_RAW_URL)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text))
    print("Dataset schema:", data.columns.tolist())
except Exception as e:
    raise RuntimeError(f"Failed to load dataset from GitHub: {str(e)}")

# Load model artifacts
try:
    model = joblib.load("../models/random_forest_model.joblib")
    scaler = joblib.load("../models/scaler.joblib")
    selected_features = joblib.load("../models/selected_features.joblib")
    print("Selected features:", selected_features)
    print("Feature importances:", dict(zip(selected_features, model.feature_importances_)))
except FileNotFoundError as e:
    raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

# High-risk input for testing
test_input = {
    "General_Health": "Poor",
    "Checkup": "In the past year",
    "Exercise": "No",
    "Skin_Cancer": "No",
    "Other_Cancer": "No",
    "Depression": "Yes",
    "Diabetes": "Yes",
    "Arthritis": "Yes",
    "Gender": "Male",
    "Age_Category": "80+",
    "Height_cm": 170.0,
    "Weight_kg": 90.0,
    "BMI": 31.1,
    "Smoking_History": "Yes",
    "Alcohol_Consumption": 10,
    "Fruit_Consumption": 20,
    "Green_Vegetables_Consumption": 10,
    "FriedPotato_Consumption": 15
}

# Preprocess and predict
try:
    print("Test input:", test_input)
    processed_input = preprocess_input(test_input, selected_features, scaler, is_training=False)
    print("Processed input columns:", processed_input.columns.tolist())
    print("Raw processed values:", processed_input.values)
    print("Scaled input values:", processed_input.to_numpy())
    processed_input_array = processed_input.to_numpy()
    prediction = model.predict(processed_input_array)[0]
    prediction_proba = model.predict_proba(processed_input_array)[0]
    response = {
        "prediction": "Yes" if prediction == 1 else "No",
        "probability": {
            "Yes": float(prediction_proba[1]),
            "No": float(prediction_proba[0]),
        }
    }
    print("Prediction response:", response)
except Exception as e:
    print(f"Error during prediction: {str(e)}")
