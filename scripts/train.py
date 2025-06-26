import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import requests
from io import StringIO
from preprocessing import preprocess_input

# Load dataset from GitHub
GITHUB_RAW_URL = "https://raw.githubusercontent.com/yourusername/CVD_Final_Year_Project_MTU/main/data/CVDdata_mini.csv"  # Replace with your URL
try:
    response = requests.get(GITHUB_RAW_URL)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text))
    print("Dataset schema:", data.columns.tolist())
except Exception as e:
    raise RuntimeError(f"Failed to load dataset from GitHub: {str(e)}")

# Prepare features and target
X = data.drop(columns=['Heart_Disease'])  # Adjust target column name
y = data['Heart_Disease'].map({'Yes': 1, 'No': 0})  # Adjust encoding

# Preprocess training data
selected_features = [
    'Height_cm', 'Weight_kg', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption', 'Exercise',
    'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History',
    'Diabetes', 'General_Health_Poor', 'General_Health_Fair', 'General_Health_Good',
    'General_Health_Very Good', 'Checkup_In the past year',
    'Checkup_Within the last 2 years', 'Checkup_In the last 5 years',
    'Checkup_5 years or longer ago', 'Gender_Male', 'Age_Category_youth',
    'Age_Category_young_adults', 'Age_Category_adults', 'Age_Category_middle_aged'
]
X_processed, scaler = preprocess_input(X, selected_features, is_training=True)

# Train Random Forest with balanced class weights
model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10, class_weight="balanced")
model.fit(X_processed, y)

# Save model and artifacts
joblib.dump(model, "../models/random_forest_model.joblib")
joblib.dump(scaler, "../models/scaler.joblib")
joblib.dump(selected_features, "../models/selected_features.joblib")

print("Class distribution:", y.value_counts())
print("Feature importances:", dict(zip(selected_features, model.feature_importances_)))
