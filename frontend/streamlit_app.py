import streamlit as st
import requests

st.title("Cardiovascular Disease Prediction")

backend_url = "https://cvd-prediction-project-mtu-final-year-2.onrender.com/predict"

def user_input():
    return {
        "General_Health": st.selectbox("General Health", ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']),
        "Checkup": st.selectbox("Checkup", ['Never', 'More than 5 years ago', 'Within past 5 years', 'Within past 2 years', 'Within past year']),
        "Exercise": st.selectbox("Exercise", ['Yes', 'No']),
        "Skin_Cancer": st.selectbox("Skin Cancer", ['Yes', 'No']),
        "Other_Cancer": st.selectbox("Other Cancer", ['Yes', 'No']),
        "Depression": st.selectbox("Depression", ['Yes', 'No']),
        "Diabetes": st.selectbox("Diabetes", ['Yes', 'No', 'Pre-diabetes', 'Gestational diabetes', 'No, borderline']),
        "Arthritis": st.selectbox("Arthritis", ['Yes', 'No']),
        "Gender": st.selectbox("Gender", ['Male', 'Female']),
        "Age_Category": st.selectbox("Age Category", ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']),
        "Height_cm": st.number_input("Height (cm)", 100.0, 250.0),
        "Weight_kg": st.number_input("Weight (kg)", 30.0, 300.0),
        "BMI": st.number_input("BMI", 10.0, 60.0),
        "Smoking_History": st.selectbox("Smoking", ['Never', 'Former smoker', 'Current smoker']),
        "Alcohol_Consumption": st.slider("Alcohol Consumption", 0, 20),
        "Fruit_Consumption": st.slider("Fruit Intake", 0, 21),
        "Green_Vegetables_Consumption": st.slider("Greens Intake", 0, 21),
        "FriedPotato_Consumption": st.slider("Fried Potatoes", 0, 21)
    }

data = user_input()

if st.button("Predict"):
    try:
        res = requests.post(backend_url, json=data)
        result = res.json()
        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Probabilities: Yes - {result['probability']['Yes']:.2f}, No - {result['probability']['No']:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

      
