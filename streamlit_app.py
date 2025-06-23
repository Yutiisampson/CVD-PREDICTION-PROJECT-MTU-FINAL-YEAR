import streamlit as st
import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App title
st.title("Cardiovascular Disease Prediction")

# Environment variable for API URL
API_URL = os.getenv("API_URL", "https://cvd-prediction-project-mtu-final-year-2.onrender.com/predict")

# Session state initialization
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Reset button
if st.button("Reset Form"):
    st.session_state.submitted = False
    st.session_state.prediction_result = None
    st.rerun()

# Input form
with st.form("prediction_form"):
    st.header("Patient Details")
    
    general_health = st.selectbox("General Health", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
    checkup = st.selectbox("Last Checkup", ["In the past year", "Within the last 2 years", 
                                            "In the last 5 years", "5 years or longer ago", "Never"])
    exercise = st.selectbox("Exercise", ["Yes", "No"])
    skin_cancer = st.selectbox("Skin Cancer", ["Yes", "No"])
    other_cancer = st.selectbox("Other Cancer", ["Yes", "No"])
    depression = st.selectbox("Depression", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes", "No, pre-diabetes or borderline diabetes", 
                                         "Yes, but female told only during pregnancy"])
    arthritis = st.selectbox("Arthritis", ["Yes", "No"])
    gender =  st.selectbox("Gender", ["Male", "Female"])
    age_category = st.selectbox("Age Category", ["18-24", "25-29", "30-34", "35-39", "40-44", 
                                                 "45-49", "50-54", "55-59", "60-64", "65-69", 
                                                 "70-74", "75-79", "80+"])
    height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.2)
    smoking_history = st.selectbox("Smoking History", ["Yes", "No"])
    alcohol_consumption = st.number_input("Alcohol Consumption (drinks/month)", min_value=0, max_value=100, value=2)
    fruit_consumption = st.number_input("Fruit Consumption (servings/month)", min_value=0, max_value=100, value=60)
    green_vegetables = st.number_input("Green Vegetables Consumption (servings/month)", min_value=0, max_value=100, value=30)
    fried_potato = st.number_input("Fried Potato Consumption (servings/month)", min_value=0, max_value=100, value=2)

    # Validate BMI consistency
    calculated_bmi = weight_kg / ((height_cm / 100) ** 2)
    if abs(bmi - calculated_bmi) > 0.5:
        st.warning(f"BMI ({bmi:.2f}) does not match height/weight ({calculated_bmi:.2f}). Please verify.")

    submitted = st.form_submit_button("Predict")

    if submitted:
        st.session_state.submitted = True
        st.session_state.prediction_result = None

        inputs = {
            "General_Health": general_health,
            "Checkup": checkup,
            "Exercise": exercise,
            "Skin_Cancer": skin_cancer,
            "Other_Cancer": other_cancer,
            "Depression": depression,
            "Diabetes": diabetes,
            "Arthritis": arthritis,
            "Gender": gender,
            "Age_Category": age_category,
            "Height_cm": height_cm,
            "Weight_kg": weight_kg,
            "BMI": bmi,
            "Smoking_History": smoking_history,
            "Alcohol_Consumption": alcohol_consumption,
            "Fruit_Consumption": fruit_consumption,
            "Green_Vegetables_Consumption": green_vegetables,
            "FriedPotato_Consumption": fried_potato
        }
        
        logger.info(f"Sending inputs to API: {inputs}")
        st.write("**Debug: Input Data to API**", inputs)

        try:
            with st.spinner("Predicting..."):
                response = requests.post(API_URL, json=inputs, timeout=30)
                response.raise_for_status()
                result = response.json()
                logger.info(f"API Response: {result}")
                st.write("**Debug: API Response**", result)
                st.session_state.prediction_result = result
        except requests.exceptions.RequestException as e:
            st.error(f"Prediction failed: {str(e)}")
            st.session_state.prediction_result = None

# Show prediction result
if st.session_state.submitted:
    result = st.session_state.prediction_result
    if result:
        st.success(f"Prediction: {result['prediction']}")
        st.write(f"🟢 Probability of No Heart Disease: **{result['probability']['No']:.2%}**")
        st.write(f"🔴 Probability of Heart Disease: **{result['probability']['Yes']:.2%}**")
        st.write("**Full API Response**:", result)
    else:
        st.warning("No prediction result available. Try again.")
