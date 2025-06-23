import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined selected features (avoiding joblib dependency)
SELECTED_FEATURES = [
    'Height_cm', 'Weight_kg', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption',
    'Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis',
    'Smoking_History', 'Diabetes',
    'General_Health_Fair', 'General_Health_Good', 'General_Health_Poor',
    'General_Health_Very Good', 'Checkup_In the last 5 years',
    'Checkup_Never', 'Checkup_Within the last 2 years',
    'Checkup_5 years or longer ago', 'Gender_Male',
    'Age_Category_adults', 'Age_Category_middle_aged', 'Age_Category_old',
    'Age_Category_young_adults'
]

def preprocess_input(data, scaler, is_training=False):
    """
    Preprocess input data for CVD prediction.
    Args:
        data: Dict or DataFrame with input features.
        scaler: Scikit-learn scaler object.
        is_training: Boolean to toggle scaling (False for inference).
    Returns:
        Processed and scaled DataFrame.
    """
    logger.info("Preprocessing input data")
    cardio = pd.DataFrame([data] if isinstance(data, dict) else data)
    logger.info(f"Raw input: {json.dumps(cardio.to_dict(orient='records'), indent=2)}")

    # Validate required columns
    required_cols = [
        'General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer',
        'Depression', 'Diabetes', 'Arthritis', 'Gender', 'Age_Category',
        'Height_cm', 'Weight_kg', 'BMI', 'Smoking_History',
        'Alcohol_Consumption', 'Fruit_Consumption',
        'Green_Vegetables_Consumption', 'FriedPotato_Consumption'
    ]
    missing_cols = [col for col in required_cols if col not in cardio.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    # Binary features (case-insensitive)
    binary_cols = ['Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']
    for col in binary_cols:
        cardio[col] = cardio[col].str.lower().map({'no': 0, 'yes': 1}).fillna(0)

    # Diabetes (case-insensitive)
    cardio['Diabetes'] = cardio['Diabetes'].str.lower().map({
        'no': 0, 'yes': 1, 'no, pre-diabetes or borderline diabetes': 0,
        'yes, but female told only during pregnancy': 1
    }).fillna(0)

    # Checkup normalization
    checkup_map = {
        'within the past year': 'in the past year',
        'within the past 2 years': 'within the last 2 years',
        'within the past 5 years': 'in the last 5 years',
        '5 or more years ago': '5 years or longer ago',
        'never': 'never'
    }
    cardio['Checkup'] = cardio['Checkup'].str.lower().map(checkup_map).fillna('in the past year')

    # Age category grouping
    age_map = {
        '18-24': 'youth', '25-29': 'youth',
        '30-34': 'young_adults', '35-39': 'young_adults',
        '40-44': 'adults', '45-49': 'adults',
        '50-54': 'middle_aged', '55-59': 'middle_aged',
        '60-64': 'middle_aged', '65-69': 'middle_aged',
        '70-74': 'old', '75-79': 'old', '80+': 'old'
    }
    cardio['Age_Category'] = cardio['Age_Category'].map(age_map).fillna('youth')

    # Numerical features
    numerical_cols = [
        'Height_cm', 'Weight_kg', 'BMI', 'Alcohol_Consumption',
        'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption'
    ]
    for col in numerical_cols:
        cardio[col] = pd.to_numeric(cardio[col], errors='coerce').fillna(cardio[col].mean())

    # Initialize processed DataFrame
    input_processed = pd.DataFrame(index=cardio.index, columns=SELECTED_FEATURES).fillna(0.0)
    
    # Copy numerical and binary features
    input_processed[numerical_cols + binary_cols + ['Diabetes']] = cardio[numerical_cols + binary_cols + ['Diabetes']]

    # Manual one-hot encoding
    for idx in cardio.index:
        gh = cardio.loc[idx, 'General_Health'].lower()
        general_health_map = {
            'poor': 'General_Health_Poor', 'fair': 'General_Health_Fair',
            'good': 'General_Health_Good', 'very good': 'General_Health_Very Good',
            'excellent': None
        }
        if gh in general_health_map and general_health_map[gh]:
            input_processed.loc[idx, general_health_map[gh]] = 1.0

        chk = cardio.loc[idx, 'Checkup']
        checkup_map = {
            'in the past year': None,
            'within the last 2 years': 'Checkup_Within the last 2 years',
            'in the last 5 years': 'Checkup_In the last 5 years',
            '5 years or longer ago': 'Checkup_5 years or longer ago',
            'never': 'Checkup_Never'
        }
        if chk in checkup_map and checkup_map[chk]:
            input_processed.loc[idx, checkup_map[chk]] = 1.0

        if cardio.loc[idx, 'Gender'].lower() == 'male':
            input_processed.loc[idx, 'Gender_Male'] = 1.0

        age = cardio.loc[idx, 'Age_Category']
        age_map = {
            'youth': None,
            'young_adults': 'Age_Category_young_adults',
            'adults': 'Age_Category_adults',
            'middle_aged': 'Age_Category_middle_aged',
            'old': 'Age_Category_old'
        }
        if age in age_map and age_map[age]:
            input_processed.loc[idx, age_map[age]] = 1.0

    logger.info(f"Processed features: {json.dumps(input_processed.to_dict(orient='records'), indent=2)}")

    # Scale numerical features
    if not is_training:
        numerical_indices = [SELECTED_FEATURES.index(col) for col in numerical_cols]
        input_processed.iloc[:, numerical_indices] = scaler.transform(input_processed.iloc[:, numerical_indices])
        logger.info(f"Scaled features: {json.dumps(input_processed.to_dict(orient='records'), indent=2)}")

    return input_processed
