import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_input(data, selected_features, scaler, is_training=False):
    """
    Preprocess input data for CVD prediction.
    Args:
        data: Dict or DataFrame with input features.
        selected_features: List of features expected by the model.
        scaler: Scikit-learn scaler object.
        is_training: Boolean to toggle scaling (False for inference).
    Returns:
        Processed and scaled DataFrame.
    """
    logger.info("Preprocessing input data")
    cardio = pd.DataFrame([data] if isinstance(data, dict) else data)
    logger.info(f"Raw input: {cardio.to_dict(orient='records')[0]}")

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
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Binary encoding
    binary_cols = ['Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']
    for col in binary_cols:
        cardio[col] = cardio[col].map({'No': 0, 'Yes': 1})
        if cardio[col].isnull().any():
            logger.warning(f"Null values in {col} after mapping")
            cardio[col].fillna(0, inplace=True)

    # Diabetes simplified
    cardio['Diabetes'] = cardio['Diabetes'].map({
        'No': 0,
        'Yes': 1,
        'No, pre-diabetes or borderline diabetes': 0,
        'Yes, but female told only during pregnancy': 1
    })
    if cardio['Diabetes'].isnull().any():
        logger.warning("Null values in Diabetes after mapping")
        cardio['Diabetes'].fillna(0, inplace=True)

    # Checkup normalization
    cardio['Checkup'] = cardio['Checkup'].replace({
        'Within the past year': 'In the past year',
        'Within the past 2 years': 'Within the last 2 years',
        'Within the past 5 years': 'In the last 5 years',
        '5 or more years ago': '5 years or longer ago',
        'Never': 'Never'
    })

    # Age category grouping
    cardio['Age_Category'] = cardio['Age_Category'].replace({
        '18-24': 'youth', '25-29': 'youth',
        '30-34': 'young_adults', '35-39': 'young_adults',
        '40-44': 'adults', '45-49': 'adults',
        '50-54': 'middle_aged', '55-59': 'middle_aged',
        '60-64': 'middle_aged', '65-69': 'middle_aged',
        '70-74': 'old', '75-79': 'old', '80+': 'old'
    })

    # Numerical columns
    numerical_cols = [
        'Height_cm', 'Weight_kg', 'BMI',
        'Alcohol_Consumption', 'Fruit_Consumption',
        'Green_Vegetables_Consumption', 'FriedPotato_Consumption'
    ]

    # Validate numerical columns
    for col in numerical_cols:
        if cardio[col].isnull().any():
            logger.warning(f"Null values in {col}")
            cardio[col].fillna(cardio[col].mean(), inplace=True)

    # One-hot encode
    categorical_cols = ['General_Health', 'Checkup', 'Gender', 'Age_Category']
    dummies = pd.get_dummies(cardio[categorical_cols], drop_first=True)
    logger.info(f"One-hot encoded columns: {dummies.columns.tolist()}")

    # Binary + numerical + encoded categorical
    input_processed = pd.concat([cardio[numerical_cols + binary_cols + ['Diabetes']], dummies], axis=1)

    # Ensure all selected features exist
    for col in selected_features:
        if col not in input_processed.columns:
            input_processed[col] = 0

    # Reorder and convert to float
    input_processed = input_processed[selected_features].astype(float)
    logger.info(f"Processed features: {input_processed.to_dict(orient='records')[0]}")

    # Scale
    if not is_training:
        input_scaled = scaler.transform(input_processed)
        logger.info(f"Scaled features: {input_scaled[0].tolist()}")
        return pd.DataFrame(input_scaled, columns=selected_features)

    return input_processed
