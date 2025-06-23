import pandas as pd
import numpy as np

def preprocess_input(data, selected_features, scaler, is_training=False):
    cardio = pd.DataFrame([data] if isinstance(data, dict) else data)

    # Binary encoding
    binary_cols = ['Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']
    for col in binary_cols:
        cardio[col] = cardio[col].map({'No': 0, 'Yes': 1})

    # Diabetes simplified
    cardio['Diabetes'] = cardio['Diabetes'].map({
        'No': 0,
        'Yes': 1,
        'No, pre-diabetes or borderline diabetes': 0,
        'Yes, but female told only during pregnancy': 1
    })

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

    # One-hot encode
    categorical_cols = [
        'General_Health', 'Checkup', 'Gender', 'Age_Category'
    ]
    dummies = pd.get_dummies(cardio[categorical_cols], drop_first=False)

    # Binary + numerical + encoded categorical
    input_processed = pd.concat([cardio[numerical_cols + binary_cols + ['Diabetes']], dummies], axis=1)

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
