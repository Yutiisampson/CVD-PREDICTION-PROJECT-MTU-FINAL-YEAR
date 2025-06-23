import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_input(data, selected_features, scaler, is_training=False):
    cardio = pd.DataFrame([data] if isinstance(data, dict) else data)
    
    # Binary encoding (set to 0/1 directly)
    binary_cols = ['Exercise', 'Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History']
    for col in binary_cols:
        cardio[col] = cardio[col].replace({'No': 0, 'Yes': 1})
    
    cardio['Diabetes'] = cardio['Diabetes'].replace(
        {
            'No': 0,
            'Yes': 1,
            'No, pre-diabetes or borderline diabetes': 0,
            'Yes, but female told only during pregnancy': 1
        }
    )
    
    # Simplify Checkup
    cardio['Checkup'] = cardio['Checkup'].replace(
        {
            'Within the past year': 'In the past year',
            'Within the past 2 years': 'Within the last 2 years',
            'Within the past 5 years': 'In the last 5 years',
            '5 or more years ago': '5 years or longer ago',
            'Never': 'Never'
        }
    )
    
    # Simplify Age_Category
    cardio['Age_Category'] = cardio['Age_Category'].replace(
        {
            '18-24': 'youth',
            '25-29': 'youth',
            '30-34': 'young_adults',
            '35-39': 'young_adults',
            '40-44': 'adults',
            '45-49': 'adults',
            '50-54': 'middle_aged',
            '55-59': 'middle_aged',
            '60-64': 'middle_aged',
            '65-69': 'middle_aged',
            '70-74': 'old',
            '75-79': 'old',
            '80+': 'old'
        }
    )
    
    # Numerical columns
    numerical_cols = ['Height_cm', 'Weight_kg', 'BMI', 'Alcohol_Consumption', 
                      'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
    
    # One-hot encode categorical features
    categorical_features = ['General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer', 
                            'Depression', 'Diabetes', 'Arthritis', 'Gender', 'Age_Category', 'Smoking_History']
    input_categorical = pd.get_dummies(cardio[categorical_features], drop_first=True).astype(int)
    
    # Combine numerical and categorical
    input_processed = pd.concat([cardio[numerical_cols], input_categorical], axis=1)
    
    # Align columns with selected_features
    for col in selected_features:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[selected_features]
    
    # Scale numerical features
    if not is_training:
        input_scaled = scaler.transform(input_processed)
        return input_scaled
    return input_processed
