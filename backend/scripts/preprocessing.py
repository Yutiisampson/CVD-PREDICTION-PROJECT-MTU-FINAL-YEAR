import pandas as pd

def preprocess_input(df, selected_features, scaler):
    df_encoded = pd.get_dummies(df)
    for col in selected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[selected_features]
    scaled = scaler.transform(df_encoded)
    return pd.DataFrame(scaled, columns=selected_features)
