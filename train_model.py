import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os

# Download from GitHub
url = "https://github.com/Yutiisampson/CVD-PREDICTION-PROJECT-MTU-FINAL-YEAR/blob/main/data/CVDdata_mini.csv"
df = pd.read_csv(url)

# Split X and y
X = df.drop("Cardio_Disease", axis=1)
y = df["Cardio_Disease"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save model
os.makedirs("backend/models", exist_ok=True)
joblib.dump(model, "backend/models/random_forest_model.joblib")
joblib.dump(scaler, "backend/models/scaler.joblib")
joblib.dump(X.columns.tolist(), "backend/models/selected_features.joblib")
