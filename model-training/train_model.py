"""
WattGrid - Isolation Forest training script
Trains an unsupervised anomaly detection model on electricity consumption
data to flag potential energy waste.

Run from project root:
    python model-training/train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
DATA_PATH = "../data/WattGrid_Database.xlsx"   # adjust path if needed
df = pd.read_excel(DATA_PATH)

# Drop fully blank separator rows
df = df.dropna(how="all").reset_index(drop=True)

print(f"Loaded {len(df)} rows after cleaning.")

# ---------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------
# Convert Time (datetime.time) to hour of day
df["Hour"] = df["Time"].apply(lambda t: t.hour)

# Day of week as number (Monday=0 ... Sunday=6) - keep it numeric for the model
df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.dayofweek

# Binary-encode Yes/No and ON/OFF fields
binary_map_yn = {"Yes": 1, "No": 0}
binary_map_onoff = {"ON": 1, "OFF": 0}

df["ClassScheduled_bin"] = df["Class Scheduled"].map(binary_map_yn)
df["Lights_bin"] = df["Lights"].map(binary_map_onoff)
df["Fans_bin"] = df["Fans"].map(binary_map_onoff)
df["AC_bin"] = df["AC"].map(binary_map_onoff)
df["Equipment_bin"] = df["Equipment Running"].map(binary_map_onoff)
df["Staff_bin"] = df["Staff Present"].map(binary_map_yn)

# Total "devices on" count - a simple but useful engineered feature
df["DevicesOn"] = df[["Lights_bin", "Fans_bin", "AC_bin", "Equipment_bin"]].sum(axis=1)

# Occupancy ratio: how full the room is (people present vs capacity)
df["OccupancyRatio"] = df["Students Present"] / df["Room Capacity"]

# Power per occupant (avoid divide-by-zero when room is empty)
df["PowerPerPerson"] = df["Power (W)"] / (df["Students Present"] + 1)

# One-hot encode Room Type (Classroom / Lab)
df = pd.get_dummies(df, columns=["Room Type"], prefix="RoomType")

# ---------------------------------------------------------------------
# 3. Select features for the model
# ---------------------------------------------------------------------
feature_cols = [
    "Hour", "DayOfWeek", "Students Present", "Staff_bin",
    "ClassScheduled_bin", "Lights_bin", "Fans_bin", "AC_bin",
    "Equipment_bin", "DevicesOn", "Power (W)", "Room Capacity",
    "OccupancyRatio", "PowerPerPerson",
] + [c for c in df.columns if c.startswith("RoomType_")]

X = df[feature_cols].copy()
X = X.fillna(0)  # safety net for any leftover NaNs (e.g. PowerPerPerson edge cases)

# Scale features - helps Isolation Forest treat all features fairly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# 4. Train Isolation Forest
# ---------------------------------------------------------------------
# contamination = expected proportion of anomalies.
# Your labeled data shows ~20% "Yes" wastage, so we use that as a guide -
# the model itself does NOT see the label column during training.
model = IsolationForest(
    n_estimators=200,
    contamination=0.2,
    random_state=42,
)
model.fit(X_scaled)

# -1 = anomaly (flagged as potential waste), 1 = normal
df["Predicted_Anomaly"] = model.predict(X_scaled)
df["Predicted_Wastage"] = df["Predicted_Anomaly"].map({-1: "Yes", 1: "No"})

# Anomaly score - lower (more negative) = more unusual
df["Anomaly_Score"] = model.decision_function(X_scaled)

# ---------------------------------------------------------------------
# 5. Compare against the existing labeled column (evaluation only)
# ---------------------------------------------------------------------
if "Potential Wastage" in df.columns:
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n--- Comparison vs existing 'Potential Wastage' labels ---")
    print(confusion_matrix(df["Potential Wastage"], df["Predicted_Wastage"], labels=["No", "Yes"]))
    print(classification_report(df["Potential Wastage"], df["Predicted_Wastage"]))

# ---------------------------------------------------------------------
# 6. Save model + scaler + feature list for the backend to use later
# ---------------------------------------------------------------------
joblib.dump(model, "../backend/model.pkl")
joblib.dump(scaler, "../backend/scaler.pkl")
joblib.dump(feature_cols, "../backend/feature_cols.pkl")

print("\nSaved model.pkl, scaler.pkl, feature_cols.pkl to backend/")
print(df[["Date", "Time", "Room", "Power (W)", "Potential Wastage",
          "Predicted_Wastage", "Anomaly_Score"]].head(10))
