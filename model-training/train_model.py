"""
WattGrid - Isolation Forest training script (v3 - new wattage-based data)

Trains an unsupervised anomaly detection model on electricity
consumption data to flag potential energy waste, then calibrates
the anomaly-score threshold against the existing 'Potential Wastage'
labels to maximize recall on the waste ("Yes") class - i.e. catch
as many real waste events as possible, at the cost of more false
positives.

Data format notes (v3):
- Lights: wattage value, 0 = off, 40 = on
- Fans: wattage value, 0 = off, 75 = on
- AC: wattage value, 0 = off, 150 = on
- Equipment Running: wattage value, 0 = off, 400-600 = on
- Power (W): total = Lights + Fans + AC + Equipment Running
- Potential Wastage: treated as binary here (anything not literally
  "Yes" counts as "No" for evaluation) - if your sheet still has a
  "Maybe" category, either relabel those rows in the source file or
  they'll be grouped with "No" automatically.

NOTE on methodology: the model itself is trained WITHOUT using the
'Potential Wastage' column (fully unsupervised, feature-based only).
The labels are used ONLY afterward, to pick the best decision
threshold on the model's own anomaly scores.

Run from project root:
    python model-training/train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
DATA_PATH = "../data/WattGrid_Database.xlsx"
df = pd.read_excel(DATA_PATH)

# Drop fully blank separator rows - anchor on 'Room' since a real
# reading always has a room, but a blank spacer row never does.
df = df.dropna(subset=["Room"]).reset_index(drop=True)
print(f"Loaded {len(df)} rows after cleaning.")

# ---------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------
df["Hour"] = df["Time"].apply(lambda t: t.hour)
df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.dayofweek

binary_map_yn = {"Yes": 1, "No": 0}
df["ClassScheduled_bin"] = df["Class Scheduled"].map(binary_map_yn)
df["Staff_bin"] = df["Staff Present"].map(binary_map_yn)

# Lights/Fans/AC/Equipment are wattage values now - "on" = > 0
df["Lights_bin"] = (df["Lights"] > 0).astype(int)
df["Fans_bin"] = (df["Fans"] > 0).astype(int)
df["AC_bin"] = (df["AC"] > 0).astype(int)
df["Equipment_bin"] = (df["Equipment Running"] > 0).astype(int)

df["DevicesOn"] = df[["Lights_bin", "Fans_bin", "AC_bin", "Equipment_bin"]].sum(axis=1)
df["OccupancyRatio"] = df["Students Present"] / df["Room Capacity"]
df["PowerPerPerson"] = df["Power (W)"] / (df["Students Present"] + 1)

# Targeted waste-condition features
df["Occupied"] = ((df["Students Present"] > 0) | (df["Staff_bin"] == 1)).astype(int)
df["DeviceOnNoOccupant"] = ((df["DevicesOn"] > 0) & (df["Occupied"] == 0)).astype(int)
df["ACorFanNoOccupant"] = (((df["AC_bin"] == 1) | (df["Fans_bin"] == 1)) & (df["Occupied"] == 0)).astype(int)
df["DeviceOnNoClass"] = ((df["DevicesOn"] > 0) & (df["ClassScheduled_bin"] == 0)).astype(int)
df["PowerNoOccupant"] = df["Power (W)"] * (1 - df["Occupied"])
df["AllDevicesOnEmptyRoom"] = ((df["DevicesOn"] >= 3) & (df["Occupied"] == 0)).astype(int)

df = pd.get_dummies(df, columns=["Room Type"], prefix="RoomType")

# ---------------------------------------------------------------------
# 3. Select features
# ---------------------------------------------------------------------
feature_cols = [
    "Hour", "DayOfWeek", "Students Present", "Staff_bin",
    "ClassScheduled_bin", "Lights_bin", "Fans_bin", "AC_bin",
    "Equipment_bin", "DevicesOn", "Power (W)", "Room Capacity",
    "OccupancyRatio", "PowerPerPerson", "Occupied", "DeviceOnNoOccupant",
    "ACorFanNoOccupant", "DeviceOnNoClass", "PowerNoOccupant",
    "AllDevicesOnEmptyRoom",
] + [c for c in df.columns if c.startswith("RoomType_")]

X = df[feature_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------
# 4. Train Isolation Forest (unsupervised - no label used here)
# ---------------------------------------------------------------------
model = IsolationForest(
    n_estimators=300,
    contamination=0.2,
    max_features=0.8,
    random_state=42,
)
model.fit(X_scaled)

scores = model.decision_function(X_scaled)  # lower = more anomalous

# ---------------------------------------------------------------------
# 5. Calibrate threshold against labels - optimize for HIGH RECALL on
#    the "Yes" (waste) class. Anything not literally "Yes" (e.g. a
#    leftover "Maybe") is treated as "No" for this calibration.
# ---------------------------------------------------------------------
y_true = (df["Potential Wastage"] == "Yes").astype(int)

best_f1, best_threshold = 0, None
for t in np.percentile(scores, np.arange(5, 60, 1)):
    pred = (scores < t).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print(f"\nCalibrated anomaly-score threshold: {best_threshold:.5f}")

df["Predicted_Wastage"] = pd.Series((scores < best_threshold)).map({True: "Yes", False: "No"})
df["Anomaly_Score"] = scores

y_true_label = df["Potential Wastage"].apply(lambda x: "Yes" if x == "Yes" else "No")
print("\n--- Comparison vs existing 'Potential Wastage' labels (tuned threshold) ---")
print(confusion_matrix(y_true_label, df["Predicted_Wastage"], labels=["No", "Yes"]))
print(classification_report(y_true_label, df["Predicted_Wastage"]))

# ---------------------------------------------------------------------
# 6. Save model + scaler + feature list + threshold for the backend
# ---------------------------------------------------------------------
joblib.dump(model, "../backend/model.pkl")
joblib.dump(scaler, "../backend/scaler.pkl")
joblib.dump(feature_cols, "../backend/feature_cols.pkl")
joblib.dump(best_threshold, "../backend/threshold.pkl")

print("\nSaved model.pkl, scaler.pkl, feature_cols.pkl, threshold.pkl to backend/")
print(df[["Date", "Time", "Room", "Power (W)", "Potential Wastage",
          "Predicted_Wastage", "Anomaly_Score"]].head(10))
