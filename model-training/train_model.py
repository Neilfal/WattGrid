"""
WattGrid - Random Forest training script (v4 - supervised)

Trains a SUPERVISED classifier on electricity consumption data to
predict potential energy waste, directly learning from the existing
'Potential Wastage' labels.

Why supervised instead of unsupervised (Isolation Forest):
We tested unsupervised anomaly detection extensively and found it
plateaus around 30-46% F1 on the waste class, because "statistically
unusual" isn't the same thing as "matches this dataset's specific
waste rule." Testing showed a supervised model achieves ~100%
accuracy on held-out data - meaning 'Potential Wastage' follows a
learnable, mostly deterministic rule in this dataset, which
supervised learning is the correct tool for.

Data format notes:
- Lights: wattage value, 0 = off, 40 = on
- Fans: wattage value, 0 = off, 75 = on
- AC: wattage value, 0 = off, 150 = on
- Equipment Running: wattage value, 0 = off, 400-600 = on
- Power (W): total = Lights + Fans + AC + Equipment Running
- Potential Wastage: treated as binary (anything not literally "Yes"
  counts as "No" - this already folds in any leftover "Maybe" rows)

Run from project root:
    python model-training/train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
DATA_PATH = "../data/WattGrid_Database.xlsx"
df = pd.read_excel(DATA_PATH)
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

df["Lights_bin"] = (df["Lights"] > 0).astype(int)
df["Fans_bin"] = (df["Fans"] > 0).astype(int)
df["AC_bin"] = (df["AC"] > 0).astype(int)
df["Equipment_bin"] = (df["Equipment Running"] > 0).astype(int)

df["DevicesOn"] = df[["Lights_bin", "Fans_bin", "AC_bin", "Equipment_bin"]].sum(axis=1)
df["OccupancyRatio"] = df["Students Present"] / df["Room Capacity"]
df["PowerPerPerson"] = df["Power (W)"] / (df["Students Present"] + 1)

df["Occupied"] = ((df["Students Present"] > 0) | (df["Staff_bin"] == 1)).astype(int)
df["DeviceOnNoOccupant"] = ((df["DevicesOn"] > 0) & (df["Occupied"] == 0)).astype(int)
df["ACorFanNoOccupant"] = (((df["AC_bin"] == 1) | (df["Fans_bin"] == 1)) & (df["Occupied"] == 0)).astype(int)
df["DeviceOnNoClass"] = ((df["DevicesOn"] > 0) & (df["ClassScheduled_bin"] == 0)).astype(int)
df["PowerNoOccupant"] = df["Power (W)"] * (1 - df["Occupied"])
df["AllDevicesOnEmptyRoom"] = ((df["DevicesOn"] >= 3) & (df["Occupied"] == 0)).astype(int)

df = pd.get_dummies(df, columns=["Room Type"], prefix="RoomType")

# ---------------------------------------------------------------------
# 3. Select features and label
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
# Anything not literally "Yes" (including any leftover "Maybe") counts as "No"
y = df["Potential Wastage"].apply(lambda v: "Yes" if v == "Yes" else "No")

# ---------------------------------------------------------------------
# 4. Train/test split - evaluate on data the model has NEVER seen,
#    for an honest accuracy number (not measuring against training data)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 5. Train supervised Random Forest classifier
# ---------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",   # handles the Yes/No imbalance in the data
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------
# 6. Evaluate on held-out test set (honest, unseen-data accuracy)
# ---------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

print("\n--- Evaluation on held-out test data (20%, never seen during training) ---")
print(confusion_matrix(y_test, y_pred, labels=["No", "Yes"]))
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------
# 7. Retrain on FULL dataset for the final deployed model
#    (common practice: once validated on a held-out split, use all
#    available data for the model that actually gets shipped)
# ---------------------------------------------------------------------
X_all_scaled = scaler.fit_transform(X)
model.fit(X_all_scaled, y)

# ---------------------------------------------------------------------
# 8. Save model + scaler + feature list for the backend
# ---------------------------------------------------------------------
joblib.dump(model, "../backend/model.pkl")
joblib.dump(scaler, "../backend/scaler.pkl")
joblib.dump(feature_cols, "../backend/feature_cols.pkl")

print("\nSaved model.pkl, scaler.pkl, feature_cols.pkl to backend/")
print("(Note: threshold.pkl is no longer used - the classifier predicts Yes/No directly)")

preview = df[["Date", "Time", "Room", "Power (W)", "Potential Wastage"]].head(10).copy()
preview["Predicted_Wastage"] = model.predict(X_all_scaled[:10])
print(preview)
