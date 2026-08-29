"""
Shared feature engineering logic.

IMPORTANT: this must produce features in EXACTLY the same way as
model-training/train_model.py, since the model was trained on that
exact transformation. If you change one, change the other.

Data format notes (v3):
- Lights: wattage value, 0 = off, 40 = on
- Fans: wattage value, 0 = off, 75 = on
- AC: wattage value, 0 = off, 150 = on
- Equipment Running: wattage value, 0 = off, 400-600 = on
- Power (W): total = Lights + Fans + AC + Equipment Running
- Class Scheduled / Staff Present: "Yes" / "No" text
"""

import pandas as pd


BINARY_MAP_YN = {"Yes": 1, "No": 0}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw dataframe with the same columns as the WattGrid Excel
    sheet and returns a dataframe with engineered feature columns added.
    """
    df = df.copy()

    def to_hour(t):
        if hasattr(t, "hour"):
            return t.hour
        return int(str(t).split(":")[0])

    df["Hour"] = df["Time"].apply(to_hour)
    df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.dayofweek

    df["ClassScheduled_bin"] = df["Class Scheduled"].map(BINARY_MAP_YN)
    df["Staff_bin"] = df["Staff Present"].map(BINARY_MAP_YN)

    # Lights/Fans/AC/Equipment are now wattage values, not ON/OFF text.
    # "On" = wattage greater than zero.
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

    return df


def build_model_input(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Given an already-engineered dataframe, select and order columns
    to exactly match what the model was trained on. Any expected
    column not present in this batch (e.g. a room type not seen
    here) is added as all-zero.
    """
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0)
    return X
