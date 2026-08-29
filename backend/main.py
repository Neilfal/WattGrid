"""
WattGrid backend - FastAPI app.

Endpoints:
    GET  /                 - health check
    POST /predict           - single reading -> waste prediction
    POST /analyze            - upload CSV/Excel -> full report of flagged rows

Run from the backend/ folder:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import io

from feature_utils import engineer_features, build_model_input

app = FastAPI(title="WattGrid API")

# Allow your frontend (running on a different port/domain) to call this API.
# For a hackathon, "*" is fine. Tighten this to your actual frontend URL
# before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Load model artifacts once at startup
# ---------------------------------------------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
threshold = joblib.load("threshold.pkl")  # calibrated for high recall on waste


# ---------------------------------------------------------------------
# Request schema for a single live reading (the "form" use case)
# ---------------------------------------------------------------------
class Reading(BaseModel):
    Date: str            # e.g. "2026-11-01"
    Time: str             # e.g. "14:00"
    Room: str
    Room_Type: str        # "Classroom" or "Lab"
    Students_Present: float
    Staff_Present: str    # "Yes" / "No"
    Class_Scheduled: str  # "Yes" / "No"
    Lights: float          # wattage: 0 = off, 40 = on
    Fans: float             # wattage: 0 = off, 75 = on
    AC: float               # wattage: 0 = off, 150 = on
    Equipment_Running: float  # wattage: 0 = off, 400-600 = on
    Power_W: float          # should equal Lights + Fans + AC + Equipment_Running
    Room_Capacity: float


def _reading_to_dataframe(r: Reading) -> pd.DataFrame:
    """Convert the API's field names (underscored) back to the
    original Excel column names the feature engineering expects."""
    return pd.DataFrame([{
        "Date": r.Date,
        "Time": r.Time,
        "Room": r.Room,
        "Room Type": r.Room_Type,
        "Students Present": r.Students_Present,
        "Staff Present": r.Staff_Present,
        "Class Scheduled": r.Class_Scheduled,
        "Lights": r.Lights,
        "Fans": r.Fans,
        "AC": r.AC,
        "Equipment Running": r.Equipment_Running,
        "Power (W)": r.Power_W,
        "Room Capacity": r.Room_Capacity,
    }])


def _predict(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Shared prediction logic used by both endpoints."""
    df_feat = engineer_features(df_raw)
    X = build_model_input(df_feat, feature_cols)
    X_scaled = scaler.transform(X)

    scores = model.decision_function(X_scaled)     # lower = more unusual
    flagged = scores < threshold                    # calibrated cutoff, tuned for high recall

    df_raw = df_raw.copy()
    df_raw["Predicted_Wastage"] = pd.Series(flagged).map({True: "Yes", False: "No"}).values
    df_raw["Anomaly_Score"] = scores
    return df_raw


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "WattGrid API is running"}


@app.post("/predict")
def predict_single(reading: Reading):
    """Single live reading -> waste prediction. Used by the website's form."""
    df_raw = _reading_to_dataframe(reading)
    try:
        result = _predict(df_raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    row = result.iloc[0]
    return {
        "predicted_wastage": row["Predicted_Wastage"],
        "anomaly_score": float(row["Anomaly_Score"]),
    }


@app.post("/analyze")
async def analyze_bulk(file: UploadFile = File(...)):
    """Upload a CSV/Excel of readings -> full report of flagged rows.
    Expects the same columns as the original WattGrid dataset."""
    contents = await file.read()

    try:
        if file.filename.endswith(".csv"):
            df_raw = pd.read_csv(io.BytesIO(contents))
        else:
            df_raw = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    df_raw = df_raw.dropna(how="all")

    try:
        result = _predict(df_raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing data: {e}")

    total = len(result)
    flagged = (result["Predicted_Wastage"] == "Yes").sum()

    # Convert dates/times to strings so they're JSON-serializable
    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            result[col] = result[col].astype(str)

    result = result.astype(object).where(pd.notnull(result), None)

    return {
        "total_rows": total,
        "flagged_rows": int(flagged),
        "flagged_percentage": round(100 * flagged / total, 2) if total else 0,
        "results": result.to_dict(orient="records"),
    }
