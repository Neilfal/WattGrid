# WattGrid

An AI-powered system for detecting potential electricity waste in classrooms/labs,
using an unsupervised Isolation Forest model trained on occupancy, device state,
and power consumption data.

## Status
- ✅ Data pipeline + feature engineering (model-training/train_model.py)
- ✅ Trained Isolation Forest model (backend/model.pkl)
- ✅ FastAPI backend with /predict and /analyze endpoints
- Rudimentary Frontend only, in progress
- Biometric status check not done
- Making ai model more accurate using more data
- Physical demonstration not integrated yet(integration left, circuit is built)

## How it works
The model learns "normal" electricity usage patterns (based on occupancy, class
schedule, device states, and power draw) and flags statistically unusual
combinations — e.g. AC running in an empty room — as potential waste.

## Run it locally
1. cd backend
2. pip install -r ../requirements.txt
3. uvicorn main:app --reload
# then visit http://127.0.0.1:8000/docs to test /predict and /analyze