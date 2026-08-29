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
1. Run: git clone https://github.com/Neilfal/WattGrid.git then cd WattGrid 
2. Make sure Python 3.10+ is installed (python --version to check). This is the only hard prerequisite besides Git
3. Run: python -m venv venv, then activate it — Windows: .\venv\Scripts\activate, Mac/Linux: source venv/bin/activate.
4. Run: pip install -r requirements.txt
5. Run: cd backend (from project root) then uvicorn main:app --reload.

# then visit http://127.0.0.1:8000/docs to test /predict and /analyze

# Optional: Since model.pkl, scaler.pkl, etc. are already committed to backend/, you technically don't need to retrain — but if you want to verify the training process itself works, you run: cd model-training then python train_model.py. This regenerates the model files from data/WattGrid_Database.xlsx and prints the accuracy report.