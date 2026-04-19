============================================================
  INDIA CRIME RISK PREDICTOR — SETUP GUIDE
============================================================

TECH STACK
  • Frontend  : HTML + CSS + JavaScript + Chart.js
  • Backend   : Python 3 + Flask
  • ML Models : Random Forest + Gradient Boosting (scikit-learn)
  • Dataset   : India District Crime Data 2005–2024 (CSV)

MODEL RESULTS
  • Random Forest Classifier  → Crime Type  (93.75% accuracy)
  • Gradient Boosting Regressor → Risk Score (R² = 0.78, RMSE = 0.53)

------------------------------------------------------------
HOW TO RUN (3 STEPS)
------------------------------------------------------------

STEP 1 — Install Python dependencies
  pip install -r requirements.txt

STEP 2 — Train the ML models (run once)
  python train.py
  → Creates models/ folder with .pkl files and JSON data

STEP 3 — Start the Flask API
  python app.py
  → Server starts at http://localhost:5000

STEP 4 — Open the frontend
  Open index.html in any browser
  → The app calls http://localhost:5000 for live ML predictions

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------

crime-risk-predictor/
│
├── data.csv              ← Your crime dataset (put here)
├── train.py              ← Train Random Forest + Gradient Boosting
├── app.py                ← Flask API (10 endpoints)
├── index.html            ← Frontend (same design, now calls Python API)
├── requirements.txt      ← Python packages
│
└── models/               ← Created automatically by train.py
    ├── rf_model.pkl          Random Forest Classifier
    ├── gb_model.pkl          Gradient Boosting Regressor
    ├── le_state.pkl          Label encoder for states
    ├── le_district.pkl       Label encoder for districts
    ├── le_crime.pkl          Label encoder for crime types
    ├── yearly_trend.json     20-year chart data
    ├── state_summary.json    State-level totals
    ├── predictions_2025.json 2025 predictions for all 60 districts
    └── meta.json             States and districts lists

------------------------------------------------------------
API ENDPOINTS
------------------------------------------------------------

GET  /                    Health check
POST /predict             Live ML prediction for any district
GET  /predict/batch       Pre-computed 2025 predictions (60 districts)
GET  /hotspots?limit=10   Top N highest-risk districts
GET  /dashboard/trend     20-year national crime trend data
GET  /dashboard/states    State-level crime summary
GET  /states              All states and district lists
GET  /districts/<state>   Districts for a given state
GET  /compare             Compare two districts side-by-side
GET  /api/info            Full API documentation

------------------------------------------------------------
HOW THE PREDICTION WORKS
------------------------------------------------------------

1. User selects State + District in the frontend
2. Frontend calls POST /predict on the Flask API
3. Flask encodes inputs using LabelEncoders
4. Gradient Boosting Regressor predicts risk score (0-10)
5. Random Forest Classifier predicts dominant crime type
6. Result is returned as JSON and displayed in the UI
7. If API is offline, pre-computed ML data is used as fallback

------------------------------------------------------------
MERGED FROM OLD BACKEND (server.js Express)
------------------------------------------------------------
The original Express.js backend used a formula:
  risk = (murder * 0.5) + (theft * 0.3) + (cyber * 0.2)

The new Python Flask backend REPLACES this with real ML:
  risk = GradientBoostingRegressor.predict([Year, State, District, Population])
  type = RandomForestClassifier.predict([Year, State, District, Population])

The formula result is still available in /predict response
under "formula_risk_score" for comparison/reference.

============================================================
