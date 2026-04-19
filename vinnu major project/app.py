"""
============================================================
 INDIA CRIME RISK PREDICTOR — PYTHON FLASK API BACKEND
============================================================
 Merges:
   • Original Express.js routes (/, /predict formula-based)
   • New ML-powered routes using Random Forest + Gradient Boosting
   • All data endpoints for the frontend dashboard

 Run:
   pip install flask flask-cors pandas scikit-learn joblib
   python app.py

 API will start at: http://localhost:5000
 Swagger-style docs: http://localhost:5000/api/info
============================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
CORS(app)   # Allow all origins — frontend can call from any port

# ─────────────────────────────────────────────────────────────
# LOAD TRAINED MODELS (on startup)
# ─────────────────────────────────────────────────────────────
MODELS_DIR = "models"

def load_all():
    try:
        rf  = joblib.load(f"{MODELS_DIR}/rf_model.pkl")
        gb  = joblib.load(f"{MODELS_DIR}/gb_model.pkl")
        les = joblib.load(f"{MODELS_DIR}/le_state.pkl")
        led = joblib.load(f"{MODELS_DIR}/le_district.pkl")
        lec = joblib.load(f"{MODELS_DIR}/le_crime.pkl")
        print("✅  All ML models loaded successfully.")
        return rf, gb, les, led, lec
    except FileNotFoundError:
        print("❌  Models not found — run train.py first!")
        return None, None, None, None, None

RF, GB, LE_STATE, LE_DISTRICT, LE_CRIME = load_all()
FEATURES = ["Year", "state_enc", "district_enc", "Population"]

def load_json(name):
    path = os.path.join(MODELS_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

# ─────────────────────────────────────────────────────────────
# HELPER: risk level label
# ─────────────────────────────────────────────────────────────
def risk_level(score):
    if score < 3:   return "LOW"
    if score < 6:   return "MODERATE"
    return "HIGH"

# ─────────────────────────────────────────────────────────────
# HELPER: get population for a district from dataset
# ─────────────────────────────────────────────────────────────
def get_population(state, district):
    try:
        df  = pd.read_csv("data.csv")
        row = df[(df["State"] == state) & (df["District"] == district)]
        if not row.empty:
            return int(row.sort_values("Year", ascending=False).iloc[0]["Population"])
    except Exception:
        pass
    return 5_000_000  # default fallback

# ═════════════════════════════════════════════════════════════
# ROUTE 1 — HOME (same as original Express route)
# Original: app.get("/", ...) → res.send("Backend is running")
# ═════════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def home():
    """Health check — same route as original Express backend."""
    return jsonify({
        "status":  "running",
        "message": "Crime Risk Predictor Python Backend is running",
        "models":  "Random Forest + Gradient Boosting",
        "version": "2.0 (Python Flask)"
    })


# ═════════════════════════════════════════════════════════════
# ROUTE 2 — /predict (UPGRADED from formula to real ML)
# Original Express: const risk = (murder*0.5)+(theft*0.3)+(cyber*0.2)
# NEW: Uses trained Random Forest + Gradient Boosting models
# ═════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict crime risk for a state+district.
    Body: { "state": "Telangana", "district": "Hyderabad" }
    Returns: risk score (0–10), crime type, risk level
    """
    if RF is None:
        return jsonify({"error": "Models not loaded. Run train.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    state    = data.get("state", "").strip()
    district = data.get("district", "").strip()
    year     = int(data.get("year", 2025))

    # Validate inputs
    if not state or not district:
        return jsonify({"error": "Both 'state' and 'district' are required"}), 400

    if state not in LE_STATE.classes_:
        return jsonify({"error": f"Unknown state: '{state}'", "valid_states": list(LE_STATE.classes_)}), 400

    if district not in LE_DISTRICT.classes_:
        return jsonify({"error": f"Unknown district: '{district}'"}), 400

    # Encode inputs the same way training did
    s_enc = int(LE_STATE.transform([state])[0])
    d_enc = int(LE_DISTRICT.transform([district])[0])
    pop   = data.get("population") or get_population(state, district)

    # Build feature vector
    X = pd.DataFrame([[year, s_enc, d_enc, pop]], columns=FEATURES)

    # ── MODEL 1: Gradient Boosting → risk score (0–10) ───────
    raw_risk   = float(GB.predict(X)[0])
    pred_risk  = round(float(np.clip(raw_risk, 0, 10)), 2)

    # ── MODEL 2: Random Forest → dominant crime type ──────────
    crime_enc  = int(RF.predict(X)[0])
    crime_type = LE_CRIME.inverse_transform([crime_enc])[0]

    # ── ALSO compute old formula score for comparison ─────────
    murder = data.get("murder", 0)
    theft  = data.get("theft", 0)
    cyber  = data.get("cyber", 0)
    formula_score = round((murder * 0.5) + (theft * 0.3) + (cyber * 0.2), 4)

    return jsonify({
        "state":                state,
        "district":             district,
        "year":                 year,
        "population":           pop,

        # ML predictions (primary)
        "ml_risk_score":        pred_risk,
        "risk_level":           risk_level(pred_risk),
        "predicted_crime_type": crime_type,

        # Model info
        "models_used": {
            "risk_score":  "Gradient Boosting Regressor (200 estimators)",
            "crime_type":  "Random Forest Classifier (200 trees, 93.75% accuracy)"
        },

        # Legacy formula score (from original Express backend — kept for compatibility)
        "formula_risk_score": formula_score,
        "note": "ml_risk_score is the primary prediction (ML-based). formula_risk_score is the legacy calculation."
    })


# ═════════════════════════════════════════════════════════════
# ROUTE 3 — /predict/batch
# Predict for ALL districts at once (used on frontend load)
# ═════════════════════════════════════════════════════════════
@app.route("/predict/batch", methods=["GET"])
def predict_batch():
    """Return pre-computed 2025 predictions for all 60 districts."""
    data = load_json("predictions_2025.json")
    if not data:
        return jsonify({"error": "Run train.py first to generate predictions"}), 503
    return jsonify({"year": 2025, "count": len(data), "districts": data})


# ═════════════════════════════════════════════════════════════
# ROUTE 4 — /hotspots
# Top N highest-risk districts
# ═════════════════════════════════════════════════════════════
@app.route("/hotspots", methods=["GET"])
def hotspots():
    """Return top N highest-risk districts for 2025."""
    limit = int(request.args.get("limit", 10))
    data  = load_json("predictions_2025.json")
    if not data:
        return jsonify({"error": "Run train.py first"}), 503
    sorted_data = sorted(data, key=lambda x: x["predicted_risk_2025"], reverse=True)
    return jsonify({
        "year":     2025,
        "hotspots": [
            {**d, "risk_level": risk_level(d["predicted_risk_2025"])}
            for d in sorted_data[:limit]
        ]
    })


# ═════════════════════════════════════════════════════════════
# ROUTE 5 — /dashboard/trend
# Yearly aggregated crime data for charts
# ═════════════════════════════════════════════════════════════
@app.route("/dashboard/trend", methods=["GET"])
def dashboard_trend():
    """Return 20-year national crime trend data."""
    data = load_json("yearly_trend.json")
    if not data:
        return jsonify({"error": "Run train.py first"}), 503
    return jsonify({"trend": data})


# ═════════════════════════════════════════════════════════════
# ROUTE 6 — /dashboard/states
# State-level crime totals
# ═════════════════════════════════════════════════════════════
@app.route("/dashboard/states", methods=["GET"])
def dashboard_states():
    """Return state-level crime summary."""
    data = load_json("state_summary.json")
    if not data:
        return jsonify({"error": "Run train.py first"}), 503
    return jsonify({"states": data})


# ═════════════════════════════════════════════════════════════
# ROUTE 7 — /states
# List of all available states
# ═════════════════════════════════════════════════════════════
@app.route("/states", methods=["GET"])
def get_states():
    """Return all available states and districts."""
    meta = load_json("meta.json")
    if not meta:
        return jsonify({"states": list(LE_STATE.classes_) if LE_STATE else []})
    return jsonify(meta)


# ═════════════════════════════════════════════════════════════
# ROUTE 8 — /districts/<state>
# Districts for a specific state
# ═════════════════════════════════════════════════════════════
@app.route("/districts/<state>", methods=["GET"])
def get_districts(state):
    """Return all districts for a given state."""
    meta = load_json("meta.json")
    if not meta:
        return jsonify({"error": "Run train.py first"}), 503
    districts = meta.get("districts", {}).get(state, [])
    if not districts:
        return jsonify({"error": f"No districts found for state: {state}"}), 404
    return jsonify({"state": state, "districts": districts})


# ═════════════════════════════════════════════════════════════
# ROUTE 9 — /compare
# Side-by-side district comparison
# ═════════════════════════════════════════════════════════════
@app.route("/compare", methods=["GET"])
def compare():
    """
    Compare two districts.
    Query: /compare?d1=Hyderabad&s1=Telangana&d2=Mumbai&s2=Maharashtra
    """
    d1 = request.args.get("d1")
    s1 = request.args.get("s1")
    d2 = request.args.get("d2")
    s2 = request.args.get("s2")
    if not all([d1, s1, d2, s2]):
        return jsonify({"error": "Provide d1, s1, d2, s2 query params"}), 400

    data   = load_json("predictions_2025.json") or []
    match1 = next((x for x in data if x["State"]==s1 and x["District"]==d1), None)
    match2 = next((x for x in data if x["State"]==s2 and x["District"]==d2), None)
    if not match1:
        return jsonify({"error": f"Not found: {d1}, {s1}"}), 404
    if not match2:
        return jsonify({"error": f"Not found: {d2}, {s2}"}), 404

    higher = d1 if match1["predicted_risk_2025"] >= match2["predicted_risk_2025"] else d2
    return jsonify({
        d1: {**match1, "risk_level": risk_level(match1["predicted_risk_2025"])},
        d2: {**match2, "risk_level": risk_level(match2["predicted_risk_2025"])},
        "higher_risk_district": higher
    })


# ═════════════════════════════════════════════════════════════
# ROUTE 10 — /api/info
# API documentation
# ═════════════════════════════════════════════════════════════
@app.route("/api/info", methods=["GET"])
def api_info():
    """API overview — all endpoints."""
    return jsonify({
        "project":  "India Crime Risk Predictor",
        "version":  "2.0",
        "backend":  "Python Flask",
        "models": {
            "classifier":  "Random Forest (200 trees) — Crime Type",
            "regressor":   "Gradient Boosting (200 estimators) — Risk Score",
            "accuracy":    "Random Forest: 93.75% | GBR R²: 0.78",
            "features":    ["Year", "state_enc", "district_enc", "Population"]
        },
        "endpoints": {
            "GET  /":                    "Health check",
            "POST /predict":             "ML prediction for state+district",
            "GET  /predict/batch":       "Pre-computed 2025 predictions (all 60 districts)",
            "GET  /hotspots?limit=10":   "Top N highest-risk districts",
            "GET  /dashboard/trend":     "20-year national crime trend",
            "GET  /dashboard/states":    "State-level crime summary",
            "GET  /states":              "All states and district lists",
            "GET  /districts/<state>":   "Districts for a given state",
            "GET  /compare?d1&s1&d2&s2": "Compare two districts side-by-side",
            "GET  /api/info":            "This documentation"
        }
    })


# ─────────────────────────────────────────────────────────────
# START SERVER
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CRIME RISK PREDICTOR — PYTHON FLASK BACKEND")
    print("=" * 60)
    print("  Server : http://localhost:5000")
    print("  Docs   : http://localhost:5000/api/info")
    print("  Models : Random Forest + Gradient Boosting")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
