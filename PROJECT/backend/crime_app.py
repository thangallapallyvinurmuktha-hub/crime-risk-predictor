from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================
# APP INITIALIZATION
# =====================================================
app = Flask(__name__)
CORS(app)

# =====================================================
# LOAD & VALIDATE DATA
# =====================================================
DATA_FILE = "city_crime_rates.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("city_crime_rates.csv not found in backend folder.")

crime_df = pd.read_csv(DATA_FILE)
crime_df.columns = crime_df.columns.str.strip()

required_columns = {"City", "CrimeType", "CrimesPerLakh"}

if not required_columns.issubset(crime_df.columns):
    raise ValueError("CSV file must contain: City, CrimeType, CrimesPerLakh")

crime_df["City"] = crime_df["City"].astype(str).str.strip()
crime_df["CrimeType"] = crime_df["CrimeType"].astype(str).str.strip()
crime_df["CrimesPerLakh"] = pd.to_numeric(
    crime_df["CrimesPerLakh"], errors="coerce"
)

crime_df = crime_df.dropna()

MAX_CRIME_RATE = crime_df["CrimesPerLakh"].max()

# =====================================================
# MACHINE LEARNING SETUP
# =====================================================
city_encoder = LabelEncoder()
crime_encoder = LabelEncoder()

crime_df["CityEncoded"] = city_encoder.fit_transform(crime_df["City"])
crime_df["CrimeEncoded"] = crime_encoder.fit_transform(crime_df["CrimeType"])

crime_df["Target"] = (
    crime_df["CrimesPerLakh"] > crime_df["CrimesPerLakh"].median()
).astype(int)

X = crime_df[["CityEncoded", "CrimeEncoded"]]
y = crime_df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_accuracy = accuracy_score(y_test, log_model.predict(X_test))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# =====================================================
# RULE-BASED WEIGHTS
# =====================================================
GENDER_WEIGHTS = {
    "Male": 1.0,
    "Female": 1.1,
    "Others": 1.05
}

FATAL_WEIGHTS = {
    "Fatal": 1.2,
    "Non-Fatal": 1.0
}

CASE_WEIGHTS = {
    "Pending": 1.1,
    "Closed": 1.0
}

PRECAUTIONS = {
    "Theft": ["Keep valuables secure", "Avoid isolated areas"],
    "Robbery": ["Stay alert in crowded areas", "Avoid late-night travel alone"],
    "Assault": ["Avoid confrontation", "Stay in well-lit areas"],
    "Rape": ["Avoid unsafe locations", "Share live location"],
    "Murder": ["Avoid high-risk zones", "Stay alert in unfamiliar areas"],
    "CyberCrime": ["Do not share OTP", "Use strong passwords"]
}

def get_risk_level(risk):
    if risk >= 70:
        return "High"
    elif risk >= 40:
        return "Moderate"
    return "Low"

# =====================================================
# API ROUTES
# =====================================================

@app.route("/cities", methods=["GET"])
def get_cities():
    return jsonify(sorted(crime_df["City"].unique()))

@app.route("/crimes", methods=["GET"])
def get_crimes():
    return jsonify(sorted(crime_df["CrimeType"].unique()))

@app.route("/model_evaluation", methods=["GET"])
def model_evaluation():
    return jsonify({
        "Logistic_Regression_Accuracy (%)": round(log_accuracy * 100, 2),
        "Random_Forest_Accuracy (%)": round(rf_accuracy * 100, 2),
        "Best_Model": "Random Forest" if rf_accuracy > log_accuracy else "Logistic Regression"
    })

@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    importance = rf_model.feature_importances_
    return jsonify({
        "City_Importance": round(float(importance[0]), 4),
        "CrimeType_Importance": round(float(importance[1]), 4)
    })

@app.route("/calculate", methods=["POST"])
def calculate_risk():
    try:
        data = request.get_json()

        city = data.get("city")
        crime = data.get("crime")
        gender = data.get("gender")
        fatal_status = data.get("fatal_status")
        case_status = data.get("case_status")

        if not all([city, crime, gender, fatal_status, case_status]):
            return jsonify({"error": "All fields are required."}), 400

        row = crime_df[
            (crime_df["City"].str.lower() == city.lower()) &
            (crime_df["CrimeType"].str.lower() == crime.lower())
        ]

        if row.empty:
            return jsonify({"error": "City or crime type not found."}), 404

        base_rate = float(row["CrimesPerLakh"].values[0])

        # Rule-based risk
        rule_risk = (base_rate / MAX_CRIME_RATE) * 100
        rule_risk *= GENDER_WEIGHTS.get(gender, 1.0)
        rule_risk *= FATAL_WEIGHTS.get(fatal_status, 1.0)
        rule_risk *= CASE_WEIGHTS.get(case_status, 1.0)

        # ML Risk
        city_encoded = city_encoder.transform([city])[0]
        crime_encoded = crime_encoder.transform([crime])[0]

        ml_probability = rf_model.predict_proba(
            [[city_encoded, crime_encoded]]
        )[0][1]

        ml_risk = ml_probability * 100

        # Hybrid Final Risk
        final_risk = round(min((rule_risk * 0.5 + ml_risk * 0.5), 100), 2)

        return jsonify({
            "city": city,
            "crime": crime,
            "exposure_risk_percent": final_risk,
            "risk_level": get_risk_level(final_risk),
            "precautions": PRECAUTIONS.get(crime, []),
            "model_used": "Hybrid (Rule-Based + Random Forest ML)"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================
# RUN SERVER
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)