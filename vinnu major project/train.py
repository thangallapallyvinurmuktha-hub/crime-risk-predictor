"""
============================================================
 INDIA CRIME RISK PREDICTOR — ML MODEL TRAINING
============================================================
 Algorithms Used:
   1. Random Forest Classifier  → predicts dominant crime type
   2. Gradient Boosting Regressor → predicts risk score (0–10)

 Run this ONCE before starting the Flask API:
   python train.py
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import json
import os

print("=" * 60)
print("  INDIA CRIME RISK PREDICTOR — TRAINING MODELS")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD & VALIDATE DATASET
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df = pd.read_csv("data.csv")
print(f"      Rows: {len(df)} | Columns: {len(df.columns)}")
print(f"      Null values: {df.isnull().sum().sum()}")
print(f"      States: {df['State'].nunique()} | Districts: {df['District'].nunique()}")
print(f"      Years: {df['Year'].min()} – {df['Year'].max()}")

# ─────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Engineering features...")

# Label-encode State and District (convert text to numbers for ML)
le_state    = LabelEncoder()
le_district = LabelEncoder()
le_crime    = LabelEncoder()

df["state_enc"]    = le_state.fit_transform(df["State"])
df["district_enc"] = le_district.fit_transform(df["District"])

# Compute dominant crime type per row (target for classifier)
crime_cols = ["Murder", "Rape", "Kidnapping", "Theft", "Cyber_Crime"]
df["dominant_crime"] = df[crime_cols].idxmax(axis=1)
df["crime_enc"]      = le_crime.fit_transform(df["dominant_crime"])

# Compute normalised risk score (target for regressor)
# Formula: crimes per 100k population, then min-max scale to 0–10
df["raw_rate"]   = df["Total_Crime"] / df["Population"] * 100_000
rate_min         = df["raw_rate"].min()
rate_max         = df["raw_rate"].max()
df["risk_score"] = (df["raw_rate"] - rate_min) / (rate_max - rate_min) * 10

print(f"      Crime types found: {list(le_crime.classes_)}")
print(f"      Risk score range: {df['risk_score'].min():.3f} – {df['risk_score'].max():.3f}")

# Features used by both models
FEATURES = ["Year", "state_enc", "district_enc", "Population"]
X = df[FEATURES]

# ─────────────────────────────────────────────────────────────
# STEP 3: TRAIN MODEL 1 — RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Training Model 1 — Random Forest Classifier...")
y_clf = df["crime_enc"]
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y_clf, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(
    n_estimators  = 200,        # 200 decision trees in ensemble
    max_depth     = None,       # grow trees to full depth
    min_samples_split = 2,
    random_state  = 42,
    n_jobs        = -1          # use all CPU cores
)
rf_model.fit(X_tr1, y_tr1)

rf_acc = accuracy_score(y_te1, rf_model.predict(X_te1))
print(f"      ✅ Random Forest Accuracy: {rf_acc * 100:.2f}%")
print(f"      Feature importances:")
for feat, imp in zip(FEATURES, rf_model.feature_importances_):
    print(f"        {feat:20s}: {imp:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 4: TRAIN MODEL 2 — GRADIENT BOOSTING REGRESSOR
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Training Model 2 — Gradient Boosting Regressor...")
y_reg = df["risk_score"]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_reg, test_size=0.2, random_state=42)

gb_model = GradientBoostingRegressor(
    n_estimators  = 200,        # 200 boosting stages
    learning_rate = 0.1,        # each tree shrinks contribution
    max_depth     = 4,          # tree complexity
    loss          = "squared_error",
    random_state  = 42
)
gb_model.fit(X_tr2, y_tr2)

preds_te  = gb_model.predict(X_te2)
gb_rmse   = np.sqrt(mean_squared_error(y_te2, preds_te))
gb_r2     = r2_score(y_te2, preds_te)
print(f"      ✅ Gradient Boosting RMSE: {gb_rmse:.4f} | R²: {gb_r2:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 5: SAVE MODELS, ENCODERS, AND STATIC DATA
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Saving models and generating static data...")
os.makedirs("models", exist_ok=True)

# Save trained models and encoders
joblib.dump(rf_model,    "models/rf_model.pkl")
joblib.dump(gb_model,    "models/gb_model.pkl")
joblib.dump(le_state,    "models/le_state.pkl")
joblib.dump(le_district, "models/le_district.pkl")
joblib.dump(le_crime,    "models/le_crime.pkl")
print("      Saved: models/rf_model.pkl  (Random Forest Classifier)")
print("      Saved: models/gb_model.pkl  (Gradient Boosting Regressor)")
print("      Saved: models/le_*.pkl      (Label Encoders)")

# --- Static data for dashboard charts ---
yearly = (
    df.groupby("Year")[["Total_Crime", "Cyber_Crime", "Murder", "Theft", "Rape", "Kidnapping"]]
    .sum()
    .reset_index()
    .to_dict(orient="records")
)

state_summary = (
    df.groupby("State")[["Total_Crime", "Murder", "Rape", "Kidnapping", "Theft", "Cyber_Crime"]]
    .sum()
    .reset_index()
    .assign(risk=lambda d: (d["Total_Crime"] / d["Total_Crime"].max() * 10).round(1))
    .to_dict(orient="records")
)

# Latest year district data
latest_year = df["Year"].max()
latest      = df[df["Year"] == latest_year].copy()
latest["risk_score"] = latest["risk_score"].round(2)

# Generate 2025 predictions for all districts using trained models
predictions_2025 = []
for _, row in latest.iterrows():
    X_pred = pd.DataFrame(
        [[2025, row["state_enc"], row["district_enc"], row["Population"]]],
        columns=FEATURES
    )
    pred_risk       = float(np.clip(gb_model.predict(X_pred)[0], 0, 10))
    pred_crime_enc  = int(rf_model.predict(X_pred)[0])
    pred_crime_type = le_crime.inverse_transform([pred_crime_enc])[0]

    predictions_2025.append({
        "State":                row["State"],
        "District":             row["District"],
        "Total_Crime":          int(row["Total_Crime"]),
        "Murder":               int(row["Murder"]),
        "Rape":                 int(row["Rape"]),
        "Kidnapping":           int(row["Kidnapping"]),
        "Theft":                int(row["Theft"]),
        "Cyber_Crime":          int(row["Cyber_Crime"]),
        "risk_score":           round(row["risk_score"], 2),
        "predicted_risk_2025":  round(pred_risk, 2),
        "predicted_crime_type": pred_crime_type,
        "population":           int(row["Population"]),
    })

# Meta: states and districts lists
meta = {
    "states": sorted(df["State"].unique().tolist()),
    "districts": {
        state: sorted(df[df["State"] == state]["District"].unique().tolist())
        for state in df["State"].unique()
    }
}

# Persist static JSON data
with open("models/yearly_trend.json",   "w") as f: json.dump(yearly, f)
with open("models/state_summary.json",  "w") as f: json.dump(state_summary, f)
with open("models/predictions_2025.json","w") as f: json.dump(predictions_2025, f)
with open("models/meta.json",           "w") as f: json.dump(meta, f)

print("      Saved: models/yearly_trend.json")
print("      Saved: models/state_summary.json")
print("      Saved: models/predictions_2025.json  (60 district forecasts)")
print("      Saved: models/meta.json")

print("\n" + "=" * 60)
print("  ✅  TRAINING COMPLETE!")
print(f"     Random Forest Accuracy : {rf_acc * 100:.2f}%")
print(f"     Gradient Boosting RMSE : {gb_rmse:.4f}")
print(f"     Gradient Boosting R²   : {gb_r2:.4f}")
print("=" * 60)
print("\n  Next step → python app.py")
print("=" * 60)
