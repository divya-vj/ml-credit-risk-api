# main.py — Credit Risk Prediction REST API
# FastAPI + XGBoost + SHAP

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import shap

# ── Load artifacts ─────────────────────────────────────────
model    = joblib.load("models/xgboost_model.joblib")
encoders = joblib.load("models/label_encoders.joblib")
features = joblib.load("models/feature_cols.joblib")
explainer = shap.TreeExplainer(model)

# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk Prediction API",
    description="XGBoost model that predicts loan default probability and explains the decision using SHAP values.",
    version="1.0.0"
)

# ── Request schema ─────────────────────────────────────────
class ApplicantData(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str   # RENT, MORTGAGE, OWN, OTHER
    person_emp_length: float
    loan_intent: str             # PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION
    loan_grade: str              # A, B, C, D, E, F, G
    loan_amnt: int
    loan_int_rate: float
    cb_person_default_on_file: str  # N, Y
    cb_person_cred_hist_length: int

    class Config:
        json_schema_extra = {
            "example": {
                "person_age": 28,
                "person_income": 55000,
                "person_home_ownership": "RENT",
                "person_emp_length": 4.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "C",
                "loan_amnt": 12000,
                "loan_int_rate": 13.5,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 5
            }
        }

# ── Response schema ────────────────────────────────────────
class PredictionResponse(BaseModel):
    default_probability: float
    decision: str
    confidence: float
    top_shap_features: list
    all_shap_values: dict

# ── Health check ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Credit Risk Prediction API is running",
        "docs": "/docs",
        "predict": "/predict"
    }

# ── Prediction endpoint ────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(data: ApplicantData):

    # Encode categoricals
    home_enc    = encoders["person_home_ownership"].transform([data.person_home_ownership])[0]
    intent_enc  = encoders["loan_intent"].transform([data.loan_intent])[0]
    grade_enc   = encoders["loan_grade"].transform([data.loan_grade])[0]
    default_enc = encoders["cb_person_default_on_file"].transform([data.cb_person_default_on_file])[0]

    # Compute loan_percent_income
    loan_percent_income = data.loan_amnt / data.person_income if data.person_income > 0 else 0

    # Build feature dataframe
    input_df = pd.DataFrame([[
        data.person_age,
        data.person_income,
        home_enc,
        data.person_emp_length,
        intent_enc,
        grade_enc,
        data.loan_amnt,
        data.loan_int_rate,
        loan_percent_income,
        default_enc,
        data.cb_person_cred_hist_length
    ]], columns=features)

    # Predict
    prob = float(model.predict_proba(input_df)[0][1])
    pred = int(model.predict(input_df)[0])

    # SHAP values
    shap_vals = explainer.shap_values(input_df)[0]

    # Top 3 SHAP features
    shap_pairs = sorted(
        zip(features, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top_shap = [
        {
            "feature": feat,
            "shap_value": round(float(val), 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk"
        }
        for feat, val in shap_pairs[:3]
    ]

    # All SHAP values
    all_shap = {feat: round(float(val), 4) for feat, val in zip(features, shap_vals)}

    return PredictionResponse(
        default_probability=round(prob, 4),
        decision="HIGH_RISK" if pred == 1 else "LOW_RISK",
        confidence=round(max(prob, 1 - prob), 4),
        top_shap_features=top_shap,
        all_shap_values=all_shap
    )