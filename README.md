# Credit Risk Prediction API

A production-ready REST API that predicts loan default risk and explains every decision — built to show the difference between a data science demo and a real ML engineering deployment.

🔗 **[Live API](https://ml-credit-risk-api.onrender.com)** · 📖 **[Interactive Docs](https://ml-credit-risk-api.onrender.com/docs)**

---

## Why I Built This

Most ML projects stop at a Jupyter notebook or a Streamlit demo. But in production, models are never served through a UI — they're called by other services, mobile apps, and backend systems through REST APIs.

This project takes the same XGBoost credit risk model from my Streamlit project and wraps it in a proper FastAPI service. Send a JSON request with applicant data, get back a JSON response with the default probability, the decision, and the exact SHAP values that explain why the model made that call.

---

## What the API Returns

Every prediction response includes three things:

- **Default probability** — a number between 0 and 1
- **Decision** — HIGH_RISK or LOW_RISK
- **SHAP explanation** — the top 3 features that drove this specific prediction, plus all 11 feature contributions

This means the API is not just a black box. Every response is explainable, which matters in regulated industries like banking and insurance.

---

## Example

**Request**
```json
POST /predict

{
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
```

**Response**
```json
{
  "default_probability": 0.42,
  "decision": "LOW_RISK",
  "confidence": 0.58,
  "top_shap_features": [
    {
      "feature": "loan_grade",
      "shap_value": 0.31,
      "direction": "increases_risk"
    },
    {
      "feature": "person_income",
      "shap_value": -0.28,
      "direction": "decreases_risk"
    },
    {
      "feature": "loan_percent_income",
      "shap_value": 0.19,
      "direction": "increases_risk"
    }
  ],
  "all_shap_values": {
    "person_age": -0.12,
    "person_income": -0.28,
    "loan_grade": 0.31,
    "loan_percent_income": 0.19,
    "..."  : "..."
  }
}
```

---

## Model Performance

The underlying XGBoost model was trained on 32,581 real loan applications from Kaggle.

| Metric | Score |
|---|---|
| ROC-AUC | 0.95 |
| Accuracy | 93% |
| F1 Score (default class) | 0.82 |
| Training samples | 26,064 |
| Test samples | 6,517 |

---

## Tech Stack

| Technology | Role |
|---|---|
| FastAPI | REST API framework — handles routing, validation, and auto-generates /docs |
| XGBoost | Gradient boosted classifier trained on credit risk data |
| SHAP | Computes per-prediction feature contributions using Shapley values |
| Pydantic | Validates incoming request data and enforces schema |
| joblib | Loads the trained model and encoders from disk |
| Render | Cloud platform for deployment |

---

## Project Structure
```
ml-credit-risk-api/
├── models/
│   ├── xgboost_model.joblib       # Trained XGBoost classifier
│   ├── label_encoders.joblib      # Fitted encoders for categorical features
│   └── feature_cols.joblib        # Feature column order
├── main.py                        # FastAPI app — all endpoints
├── requirements.txt
├── Procfile                       # Render deployment config
└── README.md
```

---

## Run Locally

**1. Clone and install**
```bash
git clone https://github.com/divya-vj/ml-credit-risk-api.git
cd ml-credit-risk-api
pip install -r requirements.txt
```

**2. Start the server**
```bash
uvicorn main:app --reload
```

**3. Open the docs**
```
http://localhost:8000/docs
```

The interactive Swagger UI lets you test the API directly from the browser — no Postman needed.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | / | Health check — confirms API is running |
| POST | /predict | Send applicant data, get prediction + SHAP |
| GET | /docs | Interactive API documentation |

---

- **[Credit Risk Predictor — Streamlit](https://github.com/divya-vj/ml-shap-explainability)** — The same model with an interactive web UI and visual SHAP charts

---



