import os, sys, json, uuid
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import engineer_features
from db_postgres import init_db, save_prediction, fetch_recent_predictions, db_health

ART = ROOT / "artifacts"
MODEL = joblib.load(ART / "models" / "champion_xgboost.joblib")
EXPLAINER = joblib.load(ART / "models" / "shap_explainer.joblib")
FEATURES = joblib.load(ART / "models" / "feature_names.joblib")
META = json.loads((ART / "models" / "model_metadata.json").read_text())
BUSINESS = json.loads((ART / "metrics" / "business_metrics.json").read_text())

app = FastAPI(title="Credit Risk AI API", version="1.0.0", description="Real-time credit delinquency scoring API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class Applicant(BaseModel):
    applicant_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0)
    age: int = Field(ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(default=0, ge=0)
    DebtRatio: float = Field(ge=0)
    MonthlyIncome: float | None = Field(default=None, ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(default=0, ge=0)
    NumberOfTimes90DaysLate: int = Field(default=0, ge=0)
    NumberRealEstateLoansOrLines: int = Field(default=0, ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(default=0, ge=0)
    NumberOfDependents: float | None = Field(default=None, ge=0)


def payload_to_frame(a: Applicant) -> pd.DataFrame:
    return pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": a.RevolvingUtilizationOfUnsecuredLines,
        "age": a.age,
        "NumberOfTime30-59DaysPastDueNotWorse": a.NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio": a.DebtRatio,
        "MonthlyIncome": np.nan if a.MonthlyIncome is None else a.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": a.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": a.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines": a.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse": a.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents": np.nan if a.NumberOfDependents is None else a.NumberOfDependents,
    }])


def explain(x: pd.DataFrame):
    vals = EXPLAINER.shap_values(x)
    if isinstance(vals, list):
        vals = vals[1]
    s = pd.Series(vals[0], index=x.columns).sort_values(key=np.abs, ascending=False).head(8)
    return [{"feature": k, "shap_value": float(v), "direction": "increases risk" if v > 0 else "decreases risk"} for k, v in s.items()]


def score(a: Applicant):
    raw = payload_to_frame(a)
    x = engineer_features(raw).reindex(columns=FEATURES)
    probability = float(MODEL.predict_proba(x)[:, 1][0])
    threshold = float(BUSINESS["threshold"])
    if probability < threshold * 0.5:
        band = "Low"
    elif probability < threshold:
        band = "Moderate"
    elif probability < min(threshold * 1.8, 0.65):
        band = "High"
    else:
        band = "Very High"
    decision = "WITHIN_APPROVAL_THRESHOLD" if probability < threshold else "MANUAL_REVIEW_OR_DECLINE_THRESHOLD"
    drivers = explain(x)
    result = {
        "request_id": str(uuid.uuid4()), "applicant_id": a.applicant_id,
        "risk_probability": probability, "risk_band": band,
        "policy_threshold": threshold, "decision": decision,
        "top_model_drivers": drivers,
        "model": META.get("champion", "XGBoost"), "timestamp": datetime.now(timezone.utc).isoformat()
    }
    try:
        save_prediction(result, raw.iloc[0].to_dict())
    except Exception as e:
        result["database_warning"] = str(e)
    return result

@app.on_event("startup")
def startup():
    try:
        init_db()
    except Exception:
        pass

@app.get("/health")
def health():
    return {"status": "ok", "model": META.get("champion"), "database": db_health()}

@app.get("/model-info")
def model_info():
    return {"metadata": META, "business_metrics": BUSINESS}

@app.post("/predict")
def predict(applicant: Applicant):
    try:
        return score(applicant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

@app.get("/predictions")
def predictions(limit: int = 50):
    return {"items": fetch_recent_predictions(min(max(limit, 1), 200))}
