"""
prediction_api.py — FastAPI application.

Pipeline per architecture:
    User Input → Scaler → Model → SHAP/LIME explainer
                                        ↓
                              Top 3 contributing features
                                        ↓
                                   Risk level
                                        ↓
                              Frontend display

Endpoints: /health, /predict/{heart|diabetes|cancer}, /history, /stats
"""
import os
import traceback
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.schemas.request_models import HeartInput, DiabetesInput, CancerInput
from app.services.inference_service import predict
from app.services.db_service import log_prediction, get_history, get_stats

app = FastAPI(
    title="Multi-Disease Prediction API",
    description=(
        "Heart Disease, Diabetes, and Breast Cancer prediction using ML. "
        "Explanations powered by SHAP TreeExplainer (primary) with "
        "LIME TabularExplainer as fallback."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cancer baseline (mean of all 30 features from training CSV) ───────────────
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CANCER_CSV = os.path.join(_BASE, "data", "raw", "breast_cancer.csv")

try:
    _df = pd.read_csv(_CANCER_CSV)
    _cancer_baseline: np.ndarray = _df.drop(columns=["target"]).mean().values
    assert len(_cancer_baseline) == 30, (
        f"Expected 30 cancer features, got {len(_cancer_baseline)}"
    )
except Exception as exc:
    raise RuntimeError(
        f"Cannot load cancer baseline from {_CANCER_CSV}: {exc}"
    ) from exc


# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_level(p: float) -> str:
    if p < 0.30:
        return "Low Risk"
    if p < 0.70:
        return "Medium Risk"
    return "High Risk"


# Recommendation messages sourced from the explainable_ai notebook
_RECOMMENDATIONS: dict[tuple[str, int], str] = {
    ("heart",    1): (
        "⚠️ Consult a cardiologist immediately. "
        "Possible indicators: abnormal cholesterol, high blood pressure, "
        "chest pain, or reduced heart performance."
    ),
    ("heart",    0): "✅ Maintain a healthy lifestyle with regular exercise and a balanced diet.",
    ("diabetes", 1): (
        "⚠️ Consult an endocrinologist. "
        "Possible indicators: high glucose level, BMI imbalance, or insulin issues."
    ),
    ("diabetes", 0): "✅ Keep monitoring glucose levels and maintain a healthy weight.",
    ("cancer",   0): "⚠️ Immediate medical diagnosis required. Important indicators detected in tissue pattern.",
    ("cancer",   1): "✅ Results appear benign. Continue routine screenings as advised by your doctor.",
}


def _get_recommendation(disease: str, pred: int) -> str:
    return _RECOMMENDATIONS.get(
        (disease, pred), "Consult a qualified healthcare professional."
    )


def _build_response(
    disease: str,
    pred: int,
    proba: float,
    risk: str,
    explanation: list,
    explainer_method: str,
    inputs: dict,
    patient_id: str = "",
) -> dict:
    log_prediction(disease, patient_id, inputs, pred, proba, risk, explanation)
    return {
        "prediction":       pred,
        "probability":      round(proba, 4),
        "risk_level":       risk,
        "explainer_method": explainer_method,   # "SHAP" | "LIME" | "none"
        "explanation":      explanation,
        "recommendation":   _get_recommendation(disease, pred),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


@app.post("/predict/heart")
def predict_heart(data: HeartInput, patient_id: str = Query(default="")):
    try:
        # BUG FIX #1: Pydantic v2 removed .dict() — use .model_dump() instead
        inputs = data.model_dump()
        pred, proba, expl, method = predict("heart", list(inputs.values()))
        return _build_response(
            "heart", pred, proba, _risk_level(proba),
            expl, method, inputs, patient_id,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput, patient_id: str = Query(default="")):
    try:
        # BUG FIX #1: Pydantic v2 removed .dict() — use .model_dump() instead
        inputs = data.model_dump()
        pred, proba, expl, method = predict("diabetes", list(inputs.values()))
        return _build_response(
            "diabetes", pred, proba, _risk_level(proba),
            expl, method, inputs, patient_id,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/cancer")
def predict_cancer(data: CancerInput, patient_id: str = Query(default="")):
    try:
        # Fill the 5 user-provided features into the 30-feature baseline vector
        features = _cancer_baseline.copy()
        features[0] = data.radius
        features[1] = data.texture
        features[2] = data.perimeter
        features[3] = data.area
        features[4] = data.smoothness

        # BUG FIX #1: Pydantic v2 removed .dict() — use .model_dump() instead
        inputs = data.model_dump()
        pred, proba, expl, method = predict("cancer", features.tolist())

        # Cancer model: class 1 = benign, class 0 = malignant
        # Invert probability for risk display (high benign prob = low risk)
        cancer_risk = _risk_level(1.0 - proba)

        # BUG FIX #2: was passing cancer_risk as both pred AND risk arguments
        # (positional args were: disease, pred=cancer_risk, proba, risk=cancer_risk, ...)
        # Correct: pred is the integer 0/1, risk is the string label
        return _build_response(
            "cancer", pred, proba, cancer_risk,
            expl, method, inputs, patient_id,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def history(
    disease: str = Query(default=None),
    limit: int = Query(default=20, le=100),
):
    return get_history(disease=disease, limit=limit)


@app.get("/stats")
def stats():
    return get_stats()
