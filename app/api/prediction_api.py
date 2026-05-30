# app/api/prediction_api.py
from fastapi import FastAPI, HTTPException
from app.schemas.request_models import HeartInput, DiabetesInput, CancerInput
from app.services.inference_service import predict
from app.utils.risk_engine import risk_level, simple_explanation
import numpy as np
import pandas as pd
import joblib

app = FastAPI(
    title="Multi-Disease Prediction API",
    description="Heart Disease, Diabetes, and Breast Cancer prediction using ML"
)

# Load cancer dataset baseline once at startup (for the 30-feature vector trick)
_cancer_df = pd.read_csv("data/raw/breast_cancer.csv")
_cancer_baseline = _cancer_df.drop("target", axis=1).mean().values


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/heart")
def predict_heart(data: HeartInput):
    try:
        model_input = list(data.dict().values())
        pred, proba = predict("heart", model_input)
        return {
            "prediction":  pred,
            "probability": round(proba, 4),
            "risk_level":  risk_level(proba),
            "explanation": simple_explanation("heart"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):
    try:
        model_input = list(data.dict().values())
        pred, proba = predict("diabetes", model_input)
        return {
            "prediction":  pred,
            "probability": round(proba, 4),
            "risk_level":  risk_level(proba),
            "explanation": simple_explanation("diabetes"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/cancer")
def predict_cancer(data: CancerInput):   
    try:
        # Build the 30-feature vector: start from dataset means, override 5 user inputs
        features = _cancer_baseline.copy()
        features[0] = data.radius
        features[1] = data.texture
        features[2] = data.perimeter
        features[3] = data.area
        features[4] = data.smoothness

        pred, proba = predict("cancer", features.tolist())
        return {
            "prediction":  pred,
            "probability": round(proba, 4),
            "risk_level":  risk_level(proba),
            "explanation": simple_explanation("cancer"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))