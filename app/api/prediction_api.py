from fastapi import FastAPI, HTTPException
from app.schemas.request_models import HeartInput, DiabetesInput, CancerInput
from app.services.inference_service import predict
from app.utils.risk_engine import risk_level, simple_explanation

app = FastAPI(title="Multi-Disease Prediction API")


# HEART ENDPOINT

@app.post("/predict/heart")
def predict_heart(data: HeartInput):

    try:
        model_input = list(data.dict().values())

        pred, proba = predict("heart", model_input)

        return {
            "prediction": int(pred),
            "probability": float(proba),
            "risk_level": risk_level(proba),
            "explanation": simple_explanation("heart")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# DIABETES ENDPOINT

@app.post("/predict/diabetes")
def predict_diabetes(data: DiabetesInput):

    try:
        model_input = list(data.dict().values())

        pred, proba = predict("diabetes", model_input)

        return {
            "prediction": int(pred),
            "probability": float(proba),
            "risk_level": risk_level(proba),
            "explanation": simple_explanation("diabetes")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# CANCER ENDPOINT

@app.post("/predict/cancer")
def predict_cancer(data: CancerInput):

    try:
        model_input = data.features

        pred, proba = predict("cancer", model_input)

        return {
            "prediction": int(pred),
            "probability": float(proba),
            "risk_level": risk_level(proba),
            "explanation": simple_explanation("cancer")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))