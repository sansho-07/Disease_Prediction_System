
import joblib
import numpy as np

# ── Load models ──────────────────────────────────────────────
heart_model    = joblib.load("models/trained_models/heart_RandomForest.pkl")
diabetes_model = joblib.load("models/trained_models/diabetes_RandomForest.pkl")
cancer_model   = joblib.load("models/trained_models/cancer_RandomForest.pkl")

# ── Load scalers ─────────────────────────────────────────────
heart_scaler    = joblib.load("models/trained_models/heart_scaler.pkl")
diabetes_scaler = joblib.load("models/trained_models/diabetes_scaler.pkl")
cancer_scaler   = joblib.load("models/trained_models/cancer_scaler.pkl")

# Registry — easy to extend to new diseases
MODELS  = {"heart": heart_model,    "diabetes": diabetes_model,    "cancer": cancer_model}
SCALERS = {"heart": heart_scaler,   "diabetes": diabetes_scaler,   "cancer": cancer_scaler}


def predict(model_type: str, data: list) -> tuple:
    """
    Scale the raw input then predict.
    Returns (prediction_int, probability_float).
    """
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: '{model_type}'. Choose from: {list(MODELS)}")

    model  = MODELS[model_type]
    scaler = SCALERS[model_type]

    # Reshape to 2D, scale, predict
    raw       = np.array(data, dtype=float).reshape(1, -1)
    scaled    = scaler.transform(raw)           # ← THIS was the missing step
    proba     = model.predict_proba(scaled)[0][1]
    pred      = int(proba >= 0.5)

    return pred, float(proba)