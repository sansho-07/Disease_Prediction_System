"""
inference_service.py
─────────────────────
Full pipeline per the architecture:

    User Input
        ↓
      Scaler          (StandardScaler, fit on training data)
        ↓
      Model           (Random Forest)
        ↓
    SHAP / LIME       (TreeExplainer primary, TabularExplainer fallback)
        ↓
    Top 3 features
        ↓
    Risk level
        ↓
    Frontend display
"""
import os
import logging
import joblib
import numpy as np

from app.services.explainability_service import (
    get_feature_importance,
    init_lime_explainers,
)

logger = logging.getLogger(__name__)

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(rel: str):
    return joblib.load(os.path.join(_BASE, rel))


# ── Models & scalers — loaded once at startup ─────────────────────────────────
MODELS = {
    "heart":    _load("models/trained_models/heart_RandomForest.pkl"),
    "diabetes": _load("models/trained_models/diabetes_RandomForest.pkl"),
    "cancer":   _load("models/trained_models/cancer_RandomForest.pkl"),
}
SCALERS = {
    "heart":    _load("models/scalers/heart_scaler.pkl"),
    "diabetes": _load("models/scalers/diabetes_scaler.pkl"),
    "cancer":   _load("models/scalers/cancer_scaler.pkl"),
}

# ── Load training data to initialise LIME explainers ─────────────────────────
# The .npy files contain RAW (unscaled) feature values.
# BUG FIX #4: The previous comment said "already scaled" which was wrong.
# We scale them here so LIME's background statistics match the model's input space.
def _load_npy(filename: str) -> np.ndarray | None:
    path = os.path.join(_BASE, "data", "processed", filename)
    if os.path.exists(path):
        return np.load(path)
    logger.warning("Training data not found at %s — LIME will be unavailable", path)
    return None


_training_data: dict[str, np.ndarray] = {}
for _key, _file in [("heart",    "Xh_train.npy"),
                     ("diabetes", "Xd_train.npy"),
                     ("cancer",   "Xc_train.npy")]:
    _arr = _load_npy(_file)
    if _arr is not None:
        # Scale raw training data so LIME background matches model input space
        _training_data[_key] = SCALERS[_key].transform(_arr)

if _training_data:
    init_lime_explainers(_training_data)
else:
    logger.warning("No training data loaded — LIME fallback will not be available")


# ─────────────────────────────────────────────────────────────────────────────
# Public predict function
# ─────────────────────────────────────────────────────────────────────────────
def predict(
    model_type: str,
    data: list,
) -> tuple[int, float, list[dict], str]:
    """
    Full pipeline: scale → infer → explain.

    Parameters
    ----------
    model_type : "heart" | "diabetes" | "cancer"
    data       : raw (unscaled) feature values as a flat list

    Returns
    -------
    (prediction, probability, explanation, explainer_method)

    prediction       : int   0 or 1
    probability      : float 0.0–1.0  (probability of positive class)
    explanation      : list[dict]  top-3 features from SHAP or LIME
    explainer_method : str   "SHAP" | "LIME" | "none"
    """
    if model_type not in MODELS:
        raise ValueError(
            f"Unknown model: {model_type!r}. Valid: {list(MODELS)}"
        )

    model  = MODELS[model_type]
    scaler = SCALERS[model_type]

    # Step 1 — scale
    raw    = np.array(data, dtype=float).reshape(1, -1)
    scaled = scaler.transform(raw)

    # Step 2 — infer
    proba = float(model.predict_proba(scaled)[0][1])
    pred  = int(proba >= 0.5)

    # Step 3 — SHAP / LIME explain (raw passed so input_value is annotated)
    explanation, method = get_feature_importance(
        model_type=model_type,
        scaled_input=scaled,
        model=model,
        raw_input=raw,
    )

    logger.debug(
        "%s | pred=%d prob=%.3f method=%s top_feature=%s",
        model_type, pred, proba, method,
        explanation[0]["feature_name"] if explanation else "—",
    )

    return pred, proba, explanation, method
