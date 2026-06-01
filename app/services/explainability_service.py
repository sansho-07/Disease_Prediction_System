"""
explainability_service.py
─────────────────────────
Architecture:
    User Input → Scaler → Model → SHAP/LIME explainer
                                        ↓
                              Top 3 contributing features
                                        ↓
                                   Risk level
                                        ↓
                              Frontend display

Strategy
--------
• Primary  : SHAP TreeExplainer  — exact, fast for Random Forest / XGBoost
• Fallback : LIME TabularExplainer — model-agnostic, used if SHAP fails

Both paths return the same output schema so the rest of the pipeline
is completely unaware of which explainer ran.
"""

import logging
import numpy as np
import shap
import lime
import lime.lime_tabular

logger = logging.getLogger(__name__)

# ── Feature name maps ─────────────────────────────────────────────────────────
FEATURE_NAMES: dict[str, list[str]] = {
    "heart": [
        "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
        "Cholesterol", "Fasting Blood Sugar", "Rest ECG",
        "Max Heart Rate", "Exercise Angina", "Oldpeak",
        "Slope", "Major Vessels", "Thal",
    ],
    "diabetes": [
        "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
        "Insulin", "BMI", "Diabetes Pedigree Function", "Age",
    ],
    "cancer": [
        "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area",
        "Mean Smoothness", "Mean Compactness", "Mean Concavity",
        "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
        "SE Radius", "SE Texture", "SE Perimeter", "SE Area",
        "SE Smoothness", "SE Compactness", "SE Concavity",
        "SE Concave Points", "SE Symmetry", "SE Fractal Dimension",
        "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area",
        "Worst Smoothness", "Worst Compactness", "Worst Concavity",
        "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension",
    ],
}

# ── Plain-English "why" templates keyed by (disease, feature_name) ────────────
_WHY: dict[tuple[str, str], str] = {
    # Heart
    ("heart", "Age"):                    "Older age increases cardiovascular strain.",
    ("heart", "Sex"):                    "Biological sex influences heart disease risk profile.",
    ("heart", "Chest Pain Type"):        "The type of chest pain is a strong clinical indicator.",
    ("heart", "Resting Blood Pressure"): "Elevated blood pressure strains the heart over time.",
    ("heart", "Cholesterol"):            "High cholesterol contributes to arterial plaque build-up.",
    ("heart", "Fasting Blood Sugar"):    "Elevated fasting glucose is linked to cardiovascular risk.",
    ("heart", "Rest ECG"):               "Resting ECG abnormalities can signal underlying heart issues.",
    ("heart", "Max Heart Rate"):         "Lower max heart rate suggests reduced cardiac capacity.",
    ("heart", "Exercise Angina"):        "Chest pain during exercise is a key warning sign.",
    ("heart", "Oldpeak"):                "ST depression during exercise signals ischemia.",
    ("heart", "Slope"):                  "Downsloping ST segment is associated with ischemia.",
    ("heart", "Major Vessels"):          "More blocked vessels directly raises heart disease risk.",
    ("heart", "Thal"):                   "Thalassemia type affects blood oxygen delivery.",
    # Diabetes
    ("diabetes", "Pregnancies"):                "More pregnancies can affect insulin sensitivity.",
    ("diabetes", "Glucose"):                    "High blood glucose is the primary diabetes marker.",
    ("diabetes", "Blood Pressure"):             "Hypertension often co-occurs with diabetes.",
    ("diabetes", "Skin Thickness"):             "Triceps skinfold thickness correlates with body fat.",
    ("diabetes", "Insulin"):                    "Abnormal insulin levels indicate metabolic issues.",
    ("diabetes", "BMI"):                        "Higher BMI is strongly linked to insulin resistance.",
    ("diabetes", "Diabetes Pedigree Function"): "Family history raises genetic predisposition.",
    ("diabetes", "Age"):                        "Risk of type-2 diabetes increases with age.",
    # Cancer
    ("cancer", "Mean Radius"):           "Larger cell radius is associated with malignancy.",
    ("cancer", "Mean Texture"):          "Rough texture indicates irregular nuclear structure.",
    ("cancer", "Mean Perimeter"):        "Irregular perimeter suggests abnormal cell growth.",
    ("cancer", "Mean Area"):             "Greater cell area is a marker of tumour aggressiveness.",
    ("cancer", "Mean Smoothness"):       "Low smoothness reflects irregular cell boundaries.",
    ("cancer", "Mean Compactness"):      "High compactness is linked to malignant cell shape.",
    ("cancer", "Mean Concavity"):        "Concave portions of the cell contour signal abnormality.",
    ("cancer", "Mean Concave Points"):   "Number of concave points reflects tumour irregularity.",
    ("cancer", "Mean Symmetry"):         "Asymmetric cells are a hallmark of malignancy.",
    ("cancer", "Mean Fractal Dimension"):"Complex fractal patterns indicate irregular growth.",
    ("cancer", "Worst Radius"):          "Worst-case radius strongly predicts malignancy.",
    ("cancer", "Worst Perimeter"):       "Worst-case perimeter is a top malignancy predictor.",
    ("cancer", "Worst Concavity"):       "Deep concavities in cell shape indicate malignancy.",
    ("cancer", "Worst Concave Points"):  "Worst-case concave points are a strong malignancy signal.",
    ("cancer", "Worst Area"):            "Largest cell area observed is a key malignancy marker.",
}

# ── SHAP explainer cache (one per model type) ─────────────────────────────────
_SHAP_CACHE: dict[str, shap.TreeExplainer] = {}

# ── LIME explainer cache — needs training data statistics ─────────────────────
_LIME_CACHE: dict[str, lime.lime_tabular.LimeTabularExplainer] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Public initialiser — call once at startup with training data
# ─────────────────────────────────────────────────────────────────────────────
def init_lime_explainers(training_data: dict[str, np.ndarray]) -> None:
    """
    Pre-build LIME explainers from scaled training data.

    Parameters
    ----------
    training_data : dict  {model_type: X_train_scaled  (n_samples, n_features)}
    """
    for model_type, X_train in training_data.items():
        names = FEATURE_NAMES.get(model_type, [])
        _LIME_CACHE[model_type] = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=names,
            class_names=["Negative", "Positive"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        logger.info("LIME explainer initialised for %s (%d training samples)",
                    model_type, len(X_train))


# ─────────────────────────────────────────────────────────────────────────────
# SHAP path
# ─────────────────────────────────────────────────────────────────────────────
def _get_shap_explainer(model_type: str, model) -> shap.TreeExplainer:
    if model_type not in _SHAP_CACHE:
        _SHAP_CACHE[model_type] = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer built for %s", model_type)
    return _SHAP_CACHE[model_type]


def _shap_values_for_positive_class(raw) -> np.ndarray:
    """
    Normalise every SHAP output format into a 1-D array of per-feature
    values for the *positive* class (class index 1).

    SHAP can return:
      • shap.Explanation  — .values shape (1, n_feat, 2)  or  (1, n_feat)
      • list of arrays    — [neg_class_array, pos_class_array]
      • plain ndarray     — shape (1, n_feat, 2)  or  (1, n_feat)

    BUG FIX #6: The previous plain-ndarray ndim==2 path did v[0] which
    returns shape (2,) for a single sample — that's the two class values
    for feature 0, not the per-feature values. Correct handling below.
    """
    # ── shap.Explanation object (modern SHAP ≥ 0.40) ──────────────────────
    if hasattr(raw, "values"):
        v = np.array(raw.values)          # ensure plain ndarray
        if v.ndim == 3:
            # shape (1, n_feat, n_classes) — take positive class
            return v[0, :, 1]
        if v.ndim == 2:
            # shape (1, n_feat) — single output (regression-style), use as-is
            return v[0]
        # already 1-D
        return v

    # ── Legacy list-of-arrays [neg_class, pos_class] ──────────────────────
    if isinstance(raw, list):
        pos = np.array(raw[1])            # positive class array
        if pos.ndim == 2:
            return pos[0]                 # shape (1, n_feat) → (n_feat,)
        return pos                        # already (n_feat,)

    # ── Plain ndarray ──────────────────────────────────────────────────────
    v = np.array(raw)
    if v.ndim == 3:
        # shape (1, n_feat, n_classes) — take positive class
        return v[0, :, 1]
    if v.ndim == 2:
        # BUG FIX #6: shape could be (1, n_feat) — single output
        # v[0] here correctly gives (n_feat,) since axis-0 is the sample axis
        return v[0]
    # already 1-D
    return v


def _explain_with_shap(
    model_type: str,
    scaled_input: np.ndarray,
    model,
) -> tuple[list[dict], str]:
    """
    Run SHAP TreeExplainer and return (top3_features, 'SHAP').
    Raises on any error so the caller can fall back to LIME.
    """
    explainer = _get_shap_explainer(model_type, model)
    arr = scaled_input.reshape(1, -1)
    raw = explainer.shap_values(arr)
    values = _shap_values_for_positive_class(raw)

    if values is None or len(values) == 0:
        raise ValueError("SHAP returned empty values")

    return _build_top3(model_type, values), "SHAP"


# ─────────────────────────────────────────────────────────────────────────────
# LIME path
# ─────────────────────────────────────────────────────────────────────────────
def _explain_with_lime(
    model_type: str,
    scaled_input: np.ndarray,
    model,
) -> tuple[list[dict], str]:
    """
    Run LIME TabularExplainer and return (top3_features, 'LIME').
    Raises if no LIME explainer is cached for this model type.
    """
    if model_type not in _LIME_CACHE:
        raise RuntimeError(
            f"LIME explainer not initialised for '{model_type}'. "
            "Call init_lime_explainers() at startup."
        )

    n_features = len(FEATURE_NAMES.get(model_type, []))
    lime_exp = _LIME_CACHE[model_type].explain_instance(
        data_row=scaled_input.flatten(),
        predict_fn=model.predict_proba,
        num_features=n_features,
        num_samples=500,
    )

    # LIME returns (feature_description_string, weight) pairs for the positive class.
    # Feature descriptions look like "Glucose > 120.00" or "0.50 < BMI <= 1.20".
    # Match back to feature index by checking if the canonical name appears in the string.
    values = np.zeros(n_features)
    names = FEATURE_NAMES.get(model_type, [])
    for feat_str, weight in lime_exp.as_list(label=1):
        feat_lower = feat_str.lower()
        for idx, name in enumerate(names):
            if name.lower() in feat_lower:
                values[idx] = weight
                break

    return _build_top3(model_type, values), "LIME"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper — build top-3 output from a 1-D importance array
# ─────────────────────────────────────────────────────────────────────────────
def _build_top3(model_type: str, values: np.ndarray) -> list[dict]:
    names = FEATURE_NAMES.get(model_type, [])
    ranked = sorted(enumerate(values), key=lambda x: abs(x[1]), reverse=True)

    result = []
    for idx, val in ranked[:3]:
        name = names[idx] if idx < len(names) else f"Feature {idx}"
        why  = _WHY.get((model_type, name), f"{name} influenced this prediction.")
        result.append({
            "feature_name":  name,
            "feature_index": int(idx),
            "impact":        round(float(val), 4),
            "abs_impact":    round(abs(float(val)), 4),
            "direction":     "increases" if val > 0 else "decreases",
            "why":           why,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def get_feature_importance(
    model_type: str,
    scaled_input: np.ndarray,
    model,
    raw_input: np.ndarray | None = None,
) -> tuple[list[dict], str]:
    """
    SHAP/LIME explainer — follows the architecture:

        User Input → Scaler → Model → SHAP/LIME explainer
                                            ↓
                                  Top 3 contributing features

    Tries SHAP first (exact, fast for tree models).
    Falls back to LIME if SHAP raises any exception.

    Parameters
    ----------
    model_type   : "heart" | "diabetes" | "cancer"
    scaled_input : (1, n_features) scaled array passed to the model
    model        : fitted sklearn / XGBoost estimator
    raw_input    : (1, n_features) original unscaled values (optional,
                   used to annotate each feature with its actual value)

    Returns
    -------
    (explanation: list[dict], method: str)

    Each dict in explanation:
        feature_name  : str
        feature_index : int
        impact        : float   signed importance score
        abs_impact    : float   absolute value (for bar width)
        direction     : "increases" | "decreases"
        why           : str     plain-English clinical reason
        input_value   : float | None   the patient's actual raw value
    """
    # ── Try SHAP ──────────────────────────────────────────────────────────
    try:
        top3, method = _explain_with_shap(model_type, scaled_input, model)
    except Exception as shap_err:
        logger.warning("SHAP failed for %s (%s) — falling back to LIME",
                       model_type, shap_err)
        try:
            top3, method = _explain_with_lime(model_type, scaled_input, model)
        except Exception as lime_err:
            logger.error("LIME also failed for %s (%s)", model_type, lime_err)
            # Last resort: return empty list rather than crash the whole request
            return [], "none"

    # ── Annotate with actual patient input values ─────────────────────────
    # BUG FIX #9: Use explicit None check (not truthiness) so that 0.0 values
    # are still shown. raw_input is a numpy array so check with `is not None`.
    if raw_input is not None:
        flat = np.array(raw_input).flatten()
        for item in top3:
            idx = item["feature_index"]
            item["input_value"] = round(float(flat[idx]), 4) if idx < len(flat) else None
    else:
        for item in top3:
            item["input_value"] = None

    return top3, method
