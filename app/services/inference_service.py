import joblib
import numpy as np

# LOAD MODELS

heart_model = joblib.load("models/trained_models/heart_RandomForest.pkl")
diabetes_model = joblib.load("models/trained_models/diabetes_RandomForest.pkl")
cancer_model = joblib.load("models/trained_models/cancer_RandomForest.pkl")


def predict(model_type, data):

    data = np.array(data).reshape(1, -1)

    if model_type == "heart":
        model = heart_model

    elif model_type == "diabetes":
        model = diabetes_model

    elif model_type == "cancer":
        model = cancer_model

    else:
        raise ValueError("Invalid model type")

    proba = model.predict_proba(data)[0][1]
    pred = model.predict(data)[0]

    return pred, proba