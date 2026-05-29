def risk_level(probability: float):

    if probability < 0.30:
        return "LOW RISK"
    elif probability < 0.70:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


def simple_explanation(disease: str):

    explanations = {
        "heart": [
            "High blood pressure is a major risk factor",
            "Cholesterol imbalance may affect arteries",
            "Low heart rate performance indicators"
        ],
        "diabetes": [
            "High glucose levels affect insulin response",
            "BMI imbalance increases diabetes risk",
            "Genetic factors may contribute"
        ],
        "cancer": [
            "Abnormal cell structure detected",
            "Irregular tissue patterns observed",
            "Cell density variation indicators"
        ]
    }

    return explanations[disease][:3]