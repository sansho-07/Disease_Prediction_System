from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000"


@app.route("/")
def home():
    return render_template("index.html")


# =========================
# HEART PREDICTION
# =========================
@app.route("/predict/heart", methods=["POST"])
def heart_predict():

    data = request.form.to_dict()

    payload = {
        "age": float(data["age"]),
        "sex": int(data["sex"]),
        "cp": int(data["cp"]),
        "trestbps": float(data["trestbps"]),
        "chol": float(data["chol"]),
        "fbs": int(data["fbs"]),
        "restecg": int(data["restecg"]),
        "thalach": float(data["thalach"]),
        "exang": int(data["exang"]),
        "oldpeak": float(data["oldpeak"]),
        "slope": int(data["slope"]),
        "ca": int(data["ca"]),
        "thal": int(data["thal"])
    }

    res = requests.post(f"{API_URL}/predict/heart", json=payload)

    return render_template("index.html", heart_result=res.json())


# =========================
# DIABETES
# =========================
@app.route("/predict/diabetes", methods=["POST"])
def diabetes_predict():

    data = request.form.to_dict()

    payload = {
        "pregnancies": float(data["pregnancies"]),
        "glucose": float(data["glucose"]),
        "blood_pressure": float(data["blood_pressure"]),
        "skin_thickness": float(data["skin_thickness"]),
        "insulin": float(data["insulin"]),
        "bmi": float(data["bmi"]),
        "dpf": float(data["dpf"]),
        "age": float(data["age"])
    }

    res = requests.post(f"{API_URL}/predict/diabetes", json=payload)

    return render_template("index.html", diabetes_result=res.json())


# =========================
# CANCER
# =========================
@app.route("/predict/cancer", methods=["POST"])
def cancer_predict():

    data = request.form.to_dict()

    features = [float(x) for x in data.values()]

    payload = {
        "features": features
    }

    res = requests.post(f"{API_URL}/predict/cancer", json=payload)

    return render_template("index.html", cancer_result=res.json())


if __name__ == "__main__":
    app.run(port=5000, debug=True)