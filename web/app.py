import os
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

_api_port = os.environ.get("API_PORT", "8000")
API_URL = f"http://127.0.0.1:{_api_port}"
TIMEOUT = 12  # seconds


def _api_post(endpoint: str, payload: dict, patient_id: str = "") -> dict:
    url = f"{API_URL}{endpoint}"
    if patient_id:
        url += f"?patient_id={patient_id}"
    res = requests.post(url, json=payload, timeout=TIMEOUT)
    res.raise_for_status()
    return res.json()


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    # active_tab="" → JS shows the landing screen, no tab pre-selected
    return render_template("index.html", active_tab="")


@app.route("/history")
def history_page():
    try:
        res = requests.get(f"{API_URL}/history?limit=50", timeout=TIMEOUT)
        rows = res.json() if res.ok else []
    except Exception:
        rows = []
    try:
        sres = requests.get(f"{API_URL}/stats", timeout=TIMEOUT)
        stats = sres.json() if sres.ok else {}
    except Exception:
        stats = {}
    return render_template("history.html", rows=rows, stats=stats)


# ── Predictions ───────────────────────────────────────────────────────────────
@app.route("/predict/heart", methods=["POST"])
def heart_predict():
    d = request.form.to_dict()
    pid = d.get("patient_id", "")
    payload = {
        "age":      float(d["age"]),
        "sex":      int(d["sex"]),
        "cp":       int(d["cp"]),
        "trestbps": float(d["trestbps"]),
        "chol":     float(d["chol"]),
        "fbs":      int(d["fbs"]),
        "restecg":  int(d["restecg"]),
        "thalach":  float(d["thalach"]),
        "exang":    int(d["exang"]),
        "oldpeak":  float(d["oldpeak"]),
        "slope":    int(d["slope"]),
        "ca":       int(d["ca"]),
        "thal":     int(d["thal"]),
    }
    try:
        result = _api_post("/predict/heart", payload, pid)
        return render_template("index.html", heart_result=result, active_tab="heart")
    except requests.exceptions.ConnectionError:
        return render_template("index.html",
            error="API server is not running. Start it with: bash run.sh",
            active_tab="heart")
    except Exception as e:
        return render_template("index.html", error=str(e), active_tab="heart")


@app.route("/predict/diabetes", methods=["POST"])
def diabetes_predict():
    d = request.form.to_dict()
    pid = d.get("patient_id", "")
    payload = {
        "pregnancies":    float(d["pregnancies"]),
        "glucose":        float(d["glucose"]),
        "blood_pressure": float(d["blood_pressure"]),
        "skin_thickness": float(d["skin_thickness"]),
        "insulin":        float(d["insulin"]),
        "bmi":            float(d["bmi"]),
        "dpf":            float(d["dpf"]),
        "age":            float(d["age"]),
    }
    try:
        result = _api_post("/predict/diabetes", payload, pid)
        return render_template("index.html", diabetes_result=result, active_tab="diabetes")
    except requests.exceptions.ConnectionError:
        return render_template("index.html",
            error="API server is not running. Start it with: bash run.sh",
            active_tab="diabetes")
    except Exception as e:
        return render_template("index.html", error=str(e), active_tab="diabetes")


@app.route("/predict/cancer", methods=["POST"])
def cancer_predict():
    d = request.form.to_dict()
    pid = d.get("patient_id", "")
    payload = {
        "radius":     float(d["radius"]),
        "texture":    float(d["texture"]),
        "perimeter":  float(d["perimeter"]),
        "area":       float(d["area"]),
        "smoothness": float(d["smoothness"]),
    }
    try:
        result = _api_post("/predict/cancer", payload, pid)
        return render_template("index.html", cancer_result=result, active_tab="cancer")
    except requests.exceptions.ConnectionError:
        return render_template("index.html",
            error="API server is not running. Start it with: bash run.sh",
            active_tab="cancer")
    except Exception as e:
        return render_template("index.html", error=str(e), active_tab="cancer")


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
