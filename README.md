---
title: MedAI — Multi-Disease Prediction System
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🧬 MedAI — Multi-Disease Prediction System

> An end-to-end machine learning web application that predicts the risk of **Heart Disease**, **Diabetes**, and **Breast Cancer** from clinical inputs — with explainable AI built in.

---

## What This Project Does

A user enters clinical measurements into a web form. The system runs those values through a trained machine learning model and returns:

- A **prediction** (positive / negative)
- A **probability score** (0–100%)
- A **risk level** (Low / Medium / High)
- The **top 3 features** that drove the prediction, with plain-English explanations
- A **clinical recommendation**

Every prediction is saved to a local database so you can review history and statistics.

---

## Live Demo Flow

```
User opens browser → Selects a disease → Fills the form → Clicks Analyse
        ↓
  FastAPI backend receives the data
        ↓
  StandardScaler normalises the input
        ↓
  Random Forest model predicts probability
        ↓
  SHAP TreeExplainer explains the prediction
  (falls back to LIME if SHAP fails)
        ↓
  Risk level assigned, recommendation generated
        ↓
  Result logged to SQLite database
        ↓
  Flask frontend renders the result card
```

---

## Project Structure

```
Disease_Prediction/
│
├── app/                            # FastAPI backend
│   ├── __init__.py
│   ├── api/
│   │   └── prediction_api.py       # All API routes (/predict, /history, /stats)
│   ├── schemas/
│   │   └── request_models.py       # Pydantic input validation models
│   └── services/
│       ├── inference_service.py    # Scale → Model → Explain pipeline
│       ├── explainability_service.py  # SHAP + LIME explainers
│       └── db_service.py           # SQLite logging and history
│
├── web/                            # Flask frontend
│   ├── app.py                      # Flask routes (proxies to FastAPI)
│   ├── templates/
│   │   ├── index.html              # Main prediction page
│   │   └── history.html            # Patient history page
│   └── static/
│       └── style.css               # Dark-theme UI styles
│
├── models/
│   ├── trained_models/             # Saved .pkl model files (RF, SVM, XGBoost)
│   └── scalers/                    # Saved StandardScaler .pkl files
│
├── data/
│   ├── raw/                        # Original CSV datasets
│   ├── processed/                  # Scaled .npy arrays (train/test splits)
│   └── predictions.db              # SQLite prediction history
│
├── notebooks/                      # Jupyter notebooks (full ML pipeline)
│   ├── data_loading.ipynb
│   ├── eda_preprocessing.ipynb
│   ├── feature_engineering_pipeline.ipynb
│   ├── model_training.ipynb
│   ├── hyperparameter_tuning.ipynb
│   ├── model_evaluation.ipynb
│   ├── explainable_ai.ipynb
│   └── unified_prediction_engine.ipynb
│
├── requirements.txt
└── run.sh                          # One-command startup script
```

---

## Diseases & Datasets

| Disease | Dataset | Input Features | Production Model |
|---|---|---|---|
| Heart Disease | Cleveland Heart Disease (UCI) | 13 clinical features | Random Forest |
| Diabetes | Pima Indians Diabetes (Kaggle) | 8 metabolic features | Random Forest |
| Breast Cancer | Wisconsin Breast Cancer (sklearn) | 5 key cell features + 25 baseline means | Random Forest |

---

## Model Performance

Four algorithms were trained and compared for each disease. Random Forest was selected for production based on the best balance of accuracy and recall (minimising false negatives matters most in medical screening).

### Heart Disease

| Model | Accuracy | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 79.5% | — | — |
| **Random Forest ✅** | **98.5%** | **0.97** | **0.985** |
| SVM | 88.8% | — | — |
| XGBoost | 98.5% | — | — |

### Diabetes

| Model | Accuracy | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 75.3% | — | — |
| **Random Forest ✅** | **77.3%** | **0.65** | **0.649** |
| SVM | 74.7% | — | — |
| XGBoost | 71.4% | — | — |

### Breast Cancer

| Model | Accuracy | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 97.4% | — | — |
| **Random Forest ✅** | **96.5%** | **0.99** | **0.972** |
| SVM | 98.2% | — | — |
| XGBoost | 95.6% | — | — |

> Diabetes scores lower because the Pima Indians dataset is small (~768 rows) and noisy. ~77% is expected and consistent with published benchmarks.

---

## Explainability — SHAP & LIME

This project implements two explainability methods so every prediction comes with a reason, not just a number.

### SHAP (Primary)
**SHapley Additive exPlanations** — uses game theory to calculate exactly how much each feature pushed the prediction up or down for *this specific patient*. Uses `shap.TreeExplainer` which is exact (not approximate) for Random Forest.

### LIME (Fallback)
**Local Interpretable Model-Agnostic Explanations** — builds a simple linear model around the prediction point to approximate feature importance. Used automatically if SHAP fails.

### What the explanation shows
For each of the top 3 features:
- **Feature name** and the patient's actual value (e.g., `Glucose = 148.0`)
- **Impact bar** — how strongly it influenced the result
- **Direction** — ▲ increases risk / ▼ decreases risk
- **Why** — a plain-English clinical sentence explaining the feature's significance
- **Method badge** — `SHAP` or `LIME` so you always know which explainer ran

---

## API Reference

Base URL: `http://127.0.0.1:8000`  
Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict/heart` | Heart disease prediction |
| `POST` | `/predict/diabetes` | Diabetes prediction |
| `POST` | `/predict/cancer` | Breast cancer prediction |
| `GET` | `/history?disease=&limit=20` | Recent predictions |
| `GET` | `/stats` | Aggregate counts by disease and risk level |

### Example — Diabetes Prediction

**Request**
```bash
curl -X POST http://127.0.0.1:8000/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "dpf": 0.627,
    "age": 50
  }'
```

**Response**
```json
{
  "prediction": 1,
  "probability": 0.7321,
  "risk_level": "High Risk",
  "explainer_method": "SHAP",
  "explanation": [
    {
      "feature_name": "Glucose",
      "feature_index": 1,
      "impact": 0.1899,
      "abs_impact": 0.1899,
      "direction": "increases",
      "why": "High blood glucose is the primary diabetes marker.",
      "input_value": 148.0
    },
    {
      "feature_name": "BMI",
      "feature_index": 5,
      "impact": 0.0921,
      "abs_impact": 0.0921,
      "direction": "increases",
      "why": "Higher BMI is strongly linked to insulin resistance.",
      "input_value": 33.6
    },
    {
      "feature_name": "Age",
      "feature_index": 7,
      "impact": 0.0412,
      "abs_impact": 0.0412,
      "direction": "increases",
      "why": "Risk of type-2 diabetes increases with age.",
      "input_value": 50.0
    }
  ],
  "recommendation": "⚠️ Consult an endocrinologist. Possible indicators: high glucose level, BMI imbalance, or insulin issues."
}
```

---

## Input Fields

### Heart Disease (13 fields)

| Field | Type | Range | Description |
|---|---|---|---|
| `age` | float | 20–100 | Age in years |
| `sex` | int | 0–1 | 0 = Female, 1 = Male |
| `cp` | int | 0–3 | Chest pain type |
| `trestbps` | float | 80–220 | Resting blood pressure (mmHg) |
| `chol` | float | 100–700 | Serum cholesterol (mg/dl) |
| `fbs` | int | 0–1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | int | 0–2 | Resting ECG results |
| `thalach` | float | 40–250 | Max heart rate achieved |
| `exang` | int | 0–1 | Exercise-induced angina |
| `oldpeak` | float | 0–10 | ST depression (exercise vs rest) |
| `slope` | int | 0–2 | Slope of peak exercise ST segment |
| `ca` | int | 0–3 | Number of major vessels |
| `thal` | int | 0–3 | Thalassemia type |

### Diabetes (8 fields)

| Field | Type | Range | Description |
|---|---|---|---|
| `pregnancies` | float | 0–20 | Number of pregnancies |
| `glucose` | float | 50–300 | Plasma glucose (mg/dl) |
| `blood_pressure` | float | 0–200 | Diastolic blood pressure (mmHg) |
| `skin_thickness` | float | 0–100 | Triceps skinfold thickness (mm) |
| `insulin` | float | 0–1000 | 2-hour serum insulin (μU/ml) |
| `bmi` | float | 0–100 | Body mass index |
| `dpf` | float | 0–5 | Diabetes pedigree function |
| `age` | float | 1–120 | Age in years |

### Breast Cancer (5 fields)

| Field | Type | Range | Description |
|---|---|---|---|
| `radius` | float | 0–50 | Mean radius of cell nuclei |
| `texture` | float | 0–50 | Mean texture (std dev of gray-scale) |
| `perimeter` | float | 0–300 | Mean perimeter of cell nuclei |
| `area` | float | 0–3000 | Mean area of cell nuclei |
| `smoothness` | float | 0–1 | Mean smoothness |

---

## Setup & Running

### Prerequisites
- Python 3.10 or higher
- `venv` module (comes with Python)

### 1. Clone / navigate to the project

```bash
cd "Disease_Prediction copy"
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run both servers with one command

```bash
bash run.sh
```

This starts:
- **FastAPI backend** → `http://127.0.0.1:8000`
- **Flask frontend** → `http://127.0.0.1:5000`

Open `http://127.0.0.1:5000` in your browser.

### 5. Stop

Press `Ctrl+C` in the terminal. Both servers shut down cleanly.

### Manual startup (two terminals)

```bash
# Terminal 1 — API
venv/bin/uvicorn app.api.prediction_api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Web UI
venv/bin/python web/app.py
```

---

## Notebooks

The `notebooks/` folder documents the complete ML pipeline from raw data to deployed model. Run them in order to reproduce everything from scratch.

```bash
source venv/bin/activate
jupyter lab
```

| Notebook | What it does |
|---|---|
| `data_loading.ipynb` | Load and inspect the three raw CSV datasets |
| `eda_preprocessing.ipynb` | EDA, missing value handling, outlier detection, scaling, train/test split |
| `feature_engineering_pipeline.ipynb` | Feature selection and engineering steps |
| `model_training.ipynb` | Train Logistic Regression, Random Forest, SVM, XGBoost — compare results |
| `hyperparameter_tuning.ipynb` | GridSearchCV / RandomizedSearchCV with 5-fold cross-validation |
| `model_evaluation.ipynb` | Confusion matrices, classification reports, ROC curves |
| `explainable_ai.ipynb` | SHAP value exploration, feature importance visualisation |
| `unified_prediction_engine.ipynb` | End-to-end prediction pipeline walkthrough |

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML models | scikit-learn, XGBoost | Random Forest, SVM, Logistic Regression, XGBoost |
| Explainability | SHAP, LIME | Per-prediction feature importance |
| API | FastAPI + Uvicorn | REST backend with auto-generated Swagger docs |
| Frontend | Flask + Jinja2 | Web UI with form inputs and result cards |
| Data | NumPy, Pandas | Data loading, processing, array operations |
| Persistence | SQLite (stdlib) + joblib | Prediction history, model serialisation |
| Notebooks | JupyterLab | Exploratory analysis and model development |

---

## Data Sources

- **Heart Disease** — [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Diabetes** — [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Breast Cancer** — [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

---

## Key Design Decisions

**Why Random Forest over SVM or XGBoost?**
Random Forest gave the best recall across all three diseases. In medical screening, a false negative (missing a real case) is more dangerous than a false positive, so recall is the primary metric. RF also supports SHAP TreeExplainer natively, which gives exact (not approximate) explanations.

**Why SHAP + LIME?**
SHAP is the gold standard for tree models — it's mathematically exact. LIME is model-agnostic and acts as a fallback. Having both means the explainability layer never silently fails.

**Why two servers (FastAPI + Flask)?**
FastAPI handles the ML inference and exposes a clean REST API with auto-generated docs. Flask handles the HTML rendering. This separation means the API can be consumed independently (e.g., by a mobile app or another service) without touching the frontend.

---

> **Disclaimer:** This project is for educational and research purposes only. It is not a substitute for professional medical diagnosis or advice.
