---
title: MedAI — Multi-Disease Prediction System
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

[ Live Demo: https://huggingface.co/spaces/saansho/Disease_Prediction_System ]


# 🧬 MedAI — Disease Prediction System

A web app that uses Machine Learning to predict the risk of **Heart Disease**, **Diabetes**, and **Breast Cancer** from clinical inputs — and explains *why* it made each prediction.

---

## 🤔 What does this app do?

You enter some health measurements (like blood pressure, glucose level, etc.), and the app tells you:

- **Prediction** — Positive or Negative for the disease
- **Probability** — How confident the model is (e.g. 73%)
- **Risk Level** — Low / Medium / High
- **Top 3 reasons** — Which health values influenced the prediction and how
- **Recommendation** — What to do next

Every prediction is also saved to a history log so you can review past results.

---

## 🖥️ How it works (simple version)

```
# Live Demo Flow

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

## 📁 Project Structure

```
Disease_Prediction_Ststem/
│
├── app/                        ← Backend (FastAPI)
│   ├── api/
│   │   └── prediction_api.py   ← All API routes
│   ├── schemas/
│   │   └── request_models.py   ← Input validation
│   └── services/
│       ├── inference_service.py    ← Scale → Predict pipeline
│       ├── explainability_service.py  ← SHAP/LIME explanations
│       └── db_service.py           ← Save/fetch history (SQLite)
│
├── web/                        ← Frontend (Flask)
│   ├── app.py                  ← Flask routes
│   └── templates/
│       ├── index.html          ← Main prediction page
│       └── history.html        ← Prediction history page
│
├── models/
│   ├── trained_models/         ← Saved ML models (.pkl files)
│   └── scalers/                ← Saved data scalers (.pkl files)
│
├── data/
│   ├── raw/                    ← Original CSV datasets
│   ├── processed/              ← Preprocessed numpy arrays
│   └── predictions.db          ← SQLite history database
│
├── notebooks/                  ← Jupyter notebooks (full ML pipeline)
│
├── requirements.txt            ← Python dependencies
├── Dockerfile                  ← Docker setup for deployment
├── run.sh                      ← One-command local startup
└── start.sh                    ← Docker/HuggingFace startup
```

---

## 🚀 Running Locally (Step by Step)

### Prerequisites
- Python 3.11 or higher
- Git

### Step 1 — Clone or navigate to the project

```bash
cd "Disease_Prediction_Ststem"
```

### Step 2 — Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate        # Mac/Linux
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Start the app

```bash
bash run.sh
```

This starts:
- **FastAPI backend** → `http://127.0.0.1:8000`
- **Flask frontend** → `http://127.0.0.1:5001`

Open `http://127.0.0.1:5001` in your browser to use the app.

### Step 5 — Stop

Press `Ctrl+C` in the terminal.

---

## 🌐 API Endpoints

The FastAPI backend exposes these endpoints:

| Method | Endpoint | What it does |
|--------|----------|--------------|
| `GET` | `/health` | Check if the server is running |
| `POST` | `/predict/heart` | Predict heart disease risk |
| `POST` | `/predict/diabetes` | Predict diabetes risk |
| `POST` | `/predict/cancer` | Predict breast cancer risk |
| `GET` | `/history` | Get recent predictions |
| `GET` | `/stats` | Get summary statistics |

Interactive docs (auto-generated): `http://127.0.0.1:8000/docs`

---

## 📊 What health values do I enter?

### ❤️ Heart Disease (13 values)

| Field | What it means | Example |
|-------|--------------|---------|
| `age` | Age in years | 55 |
| `sex` | 0 = Female, 1 = Male | 1 |
| `cp` | Chest pain type (0–3) | 0 |
| `trestbps` | Resting blood pressure (mmHg) | 130 |
| `chol` | Cholesterol (mg/dl) | 250 |
| `fbs` | Fasting blood sugar > 120? (0/1) | 0 |
| `restecg` | Resting ECG result (0–2) | 1 |
| `thalach` | Max heart rate achieved | 150 |
| `exang` | Exercise-induced chest pain? (0/1) | 0 |
| `oldpeak` | ST depression (0–6) | 1.5 |
| `slope` | Slope of ST segment (0–2) | 1 |
| `ca` | Number of major vessels (0–3) | 1 |
| `thal` | Thalassemia type (0–3) | 2 |

### 🩸 Diabetes (8 values)

| Field | What it means | Example |
|-------|--------------|---------|
| `pregnancies` | Number of pregnancies | 2 |
| `glucose` | Plasma glucose (mg/dl) | 148 |
| `blood_pressure` | Diastolic blood pressure | 72 |
| `skin_thickness` | Triceps skinfold (mm) | 35 |
| `insulin` | 2-hour serum insulin | 0 |
| `bmi` | Body mass index | 33.6 |
| `dpf` | Diabetes pedigree function | 0.627 |
| `age` | Age in years | 50 |

### 🎗 Breast Cancer (5 values)

| Field | What it means | Example |
|-------|--------------|---------|
| `radius` | Mean cell nucleus radius | 14.0 |
| `texture` | Mean texture | 19.0 |
| `perimeter` | Mean cell perimeter | 92.0 |
| `area` | Mean cell area | 655 |
| `smoothness` | Mean smoothness | 0.096 |

---

## 🤖 Machine Learning Models

Four algorithms were trained for each disease. **Random Forest** was selected for production because it gave the best balance of accuracy and recall.

| Disease | Model | Accuracy |
|---------|-------|----------|
| Heart Disease | Random Forest | ~98.5% |
| Diabetes | Random Forest | ~77% |
| Breast Cancer | Random Forest | ~96.5% |

> Diabetes scores lower because the dataset is small (~768 rows). ~77% is normal and consistent with published research.

---

## 💡 How are predictions explained?

Every prediction comes with an explanation using **SHAP** (primary) or **LIME** (fallback):

- **SHAP** — Mathematically exact method. Shows exactly how much each health value pushed the prediction up or down.
- **LIME** — Model-agnostic approximation. Used automatically if SHAP fails.

The top 3 most influential health values are shown with:
- The actual value you entered
- Whether it increases or decreases risk (▲ / ▼)
- A plain-English explanation of why that value matters

---

## 🛠️ Tech Stack

| Part | Technology |
|------|-----------|
| ML Models | scikit-learn, XGBoost |
| Explainability | SHAP, LIME |
| Backend API | FastAPI + Uvicorn |
| Frontend | Flask + Jinja2 |
| Data processing | NumPy, Pandas |
| History storage | SQLite |
| Notebooks | JupyterLab |

---

## 📚 Datasets Used

- **Heart Disease** — Cleveland Heart Disease Dataset (UCI)
- **Diabetes** — Pima Indians Diabetes Dataset (Kaggle)
- **Breast Cancer** — Wisconsin Breast Cancer Dataset (sklearn)

---

## 🐳 Deploying to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the SDK
3. Push this repository:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
git push space main
```

HuggingFace will build the Docker image and host the app. The Flask UI will be available at your Space URL.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis or advice**.
Always consult a qualified healthcare professional.
