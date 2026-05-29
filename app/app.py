from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import requests

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

API_URL = "http://127.0.0.1:8000"



# HOME PAGE

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# HEART FORM PAGE
f
@app.get("/heart", response_class=HTMLResponse)
def heart_page(request: Request):
    return templates.TemplateResponse("heart.html", {"request": request})


@app.post("/heart/predict", response_class=HTMLResponse)
def heart_predict(request: Request,
                  age: float = Form(...),
                  sex: int = Form(...),
                  cp: int = Form(...),
                  trestbps: float = Form(...),
                  chol: float = Form(...),
                  fbs: int = Form(...),
                  restecg: int = Form(...),
                  thalach: float = Form(...),
                  exang: int = Form(...),
                  oldpeak: float = Form(...),
                  slope: int = Form(...),
                  ca: int = Form(...),
                  thal: int = Form(...)):

    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    res = requests.post(f"{API_URL}/predict/heart", json=payload).json()

    return templates.TemplateResponse("heart.html", {
        "request": request,
        "result": res
    })