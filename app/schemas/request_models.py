
from pydantic import BaseModel, Field

class HeartInput(BaseModel):
    age:      float = Field(..., ge=1,   le=120,  description="Age in years")
    sex:      int   = Field(..., ge=0,   le=1)
    cp:       int   = Field(..., ge=0,   le=3,    description="Chest pain type 0-3")
    trestbps: float = Field(..., ge=80,  le=220,  description="Resting blood pressure")
    chol:     float = Field(..., ge=100, le=700,  description="Serum cholesterol mg/dl")
    fbs:      int   = Field(..., ge=0,   le=1)
    restecg:  int   = Field(..., ge=0,   le=2)
    thalach:  float = Field(..., ge=40,  le=250,  description="Max heart rate achieved")
    exang:    int   = Field(..., ge=0,   le=1)
    oldpeak:  float = Field(..., ge=0.0, le=10.0)
    slope:    int   = Field(..., ge=0,   le=2)
    ca:       int   = Field(..., ge=0,   le=4)
    thal:     int   = Field(..., ge=0,   le=7)


class DiabetesInput(BaseModel):
    pregnancies: float = Field(..., ge=0,   le=20)
    glucose:     float = Field(..., ge=0,   le=300)
    blood_pressure: float = Field(..., ge=0, le=200)
    skin_thickness: float = Field(..., ge=0, le=100)
    insulin:     float = Field(..., ge=0,   le=1000)
    bmi:         float = Field(..., ge=0.0, le=100.0)
    dpf:         float = Field(..., ge=0.0, le=5.0,  description="Diabetes pedigree function")
    age:         float = Field(..., ge=1,   le=120)


class CancerInput(BaseModel):
    radius:     float = Field(..., ge=0.0, le=50.0)
    texture:    float = Field(..., ge=0.0, le=50.0)
    perimeter:  float = Field(..., ge=0.0, le=300.0)
    area:       float = Field(..., ge=0.0, le=3000.0)
    smoothness: float = Field(..., ge=0.0, le=1.0)