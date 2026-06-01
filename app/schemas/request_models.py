from pydantic import BaseModel, Field


class HeartInput(BaseModel):
    age:      float = Field(..., ge=1,   le=120,  description="Age in years")
    sex:      int   = Field(..., ge=0,   le=1,    description="0 = Female, 1 = Male")
    cp:       int   = Field(..., ge=0,   le=3,    description="Chest pain type (0–3)")
    trestbps: float = Field(..., ge=80,  le=220,  description="Resting blood pressure (mmHg)")
    chol:     float = Field(..., ge=100, le=700,  description="Serum cholesterol (mg/dl)")
    fbs:      int   = Field(..., ge=0,   le=1,    description="Fasting blood sugar > 120 mg/dl")
    restecg:  int   = Field(..., ge=0,   le=2,    description="Resting ECG results (0–2)")
    thalach:  float = Field(..., ge=40,  le=250,  description="Max heart rate achieved")
    exang:    int   = Field(..., ge=0,   le=1,    description="Exercise-induced angina")
    oldpeak:  float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope:    int   = Field(..., ge=0,   le=2,    description="Slope of peak exercise ST segment")
    ca:       int   = Field(..., ge=0,   le=3,    description="Number of major vessels (0–3)")
    thal:     int   = Field(..., ge=0,   le=3,    description="Thalassemia (0=Normal, 1=Fixed, 2=Reversible, 3=Unknown)")


class DiabetesInput(BaseModel):
    pregnancies:    float = Field(..., ge=0,   le=20)
    glucose:        float = Field(..., ge=0,   le=300)
    blood_pressure: float = Field(..., ge=0,   le=200)
    skin_thickness: float = Field(..., ge=0,   le=100)
    insulin:        float = Field(..., ge=0,   le=1000)
    bmi:            float = Field(..., ge=0.0, le=100.0)
    dpf:            float = Field(..., ge=0.0, le=5.0,  description="Diabetes pedigree function")
    age:            float = Field(..., ge=1,   le=120)


class CancerInput(BaseModel):
    radius:     float = Field(..., ge=0.0, le=50.0,   description="Mean radius of cell nuclei")
    texture:    float = Field(..., ge=0.0, le=50.0,   description="Mean texture (std dev of gray-scale values)")
    perimeter:  float = Field(..., ge=0.0, le=300.0,  description="Mean perimeter of cell nuclei")
    area:       float = Field(..., ge=0.0, le=3000.0, description="Mean area of cell nuclei")
    smoothness: float = Field(..., ge=0.0, le=1.0,    description="Mean smoothness (local variation in radius lengths)")
