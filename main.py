from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from preprocess import preprocess_input
import shap
import numpy as np 
np.bool = bool # Fix for SHAP compatibility with NumPy 1.24+

app = FastAPI()
model = joblib.load("mental_health_model.pkl")
explainer = shap.Explainer(model)
class SurveyInput(BaseModel):
    Age: int
    Gender: str
    self_employed: str
    family_history: str
    work_interfere: str
    no_employees: str
    remote_work: str
    tech_company: str
    benefits: str
    care_options: str
    wellness_program: str
    seek_help: str
    anonymity: str
    leave: str
    mental_health_consequence: str
    phys_health_consequence: str
    coworkers: str
    supervisor: str
    mental_health_interview: str
    phys_health_interview: str
    mental_vs_physical: str
    obs_consequence: str

@app.post("/predict")
def predict(data: SurveyInput):
    input_df = preprocess_input(data.dict())
    prediction = model.predict(input_df)[0]
    shap_vals = explainer(input_df)
    return {
        "prediction": int(prediction),
        "shap_values": shap_vals.values[0].tolist()
    }
@app.get("/") 
def root():
    return {"message": "✅ FastAPI is running. Use /predict to POST survey data."}