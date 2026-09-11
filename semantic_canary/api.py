from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .model import load_model
from .canaries import SemanticCanary

app = FastAPI(title="SemanticCanary", version="0.1.0")

MODEL_PATH = "artifacts/model.joblib"
model = None

class Claim(BaseModel):
    claim_type: str
    state: str
    policyholder_tenure_years: float = Field(ge=0)
    previous_claims_count: float = Field(ge=0)
    claim_amount: float = Field(gt=0)
    deductible: float = Field(ge=0)
    days_between_incident_and_filing: float = Field(ge=0)

@app.on_event("startup")
def startup():
    global model
    if Path(MODEL_PATH).exists():
        model = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(claim: Claim):
    if model is None:
        return {"error": "model not trained"}
    df = pd.DataFrame([claim.model_dump()])
    p = float(model.predict_proba(df)[:,1][0])
    return {"fraud_probability": p}
