import pickle
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn


MODEL_FILE = Path(__file__).parent / "model.bin"


class Patient(BaseModel):
    Age: int = Field(ge=0)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: int = Field(ge=0)
    Cholesterol: int = Field(ge=0)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=0)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float
    ST_Slope: Literal["Up", "Flat", "Down"]


class PredictResponse(BaseModel):
    heart_disease_probability: float
    heart_disease: bool


app = FastAPI(title="heart-failure-prediction")

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


@app.get("/")
def root():
    return {
        "message": "Heart disease prediction API",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
    }


def predict_single(patient: dict) -> float:
    return float(model.predict_proba([patient])[0, 1])


@app.post("/predict", response_model=PredictResponse)
def predict(patient: Patient) -> PredictResponse:
    probability = predict_single(patient.model_dump())
    return PredictResponse(
        heart_disease_probability=probability,
        heart_disease=probability >= 0.5,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
