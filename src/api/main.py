from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictRequest, PredictResponse

app = FastAPI(title="Credit Risk API")

DEFAULT_MODEL_PATH = Path("models/model.joblib")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found. Train and save a model to models/model.joblib")

    model = joblib.load(DEFAULT_MODEL_PATH)

    # The pipeline expects a DataFrame, so we convert the request features
    df = pd.DataFrame([req.dict()["features"]])

    # The pipeline handles all preprocessing
    prob_default = float(model.predict_proba(df)[:, 1][0])
    return PredictResponse(prob_default=prob_default)
