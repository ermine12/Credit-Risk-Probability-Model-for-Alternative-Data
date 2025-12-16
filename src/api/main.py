import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk API")

# --- MLflow Model Loading ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "production"

try:
    model = mlflow.sklearn.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Accepts customer data and returns the credit risk probability.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Please check MLflow setup.")

    input_df = pd.DataFrame([request.dict()])

    risk_probability = model.predict_proba(input_df)[:, 1][0]

    return PredictionResponse(risk_probability=risk_probability)
