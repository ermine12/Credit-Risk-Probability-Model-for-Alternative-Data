from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float


class PredictionResponse(BaseModel):
    risk_probability: float
