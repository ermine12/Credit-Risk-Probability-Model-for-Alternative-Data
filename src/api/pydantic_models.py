from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    prob_default: float
