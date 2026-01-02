from pydantic import BaseModel
from typing import List

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class CropRecommendation(BaseModel):
    crop: str
    confidence: float

class PredictionResponse(BaseModel):
    recommendations: List[CropRecommendation]
