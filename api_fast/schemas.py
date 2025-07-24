"""
Schémas Pydantic pour l'API
"""

from pydantic import BaseModel
from typing import List, Dict, Optional

class HousePredictionRequest(BaseModel):
    bedrooms: int
    bathrooms: int
    sqft_living: float
    city: str

class HousePredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: Dict[str, float]
    model_version: str
    processing_time: float

class BatchPredictionRequest(BaseModel):
    houses: List[HousePredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[HousePredictionResponse]
    batch_size: int
    processing_time: float
    model_version: str

class ModelInfoResponse(BaseModel):
    version: str
    name: str
    description: Optional[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    model_loaded: bool

class ValidationResponse(BaseModel):
    is_valid: bool
    message: str
    input_data: dict
