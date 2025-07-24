"""
Routes de l'API FastAPI pour la prédiction des prix immobiliers
"""
import logging
import time
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends


from .schemas import (
    HousePredictionRequest,
    HousePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse
)

from api_fast.model import HousePriceModel


# Configuration du logger
logger = logging.getLogger("house_price_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Créer le routeur
router = APIRouter(tags=["predictions"])

# Dépendance pour injecter le modèle
def get_model() -> HousePriceModel:
    try:
        from main import model_instance
        if model_instance is None:
            raise ValueError("Le modèle n'a pas encore été chargé")
        return model_instance
    except ImportError:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

# ---------------------- Prédiction individuelle ----------------------

@router.post("/predict", response_model=HousePredictionResponse)
async def predict_house_price(
    request: HousePredictionRequest,
    model: HousePriceModel = Depends(get_model)
):
    start_time = time.time()

    try:
        input_df = pd.DataFrame([request.dict()])
        prediction = model.predict(input_df)
        prediction_value = float(prediction[0])
        duration = time.time() - start_time

        logger.info(f"Prédiction réussie en {duration:.3f}s : {prediction_value}")

        return HousePredictionResponse(
            predicted_price=prediction_value,
            confidence_interval={
                "lower": round(prediction_value * 0.9, 2),
                "upper": round(prediction_value * 1.1, 2),
            },
            model_version=model.get_model_info()["version"],
            processing_time=round(duration, 3)
        )

    except Exception as e:
        logger.error(f"Erreur de prédiction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# ---------------------- Prédictions en batch ----------------------

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model: HousePriceModel = Depends(get_model)
):
    start_time = time.time()

    try:
        if len(request.houses) > 100:
            raise HTTPException(status_code=400, detail="Limite de 100 prédictions par lot")

        input_df = pd.DataFrame([house.dict() for house in request.houses])
        predictions = model.predict(input_df)

        results = [
            {
                "house_id": idx,
                "predicted_price": float(pred),
                "input_features": house.dict()
            }
            for idx, (house, pred) in enumerate(zip(request.houses, predictions))
        ]

        duration = time.time() - start_time

        return BatchPredictionResponse(
            predictions=results,
            batch_size=len(results),
            processing_time=round(duration, 3),
            model_version=model.get_model_info()["version"]
        )

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lot: {str(e)}")

# ---------------------- Informations du modèle ----------------------

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(model: HousePriceModel = Depends(get_model)):
    try:
        return ModelInfoResponse(**model.get_model_info())
    except Exception as e:
        logger.error(f"Erreur model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur info modèle")

# ---------------------- Liste des features ----------------------

@router.get("/model/features")
async def get_model_features(model: HousePriceModel = Depends(get_model)):
    try:
        features = model.get_expected_features()
        return {
            "features": features,
            "feature_count": len(features)
        }
    except Exception as e:
        logger.error(f"Erreur features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur récupération features")

# ---------------------- Validation des données ----------------------

@router.post("/model/validate")
async def validate_input(
    request: HousePredictionRequest,
    model: HousePriceModel = Depends(get_model)
):
    try:
        input_df = pd.DataFrame([request.dict()])
        is_valid, message = model.validate_input(input_df)
        return {
            "is_valid": is_valid,
            "message": message,
            "input_data": request.dict()
        }
    except Exception as e:
        logger.error(f"Erreur validation input: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur validation: {str(e)}")
