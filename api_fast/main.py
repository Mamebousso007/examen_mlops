"""
API FastAPI pour la prédiction des prix immobiliers
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time
import logging
from contextlib import asynccontextmanager
from api_fast.router import router

from api_fast.model import HousePriceModel
from config import settings
from logger.logging_config import setup_logging

# Configuration du logging
setup_logging()
logger = logging.getLogger(__name__)

# Instance globale du modèle
model_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    global model_instance
    
    # Startup
    logger.info("🚀 Démarrage de l'API House Price Prediction")
    try:
        model_instance = HousePriceModel()
        model_instance.load_model()
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt de l'API")

# Création de l'application FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="API de prédiction des prix immobiliers utilisant le machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Middleware pour le logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware pour logger toutes les requêtes"""
    start_time = time.time()
    
    # Log de la requête entrante
    logger.info(
        "Requête reçue",
        extra={
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent", "")
        }
    )
    
    # Traitement de la requête
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Log de la réponse
        logger.info(
            "Requête traitée",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration": f"{duration:.3f}s"
            }
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Erreur lors du traitement de la requête",
            extra={
                "method": request.method,
                "url": str(request.url),
                "error": str(e),
                "duration": f"{duration:.3f}s"
            }
        )
        raise

# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'erreurs global"""
    logger.error(
        "Erreur non gérée",
        extra={
            "method": request.method,
            "url": str(request.url),
            "error": str(exc)
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "message": "Une erreur inattendue s'est produite"
        }
    )

# Route de santé
@app.get("/health")
async def health_check():
    """Endpoint de vérification de santé"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model_instance is not None
    }

# Inclusion des routes
app.include_router(router, prefix="/api/v1")

# Point d'entrée pour le développement
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )