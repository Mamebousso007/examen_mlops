from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path

class Settings(BaseSettings):
    # === Environnement général ===
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # === API ===
    APP_NAME: str = "House Price Prediction API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API de prédiction des prix immobiliers"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # === CORS ===
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080"
    ]

    # === Chemins ===
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    MODEL_PATH: str = "models/house_price_model.pkl"

    # === Logging ===
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/api.log"
    ENABLE_LOG_FILE: bool = True
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # === Validation ===
    MIN_BEDROOMS: int = 0
    MAX_BEDROOMS: int = 20
    MIN_BATHROOMS: float = 0.0
    MAX_BATHROOMS: float = 20.0
    MIN_SQFT_LIVING: int = 100
    MAX_SQFT_LIVING: int = 50000
    MIN_SQFT_LOT: int = 100
    MAX_SQFT_LOT: int = 1000000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Crée une instance globale de la configuration
settings = Settings()
