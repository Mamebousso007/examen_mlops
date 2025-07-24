from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    ALLOWED_ORIGINS: list = ["*"]
    MODEL_PATH: str = "./models/house_price_model.pkl"

settings = Settings()
