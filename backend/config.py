"""
Configuration management for Water Watch Backend
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Gemini API
    GEMINI_API_KEY: str
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_RAW_TOPIC: str = "sensor.raw"
    KAFKA_CLEAN_TOPIC: str = "sensor.cleaned"
    KAFKA_EVENT_TOPIC: str = "events.queue"
    
    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "water_watch_vectors"
    QDRANT_VECTOR_SIZE: int = 384
    
    # SMTP Configuration
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM: str = "waterwatch@gmail.com"
    SMTP_TO: str = "admin@waterwatch.com"
    
    # Risk Thresholds
    RISK_HIGH_THRESHOLD: float = 0.7
    RISK_MEDIUM_THRESHOLD: float = 0.4
    
    # FastAPI Configuration
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = 'allow'  # Allow extra fields from .env


# Global settings instance
settings = Settings()
