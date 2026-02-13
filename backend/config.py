from pydantic_settings import BaseSettings
from typing import Optional
from pydantic import model_validator


class Settings(BaseSettings):
    GEMINI_API_KEY: str

    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_RAW_TOPIC: str = "sensor.raw"
    KAFKA_CLEAN_TOPIC: str = "sensor.cleaned"
    KAFKA_EVENT_TOPIC: str = "events.queue"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "water_watch_vectors"
    QDRANT_VECTOR_SIZE: int = 384

    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM: str = "waterwatch@gmail.com"
    SMTP_TO: str = "admin@waterwatch.com"

    RISK_HIGH_THRESHOLD: float = 0.7
    RISK_MEDIUM_THRESHOLD: float = 0.4

    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    ALERT_DEDUP_TTL_SECONDS: int = 900  # 15 min

    @model_validator(mode="after")
    def validate_smtp(self):
        creds = [self.SMTP_USER, self.SMTP_PASSWORD]
        if any(creds) and not all(creds):
            raise ValueError("Incomplete SMTP credentials")
        return self

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
