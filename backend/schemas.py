"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


# ============ Sensor Data Schemas ============

class SensorReading(BaseModel):
    site_id: str
    salinity: Optional[float] = None
    dissolved_oxygen: Optional[float] = None
    ph: Optional[float] = None
    secchi_depth: Optional[float] = None
    water_depth: Optional[float] = None
    water_temp: Optional[float] = None
    air_temp: Optional[float] = None
    timestamp: Optional[datetime] = None


class SensorStatus(BaseModel):
    total_readings: int
    last_reading: Optional[datetime]
    active_sites: List[str]
    kafka_lag: int


# ============ Multimodal Input Schemas ============

class MultimodalUpload(BaseModel):
    type: Literal["image", "video", "audio", "text"]
    content: str  # base64 encoded or text
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    upload_id: str
    status: str
    message: str
    embedding_id: Optional[str] = None


# ============ Risk & Forecasting Schemas ============


class TrendPoint(BaseModel):
    timestamp: datetime
    value: float

class RiskPrediction(BaseModel):
    ph_trend: str
    do_trend: str
    salinity_trend: str
    ph_history: List[TrendPoint] = []
    do_history: List[TrendPoint] = []
    salinity_history: List[TrendPoint] = []


class RiskForecast(BaseModel):
    site_id: str
    risk_level: Literal["high", "medium", "low", "unknown"]
    risk_score: float
    predictions: RiskPrediction
    recommendations: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RiskHistoryItem(BaseModel):
    timestamp: datetime
    risk_score: float
    risk_level: str


class RiskHistory(BaseModel):
    site_id: str
    period: str
    history: List[RiskHistoryItem]


# ============ Alert Schemas ============

class AlertRequest(BaseModel):
    site_id: str
    risk_level: str
    recipient: Optional[str] = None


class AlertHistoryItem(BaseModel):
    id: str
    timestamp: datetime
    site_id: str
    risk_level: str
    sent_to: str
    status: str


class AlertHistory(BaseModel):
    alerts: List[AlertHistoryItem]


# ============ RAG Chatbot Schemas ============

class ChatQuery(BaseModel):
    query: str
    context_limit: int = 5


class ChatSource(BaseModel):
    timestamp: datetime
    content: str
    score: float


class ChatResponse(BaseModel):
    response: str
    sources: List[ChatSource]
    query_id: str


class ChatHistoryItem(BaseModel):
    query: str
    response: str
    timestamp: datetime


class ChatHistory(BaseModel):
    conversations: List[ChatHistoryItem]


# ============ Health Check Schemas ============

class ServiceStatus(BaseModel):
    kafka: str
    qdrant: str
    smtp: str


class HealthCheck(BaseModel):
    status: str
    services: ServiceStatus
    uptime: int
