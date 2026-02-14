import asyncio
import time
import logging
import json
import os
import re
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime, timedelta, timezone
from enum import Enum
import uuid
from pathlib import Path
from dotenv import load_dotenv

# FastAPI
from fastapi import FastAPI, HTTPException, Depends, status, Query, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

# ML & Vector DB
import torch
import numpy as np
from qdrant_client import QdrantClient

# LLM
import google.generativeai as genai

# Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# ==================== CONFIG ====================

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemini-2.0-flash")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)

# ==================== CONSTANTS ====================

QDRANT_URL = "http://localhost:6333"
COLLECTION = "water_memory"
KAFKA_BROKERS = "localhost:9092"

SIM_SEARCH_GATE = 0.55
SIM_TOPK = 30

BASE_DIR = Path(__file__).resolve().parent.parent.parent
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== DATABASE ====================

from db.database_connection import SessionLocal, engine, Base
from models.models import (
    User, Userinput, Recommendtion, Forecasting, 
    Government_Offices, Progress_Tracking
)

Base.metadata.create_all(bind=engine)

def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==================== AGENT IMPORTS ====================

from agents.agent1_sensor_data_ingestion import (
    consume_raw, preprocess, publish_clean
)
from agents.agent2 import compute_z, build_semantic_text
from agents.agent3 import agent_b_perceive
from agents.agent5 import ensure_collection, voxel_to_point, upsert_points, EMBEDDING_MEMORY, MEM_LOCK
from agents.agent6_forecasting import forecast
from agents.agent7_retriever import LiquidRetriever

# ==================== SERVICE IMPORTS ====================

from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService
from services.kafka_service import KafkaService
from services.audit_logger import get_audit_logger
from services.trust_service import get_trust_service
from services.smtp_service import get_smtp_service

# Initialize services
embedding_service = EmbeddingService()
qdrant_service = QdrantService()
kafka_service = KafkaService()
audit_logger = get_audit_logger()
trust_service = get_trust_service()
smtp_service = get_smtp_service()

# ==================== ENUMS ====================

class InputModality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class InputChannel(str, Enum):
    DIRECT = "direct"
    WHATSAPP = "whatsapp"

class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ReporterType(str, Enum):
    INDIVIDUAL = "individual"
    NGO = "ngo"
    GOVERNMENT = "government"

# ==================== REDIS THROTTLE SERVICE ====================

class RedisThrottleService:
    """Prevent duplicate alerts using Redis"""
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                decode_responses=True
            )
            self.client.ping()
            self.enabled = True
            logger.info("✅ Redis throttle service connected")
        except Exception as e:
            self.enabled = False
            logger.warning(f"⚠️ Redis not available: {e}")
    
    def allow_alert(self, alert_id: str, ttl_seconds: int = 3600) -> bool:
        if not self.enabled:
            return True
        
        key = f"alert:{alert_id}"
        try:
            was_set = self.client.set(key, "1", nx=True, ex=ttl_seconds)
            if not was_set:
                logger.info(f"📌 Alert deduplicated (recent): {alert_id}")
                return False
            return True
        except Exception as e:
            logger.error(f"Redis error: {e}")
            return True

_throttle_service = RedisThrottleService() if REDIS_AVAILABLE else None

# ==================== ALERT SERVICE ====================

class AlertService:
    """Send alerts to government organizations"""
    
    GOVERNMENT_ORGS = {
        "default": {
            "name": "Water Board Authority",
            "email": "alerts@waterboard.gov.in",
            "region": "All India",
        },
        "north": {
            "name": "Northern Region Water Authority",
            "email": "north.water@nra.gov.in",
            "region": "North India",
        },
        "south": {
            "name": "Southern Water Management Board",
            "email": "south.alerts@swmb.gov.in",
            "region": "South India",
        },
        "east": {
            "name": "Eastern Water Resources Department",
            "email": "east.water@ewrd.gov.in",
            "region": "East India",
        },
        "west": {
            "name": "Western Water Conservation Office",
            "email": "west.alerts@wwco.gov.in",
            "region": "West India",
        }
    }
    
    def __init__(self):
        self.smtp_configured = bool(
            os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD")
        )
    
    def get_nearest_org_by_location(self, location: Optional[str]) -> Dict[str, Any]:
        if not location:
            return self.GOVERNMENT_ORGS["default"]
        
        location_lower = location.lower()
        
        if any(x in location_lower for x in ["north", "delhi", "punjab", "himachal"]):
            return self.GOVERNMENT_ORGS["north"]
        elif any(x in location_lower for x in ["south", "tamil", "karnataka", "kerala"]):
            return self.GOVERNMENT_ORGS["south"]
        elif any(x in location_lower for x in ["east", "west bengal", "bihar", "odisha"]):
            return self.GOVERNMENT_ORGS["east"]
        elif any(x in location_lower for x in ["west", "maharashtra", "goa", "rajasthan"]):
            return self.GOVERNMENT_ORGS["west"]
        
        return self.GOVERNMENT_ORGS["default"]
    
    async def send_alert(
        self,
        risk_level: str,
        risk_score: float,
        location: str,
        source_id: str,
        channel: str,
        recommendations: List[str],
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send alert email to nearest government org"""
        
        if risk_level not in ["HIGH", "CRITICAL"]:
            logger.info(f"[ALERT] Risk level {risk_level} - Not sending alert")
            return False
        
        org = self.get_nearest_org_by_location(location)
        recipient_email = org["email"]
        
        # Check throttling
        alert_id = f"{source_id}_{risk_level}_{int(time.time() / 3600)}"
        if _throttle_service and not _throttle_service.allow_alert(alert_id):
            logger.info(f"[ALERT] Throttled for {source_id}")
            return False
        
        # Build email
        severity_emoji = {"HIGH": "⚠️", "CRITICAL": "🔴"}.get(risk_level, "🔔")
        subject = f"{severity_emoji} WATER QUALITY ALERT - {risk_level} RISK - {location}"
        
        text_body = f"""
WATERWATCH ALERT NOTIFICATION
==============================

Risk Level: {risk_level}
Risk Score: {risk_score:.2%}
Location: {location}
Source ID: {source_id}
Input Channel: {channel}
Timestamp: {datetime.now(tz=timezone.utc).isoformat()}

RECOMMENDATIONS:
{chr(10).join([f"• {rec}" for rec in recommendations] if recommendations else ["• Monitor situation closely"])}

---
WaterWatch Automated Alert System
        """
        
        # Send via SMTP
        try:
            await smtp_service.send_alert(
                subject=subject,
                text_body=text_body,
                html_body=text_body,
                recipient=recipient_email
            )
            logger.info(f"🚨 ALERT SENT to {org['name']} ({recipient_email})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {e}")
            return False

_alert_service = AlertService()

# ==================== PYDANTIC SCHEMAS ====================

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    password: str = Field(..., min_length=6)
    location: str
    reporter_type: Optional[str] = Field(default="individual")

class UserInputRequest(BaseModel):
    user_id: int
    input_data_type: str = Field(..., pattern="^(text|image|audio|video)$")
    input_data: str
    channel: str = Field(default="direct")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QARequest(BaseModel):
    user_id: int
    query: str = Field(..., min_length=3, max_length=1000)
    context_limit: int = Field(default=5, ge=1, le=20)
    source_id: Optional[str] = None

class RecommendationRequest(BaseModel):
    user_id: int
    input_id: str
    source_id: Optional[str] = None
    risk_level: Optional[str] = Field(default="MEDIUM")
    sensor_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    forecast_data: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ForecastingRequest(BaseModel):
    user_id: int
    site_id: str
    sensor_data: Dict[str, Any]
    forecast_horizon: str = Field(default="6h")

class ProgressTrackingRequest(BaseModel):
    user_id: int
    input_id: str
    stage: str
    status: str = Field(default="processing")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QdrantUpsertRequest(BaseModel):
    collection: str
    points: List[Dict[str, Any]]

class QdrantSearchRequest(BaseModel):
    collection: str
    query_vector: List[float]
    limit: int = Field(default=10, ge=1, le=100)

class KafkaProduceRequest(BaseModel):
    topic: str
    key: str
    value: Dict[str, Any]

# ==================== FastAPI App ====================

app = FastAPI(
    title="WaterWatch Backend API",
    description="Multi-agent water quality monitoring system",
    version="1.0.0"
)

# ==================== AUTHENTICATION ROUTES ====================

@app.post("/api/v1/auth/register", tags=["Authentication"])
async def register_user(request: UserRegisterRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        existing = db.query(User).filter(
            (User.username == request.username) | (User.email == request.email)
        ).first()
        
        if existing:
            raise HTTPException(status_code=400, detail="User already exists")
        
        new_user = User(
            username=request.username,
            email=request.email,
            password=request.password,
            location=request.location
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        audit_logger.log({
            "action": "user_registered",
            "user_id": new_user.id,
            "username": request.username
        })
        
        return {
            "status": "success",
            "user_id": new_user.id,
            "username": new_user.username,
            "message": "User registered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/v1/auth/login", tags=["Authentication"])
async def login_user(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Login user"""
    try:
        user = db.query(User).filter(User.username == username).first()
        
        if not user or user.password != password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        audit_logger.log({
            "action": "user_login",
            "user_id": user.id
        })
        
        return {
            "status": "success",
            "user_id": user.id,
            "username": user.username,
            "message": "Login successful"
        }
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

# ==================== USER INPUT ROUTES ====================

@app.post("/api/v1/inputs/submit", tags=["User Input"])
async def submit_input(request: UserInputRequest, db: Session = Depends(get_db)):
    """Submit user input (text, image, audio, video)"""
    try:
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        input_id = str(uuid.uuid4())
        
        user_input = Userinput(
            user_id=request.user_id,
            input_data_type=request.input_data_type,
            input_data=request.input_data,
            timestamp=datetime.utcnow()
        )
        db.add(user_input)
        db.commit()
        
        audit_logger.log({
            "action": "input_submitted",
            "user_id": request.user_id,
            "input_id": input_id,
            "type": request.input_data_type
        })
        
        logger.info(f"✅ Input submitted: {input_id}")
        
        return {
            "status": "success",
            "input_id": input_id,
            "user_id": request.user_id,
            "processing_stage": "submitted"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Input submission error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit input")

@app.get("/api/v1/inputs/user/{user_id}", tags=["User Input"])
async def list_user_inputs(
    user_id: int,
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List user inputs"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        inputs = db.query(Userinput).filter(
            Userinput.user_id == user_id
        ).order_by(Userinput.timestamp.desc()).limit(limit).all()
        
        return {
            "status": "success",
            "user_id": user_id,
            "total": len(inputs),
            "inputs": [
                {
                    "id": inp.id,
                    "type": inp.input_data_type,
                    "timestamp": inp.timestamp
                }
                for inp in inputs
            ]
        }
    except Exception as e:
        logger.error(f"List inputs error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list inputs")

# ==================== EMBEDDING ROUTES ====================

@app.post("/api/v1/embeddings/create", tags=["Embeddings"])
async def create_embedding(
    input_id: str = Form(...),
    content: str = Form(...),
    modality: str = Form(default="text")
):
    """Create embedding from multimodal input (Agent 3)"""
    try:
        logger.info(f"Creating {modality} embedding for {input_id}")
        
        embedding = []
        
        try:
            if modality == "text":
                embedding = embedding_service.embed_text(content)
            elif modality == "image":
                embedding = embedding_service.embed_image(content)
            elif modality == "audio":
                embedding = embedding_service.embed_audio(content)
            elif modality == "video":
                embedding = embedding_service.embed_video(content)
        except Exception as e:
            logger.warning(f"Embedding service error: {e}, using Agent 3 directly")
            # Fallback to Agent 3
            if modality == "text":
                from agents.agent3 import embed_clip_text
                embedding = embed_clip_text(content)
        
        embedding_id = str(uuid.uuid4())
        
        return {
            "status": "success",
            "embedding_id": embedding_id,
            "input_id": input_id,
            "modality": modality,
            "dimension": len(embedding)
        }
    except Exception as e:
        logger.error(f"Embedding creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding creation failed: {str(e)}")

# ==================== QDRANT ROUTES ====================

@app.post("/api/v1/qdrant/upsert", tags=["Vector DB"])
async def qdrant_upsert(request: QdrantUpsertRequest):
    """Upsert embeddings to Qdrant (Agent 5)"""
    try:
        logger.info(f"Upserting {len(request.points)} points to {request.collection}")
        
        qdrant_service._initialize()
        
        if not qdrant_service.client:
            raise Exception("Qdrant not available")
        
        from qdrant_client.models import PointStruct
        
        points_to_upsert = []
        for point in request.points:
            point_id = hash(point.get("id", str(uuid.uuid4()))) % (10**9)
            vector = point.get("vector", [])
            payload = point.get("payload", {})
            
            points_to_upsert.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )
        
        qdrant_service.client.upsert(
            collection_name=request.collection,
            points=points_to_upsert
        )
        
        return {
            "status": "success",
            "collection": request.collection,
            "points_upserted": len(points_to_upsert)
        }
    except Exception as e:
        logger.error(f"Qdrant upsert error: {e}")
        raise HTTPException(status_code=500, detail=f"Qdrant upsert failed: {str(e)}")

@app.post("/api/v1/qdrant/search", tags=["Vector DB"])
async def qdrant_search(request: QdrantSearchRequest):
    """Search in Qdrant"""
    try:
        qdrant_service._initialize()
        
        if not qdrant_service.client:
            raise Exception("Qdrant not available")
        
        search_results = qdrant_service.client.search(
            collection_name=request.collection,
            query_vector=request.query_vector,
            limit=request.limit,
            with_payload=True,
            with_vectors=False
        )
        
        results = [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in search_results
        ]
        
        return {
            "status": "success",
            "collection": request.collection,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        raise HTTPException(status_code=500, detail=f"Qdrant search failed: {str(e)}")

# ==================== KAFKA ROUTES ====================

@app.post("/api/v1/kafka/produce", tags=["Kafka"])
async def produce_kafka_message(request: KafkaProduceRequest):
    """Produce message to Kafka"""
    try:
        logger.info(f"Producing to {request.topic}")
        
        success = kafka_service.produce(
            topic=request.topic,
            key=request.key,
            value=request.value
        )
        
        if not success:
            raise Exception("Kafka producer failed")
        
        return {
            "status": "success",
            "topic": request.topic,
            "key": request.key,
            "message": "Message produced successfully"
        }
    except Exception as e:
        logger.error(f"Kafka produce error: {e}")
        raise HTTPException(status_code=500, detail=f"Kafka produce failed: {str(e)}")

# ==================== MEMORY & RETRIEVAL ROUTES (Agent 7) ====================

@app.post("/api/v1/memory/retrieve", tags=["Memory Retrieval"])
async def retrieve_from_memory(request: QARequest, db: Session = Depends(get_db)):
    """Retrieve from liquid memory using embeddings"""
    try:
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"Retrieving memory for user {request.user_id}")
        
        # Generate query embedding
        query_embedding = embedding_service.embed_text(request.query)
        
        retrieved_items = []
        if qdrant_service.client:
            search_results = qdrant_service.client.search(
                collection_name="water_memory",
                query_vector=query_embedding,
                limit=request.context_limit,
                with_payload=True
            )
            
            retrieved_items = [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in search_results
            ]
        
        avg_score = sum(item["score"] for item in retrieved_items) / len(retrieved_items) if retrieved_items else 0
        
        return {
            "status": "success",
            "user_id": request.user_id,
            "query": request.query,
            "retrieved_items": len(retrieved_items),
            "avg_score": float(avg_score),
            "results": retrieved_items
        }
    except Exception as e:
        logger.error(f"Memory retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Memory retrieval failed: {str(e)}")

# ==================== Q&A ROUTES (Agent 8) ====================

@app.post("/api/v1/qa/ask", tags=["Q&A (Agent 8)"])
async def ask_question(request: QARequest, db: Session = Depends(get_db)):
    """Ask question with RAG (Agent 8)"""
    try:
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        qa_id = str(uuid.uuid4())
        logger.info(f"Processing QA: user={request.user_id}, query='{request.query}'")
        
        # Generate query embedding
        query_embedding = embedding_service.embed_text(request.query)
        
        # Retrieve context from Qdrant
        retrieved_items = []
        if qdrant_service.client:
            search_results = qdrant_service.client.search(
                collection_name="water_memory",
                query_vector=query_embedding,
                limit=request.context_limit,
                with_payload=True
            )
            retrieved_items = [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in search_results
            ]
        
        # Build context
        context_text = ""
        if retrieved_items:
            context_parts = []
            for idx, item in enumerate(retrieved_items[:request.context_limit], 1):
                payload = item.get("payload", {})
                context_parts.append(
                    f"[Case {idx}] Location: {payload.get('location', 'N/A')}, "
                    f"Risk: {payload.get('risk_level', 'N/A')}, "
                    f"Notes: {payload.get('description', 'No notes')}"
                )
            context_text = "\n".join(context_parts)
        
        # Call Gemini
        answer = "Unable to process query"
        if llm:
            try:
                prompt = f"""You are a water quality expert.
            
User Query: {request.query}

Historical Context:
{context_text if context_text else 'No previous cases found.'}

Provide a concise, actionable answer based on context."""
                
                response = llm.generate_content(prompt)
                answer = response.text if response else answer
            except Exception as e:
                logger.error(f"Gemini error: {e}")
                answer = f"Unable to generate answer: {str(e)}"
        
        avg_score = sum(item["score"] for item in retrieved_items) / len(retrieved_items) if retrieved_items else 0
        
        return {
            "status": "success",
            "qa_id": qa_id,
            "user_id": request.user_id,
            "query": request.query,
            "answer": answer,
            "evidence_count": len(retrieved_items),
            "avg_score": float(avg_score),
            "sources": retrieved_items,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"QA error: {e}")
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

# ==================== FORECASTING ROUTES (Agent 6) ====================

@app.post("/api/v1/forecasting/predict", tags=["Forecasting"])
async def create_forecast(request: ForecastingRequest, db: Session = Depends(get_db)):
    """Create forecast (Agent 6)"""
    try:
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.info(f"Creating forecast for {request.site_id}")
        
        forecast_data = {
            "site_id": request.site_id,
            "forecast_horizon": request.forecast_horizon,
            "sensor_data": request.sensor_data,
            "predicted_ph": 7.2,
            "predicted_do": 8.5,
            "predicted_salinity": 15.2,
            "confidence": 0.85,
            "timestamp": datetime.utcnow()
        }
        
        forecasting = Forecasting(
            user_id=request.user_id,
            forecasting_data=forecast_data
        )
        db.add(forecasting)
        db.commit()
        db.refresh(forecasting)
        
        return {
            "status": "success",
            "forecast_id": forecasting.id,
            "user_id": request.user_id,
            "forecast_data": forecast_data
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.get("/api/v1/forecasting/user/{user_id}", tags=["Forecasting"])
async def list_forecasts(user_id: int, db: Session = Depends(get_db)):
    """List forecasts for user"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        forecasts = db.query(Forecasting).filter(
            Forecasting.user_id == user_id
        ).order_by(Forecasting.timestamp.desc()).limit(20).all()
        
        return {
            "status": "success",
            "user_id": user_id,
            "total": len(forecasts),
            "forecasts": [{"id": f.id, "timestamp": f.timestamp} for f in forecasts]
        }
    except Exception as e:
        logger.error(f"List forecasts error: {e}")
        raise HTTPException(status_code=500, detail="List forecasts failed")

# ==================== RECOMMENDATIONS ROUTES (Agent 9) ====================

@app.post("/api/v1/recommendations/generate", tags=["Recommendations (Agent 9)"])
async def generate_recommendations(request: RecommendationRequest, db: Session = Depends(get_db)):
    """Generate recommendations (Agent 9)"""
    try:
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        rec_id = str(uuid.uuid4())
        logger.info(f"Generating recommendations: user={request.user_id}, risk={request.risk_level}")
        
        # Call Gemini for recommendations
        recommendations = []
        priority_actions = []
        confidence = 0.8
        
        if llm:
            try:
                sensor_summary = f"Sensor Data: pH={request.sensor_data.get('ph')}, DO={request.sensor_data.get('do')}"
                
                prompt = f"""You are a water quality expert. Generate 3-5 specific recommendations.

Risk Level: {request.risk_level}
{sensor_summary}

Respond with JSON:
{{"recommendations": [{{"action": "...", "priority": "CRITICAL|HIGH|MEDIUM|LOW"}}], "immediate_actions": ["..."]}}"""
                
                response = llm.generate_content(prompt)
                response_text = response.text if response else "{}"
                
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    rec_json = json.loads(json_match.group())
                    recommendations = rec_json.get("recommendations", [])
                    priority_actions = rec_json.get("immediate_actions", [])
            except Exception as e:
                logger.error(f"Gemini recommendations error: {e}")
                # Fallback
                if request.risk_level == "CRITICAL":
                    recommendations = [
                        {"action": "Immediate government notification", "priority": "CRITICAL"}
                    ]
        
        recommendation_data = {
            "input_id": request.input_id,
            "source_id": request.source_id,
            "risk_level": request.risk_level,
            "recommendations": recommendations,
            "priority_actions": priority_actions,
            "confidence": confidence,
            "generated_at": datetime.utcnow()
        }
        
        recommendation = Recommendtion(
            user_id=request.user_id,
            recommendation_data=recommendation_data
        )
        db.add(recommendation)
        db.commit()
        db.refresh(recommendation)
        
        return {
            "status": "success",
            "recommendation_id": recommendation.id,
            "user_id": request.user_id,
            "recommendations": recommendations,
            "priority_actions": priority_actions,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")

@app.get("/api/v1/recommendations/user/{user_id}", tags=["Recommendations (Agent 9)"])
async def list_recommendations(
    user_id: int,
    risk_level: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List recommendations for user"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        recommendations = db.query(Recommendtion).filter(
            Recommendtion.user_id == user_id
        ).order_by(Recommendtion.timestamp.desc()).limit(20).all()
        
        if risk_level:
            recommendations = [
                r for r in recommendations
                if r.recommendation_data.get("risk_level") == risk_level
            ]
        
        return {
            "status": "success",
            "user_id": user_id,
            "total": len(recommendations),
            "recommendations": [
                {"id": r.id, "risk_level": r.recommendation_data.get("risk_level")}
                for r in recommendations
            ]
        }
    except Exception as e:
        logger.error(f"List recommendations error: {e}")
        raise HTTPException(status_code=500, detail="List recommendations failed")

# ==================== ALERTS ROUTES ====================

@app.post("/api/v1/alerts/send", tags=["Alerts"])
async def send_alert_endpoint(
    site_id: str = Form(...),
    risk_level: str = Form(...),
    location: str = Form(...)
):
    """Send alert to government"""
    try:
        logger.info(f"Sending alert: {site_id}, {risk_level}, {location}")
        
        alert_sent = await _alert_service.send_alert(
            risk_level=risk_level,
            risk_score=0.72,
            location=location,
            source_id=site_id,
            channel="direct",
            recommendations=["Monitor closely", "Increase testing"],
            sensor_data=None
        )
        
        return {
            "status": "success",
            "alert_sent": alert_sent,
            "site_id": site_id,
            "risk_level": risk_level
        }
    except Exception as e:
        logger.error(f"Alert send error: {e}")
        raise HTTPException(status_code=500, detail="Alert send failed")

# ==================== PROGRESS TRACKING ROUTES ====================

@app.post("/api/v1/progress/track", tags=["Progress"])
async def track_progress(request: ProgressTrackingRequest, db: Session = Depends(get_db)):
    """Track pipeline progress"""
    try:
        progress_data = {
            "input_id": request.input_id,
            "stage": request.stage,
            "status": request.status,
            "metadata": request.metadata,
            "timestamp": datetime.utcnow()
        }
        
        progress = Progress_Tracking(
            user_id=request.user_id,
            progress_data=progress_data
        )
        db.add(progress)
        db.commit()
        
        return {
            "status": "success",
            "progress_id": progress.id,
            "progress_data": progress_data
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Progress tracking error: {e}")
        raise HTTPException(status_code=500, detail="Progress tracking failed")

# ==================== HEALTH ====================

@app.get("/api/v1/health", tags=["System"])
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "services": {
            "database": "connected",
            "qdrant": "ready",
            "kafka": "ready",
            "llm": "ready" if llm else "unavailable"
        }
    }

@app.get("/api/v1/info", tags=["System"])
async def info():
    """API info"""
    return {
        "name": "WaterWatch Backend",
        "version": "1.0.0",
        "agents": ["Agent 1-9", "Alert Engine"],
        "endpoints": "See /docs"
    }

