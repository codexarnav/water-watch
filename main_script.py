import asyncio
import threading
import time
import logging
import json
import subprocess
from typing import Dict, Any, Optional, List, TypedDict, Literal, Union, Tuple
from datetime import datetime, timezone
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ML/Torch
import torch
import numpy as np
import pandas as pd

# API & LLM
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Vector DB
from qdrant_client import QdrantClient

# LangGraph
from langgraph.graph import StateGraph, END

# Redis for throttling
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

VECTOR_SENSOR = "sensor_dense"
VECTOR_FALLBACK = "semantic_bind"

DEFAULT_WINDOW = "24h"
DEFAULT_HORIZON = "6h"
DEFAULT_MODE = "risk+evidence+why"

SIM_SEARCH_GATE = 0.55
SIM_TOPK = 30
SIM_SEVERITY_MIN = 0.4

BASE_DIR = Path(__file__).resolve().parent
WHATSAPP_DIR = BASE_DIR / "whatsapp"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== IMPORTS ====================
from agents.agent1_sensor_data_ingestion import (
    consume_raw, preprocess, publish_clean, Agent1State
)

from agents.agent2 import (
    make_consumer, parse_iso, compute_z, 
    build_semantic_text, STM_SECONDS, MIN_POINTS, Z_THRESH, STM
)

from agents.agent3 import agent_b_perceive

from agents.agent5 import (
    ensure_collection, voxel_to_point, upsert_points, EMBEDDING_MEMORY, MEM_LOCK
)

from agents.agent6_forecasting import forecast

from agents.agent7_retriever import LiquidRetriever

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

# ==================== TRUST SERVICE ====================

class TrustService:
    """Compute trust scores for different reporter types"""
    TRUST_SCORES = {
        "individual": 0.4,
        "ngo": 0.75,
        "government": 0.9,
    }
    
    def get_trust_score(self, reporter_type: str) -> float:
        return self.TRUST_SCORES.get(reporter_type, 0.3)
    
    def compute_effective_score(self, risk_score: float, reporter_type: str) -> float:
        trust = self.get_trust_score(reporter_type)
        return risk_score * trust

_trust_service = TrustService()

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
        """
        Check if alert should be sent (deduplication).
        Returns True if alert is new, False if it's a duplicate.
        """
        if not self.enabled:
            return True
        
        key = f"alert:{alert_id}"
        try:
            was_set = self.client.set(
                key,
                "1",
                nx=True,
                ex=ttl_seconds
            )
            
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
    
    # Demo government organizations database
    GOVERNMENT_ORGS = {
        "default": {
            "name": "Water Board Authority",
            "email": "alerts@waterboard.gov.in",
            "region": "All India",
            "contact": "+91-1234567890"
        },
        "north": {
            "name": "Northern Region Water Authority",
            "email": "north.water@nra.gov.in",
            "region": "North India",
            "contact": "+91-2345678901"
        },
        "south": {
            "name": "Southern Water Management Board",
            "email": "south.alerts@swmb.gov.in",
            "region": "South India",
            "contact": "+91-3456789012"
        },
        "east": {
            "name": "Eastern Water Resources Department",
            "email": "east.water@ewrd.gov.in",
            "region": "East India",
            "contact": "+91-4567890123"
        },
        "west": {
            "name": "Western Water Conservation Office",
            "email": "west.alerts@wwco.gov.in",
            "region": "West India",
            "contact": "+91-5678901234"
        }
    }
    
    def __init__(self):
        self.smtp_configured = bool(
            os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD")
        )
    
    def get_nearest_org_by_location(self, location: Optional[str]) -> Dict[str, Any]:
        """
        Get nearest government org for a location.
        Currently uses demo data - can be extended with actual geohashing.
        """
        if not location:
            return self.GOVERNMENT_ORGS["default"]
        
        # Demo location mapping
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
    
    def build_alert_email(
        self,
        risk_level: str,
        risk_score: float,
        location: str,
        source_id: str,
        channel: str,
        recommendations: List[str],
        sensor_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, str]:
        """Build alert email subject and body (text + HTML)"""
        
        severity_emoji = {
            "HIGH": "🚨",
            "MEDIUM": "⚠️",
            "LOW": "ℹ️",
            "CRITICAL": "🔴"
        }.get(risk_level, "🔔")
        
        subject = f"{severity_emoji} WATER QUALITY ALERT - {risk_level} RISK - {location}"
        
        # Text body
        text_body = f"""
WATERWATCH ALERT NOTIFICATION
==============================

Risk Level: {risk_level}
Risk Score: {risk_score:.2%}
Location: {location}
Source ID: {source_id}
Input Channel: {channel}
Timestamp: {datetime.now(tz=timezone.utc).isoformat()}

ALERT DETAILS:
{sensor_data if sensor_data else 'Sensor data processing...'}

RECOMMENDATIONS:
{chr(10).join([f"• {rec}" for rec in recommendations] if recommendations else ["• Monitor situation closely"])}

ACTION REQUIRED:
If risk level is HIGH or CRITICAL, immediate action is recommended.
Please deploy field teams for verification and containment.

---
WaterWatch Automated Alert System
Government Water Resources Department
        """
        
        
        return subject, text_body
    
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
        
        # Get organization
        org = self.get_nearest_org_by_location(location)
        recipient_email = org["email"]
        
        # Check throttling
        alert_id = f"{source_id}_{risk_level}_{int(time.time() / 3600)}"
        if _throttle_service and not _throttle_service.allow_alert(alert_id):
            logger.info(f"[ALERT] Throttled for {source_id}")
            return False
        
        # Build email
        subject, text_body, html_body = self.build_alert_email(
            risk_level=risk_level,
            risk_score=risk_score,
            location=location,
            source_id=source_id,
            channel=channel,
            recommendations=recommendations,
            sensor_data=sensor_data
        )
        
        # Send via SMTP (mock if not configured)
        if self.smtp_configured:
            try:
                import aiosmtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = os.getenv("SMTP_FROM", "waterwatch@gov.in")
                msg["To"] = recipient_email
                
                msg.attach(MIMEText(text_body, "plain"))
                msg.attach(MIMEText(html_body, "html"))
                
                await aiosmtplib.send(
                    msg,
                    hostname=os.getenv("SMTP_HOST"),
                    port=int(os.getenv("SMTP_PORT", 587)),
                    username=os.getenv("SMTP_USER"),
                    password=os.getenv("SMTP_PASSWORD"),
                    start_tls=True,
                    timeout=10,
                )
                
                logger.info(f"🚨 ALERT SENT to {org['name']} ({recipient_email})")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to send alert: {e}")
                return False
        else:
            # Mock send for demo
            logger.warning(f"[MOCK ALERT] {subject}")
            logger.warning(f"[MOCK ALERT] To: {recipient_email} ({org['name']})")
            logger.warning(f"[MOCK ALERT] Risk: {risk_level} ({risk_score:.2%})")
            logger.warning(f"[MOCK ALERT] Location: {location}")
            return True

_alert_service = AlertService()

# ==================== STATE SCHEMA ====================

class UserInputPacket(TypedDict):
    """User-provided content"""
    input_id: str
    channel: InputChannel
    modality: InputModality
    content: str
    source_id: str
    location: str  # Demo location for alert purposes
    metadata: Dict[str, Any]

class SensorDataPacket(TypedDict):
    """Sensor stream data"""
    raw_data: Optional[Dict[str, Any]]
    cleaned_data: Optional[Dict[str, Any]]
    spike_event: Optional[Dict[str, Any]]
    semantic_event: Optional[Dict[str, Any]]

class EmbeddingPacket(TypedDict):
    """User content embeddings"""
    percept: Optional[Dict[str, Any]]
    percept_id: Optional[str]
    embeddings: Dict[str, Any]
    vector_stored: bool

class LiquidMemoryPacket(TypedDict):
    """Retrieval results from Agent 7"""
    query_vector: Optional[List[float]]
    retrieved_items: List[Dict[str, Any]]
    num_items: int
    avg_score: float

class AnalysisPacket(TypedDict):
    """Downstream analysis outputs"""
    forecast_result: Optional[Dict[str, Any]]
    qa_result: Optional[Dict[str, Any]]
    recommendations: Optional[List[Dict[str, Any]]]
    priority_actions: Optional[List[str]]

class AlertPacket(TypedDict):
    """Alert status"""
    alert_sent: bool
    alert_level: Optional[str]
    recipient_org: Optional[str]
    alert_time: Optional[float]

class PipelineState(TypedDict):
    """Complete pipeline state"""
    pipeline_id: str
    created_at: float
    user_input: Optional[UserInputPacket]
    embedding_packet: Optional[EmbeddingPacket]
    sensor_packet: Optional[SensorDataPacket]
    liquid_memory: Optional[LiquidMemoryPacket]
    analysis: Optional[AnalysisPacket]
    alert: Optional[AlertPacket]
    completed_agents: List[str]
    errors: List[str]

# ==================== DEMO LOCATION RESOLVER ====================

def resolve_location_from_input(user_input: UserInputPacket) -> str:
    """
    Extract location from user input.
    For demo: Use metadata if available, otherwise use default.
    In production: Use geohashing, GPS, reverse geocoding, etc.
    """
    # Check metadata first
    if "location" in user_input["metadata"]:
        return user_input["metadata"]["location"]
    
    if "geohash" in user_input["metadata"]:
        return user_input["metadata"]["geohash"]
    
    # Demo locations based on source_id
    demo_locations = {
        "well_001": "North Delhi",
        "well_002": "Tamil Nadu (South)",
        "well_003": "Maharashtra (West)",
        "whatsapp_user": "East India",
    }
    
    return demo_locations.get(user_input["source_id"], "All India")

# ==================== PARSE WHATSAPP ====================

def parse_whatsapp_message(message_data: Dict[str, Any]) -> UserInputPacket:
    """Convert WhatsApp message to UserInputPacket"""
    modality_str = message_data.get("modality", "text").lower()
    try:
        modality = InputModality[modality_str.upper()]
    except KeyError:
        modality = InputModality.TEXT
    
    payload = message_data.get("payload", {})
    context = message_data.get("context", {})
    
    content = ""
    if modality == InputModality.TEXT:
        content = payload.get("text", "")
    elif modality == InputModality.IMAGE:
        content = payload.get("image_uri", "")
    elif modality == InputModality.AUDIO:
        content = payload.get("audio_uri", "")
    elif modality == InputModality.VIDEO:
        content = payload.get("video_uri", "")
    
    return UserInputPacket(
        input_id=str(uuid.uuid4()),
        channel=InputChannel.WHATSAPP,
        modality=modality,
        content=content,
        source_id=context.get("geohash", "whatsapp_user"),
        location=context.get("geohash", "East India"),  # Demo
        metadata={
            "source_channel": "whatsapp",
            "timestamp": context.get("timestamp"),
            "geohash": context.get("geohash"),
        }
    )

# ==================== PIPELINE NODES ====================

def node_user_input(state: PipelineState) -> PipelineState:
    """Entry point"""
    try:
        user_input = state["user_input"]
        if not user_input:
            state["errors"].append("No user input")
            return state
        
        # Resolve location for alerts
        location = resolve_location_from_input(user_input)
        user_input["location"] = location
        
        channel_emoji = "📱" if user_input["channel"] == InputChannel.WHATSAPP else "📥"
        logger.info(
            f"[USER_INPUT] {channel_emoji} [{user_input['channel'].value}] "
            f"{user_input['modality'].value} | location={location}"
        )
        return state
        
    except Exception as e:
        state["errors"].append(f"USER_INPUT: {str(e)}")
        logger.error(f"[USER_INPUT] ❌ Error: {e}")
        return state

def node_request_sensor_data(state: PipelineState) -> PipelineState:
    """
    Alert nearest government org to send sensor data for user's location
    (Initial request before pipeline execution)
    """
    try:
        user_input = state["user_input"]
        if not user_input:
            return state
        
        location = user_input["location"]
        source_id = user_input["source_id"]
        channel = user_input["channel"].value
        
        # Get nearest org for this location
        org = _alert_service.get_nearest_org_by_location(location)
        
        logger.info(
            f"[SENSOR_REQUEST] 📡 Requesting sensor data from {org['name']} for {location}"
        )
        
        # In production, this would trigger an actual API call to gov org
        # For now, just log it
        logger.info(f"[SENSOR_REQUEST] Message: 'Please provide sensor readings for {location}'")
        
        return state
        
    except Exception as e:
        logger.warning(f"[SENSOR_REQUEST] ⚠️ Error: {e}")
        return state

def node_perception_embedding(state: PipelineState) -> PipelineState:
    """Agent 3: Multimodal Perception & Embedding"""
    try:
        user_input = state["user_input"]
        if not user_input:
            return state
        
        modality = user_input["modality"]
        content = user_input["content"]
        
        routed_signal = {
            "modality": modality.value,
            "payload": {
                "text": content if modality == InputModality.TEXT else None,
                "image_uri": content if modality == InputModality.IMAGE else None,
                "audio_uri": content if modality == InputModality.AUDIO else None,
                "video_uri": content if modality == InputModality.VIDEO else None,
            },
            "context": {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "source_id": user_input.get("source_id", "user"),
                "channel": user_input["channel"].value,
                "location": user_input["location"]
            }
        }
        
        percept = agent_b_perceive(routed_signal)
        
        if percept is None:
            state["errors"].append("Perception failed")
            return state
        
        embeddings = {
            "semantic_bind": percept.get("semantic_bind"),
            "lexical_sparse": percept.get("lexical_sparse"),
        }
        
        state["embedding_packet"] = {
            "percept": percept,
            "percept_id": percept.get("percept_id"),
            "embeddings": embeddings,
            "vector_stored": False
        }
        
        logger.info(f"[AGENT3] ✅ Embeddings generated | modality={modality.value}")
        state["completed_agents"].append("AGENT3")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT3: {str(e)}")
        logger.error(f"[AGENT3] ❌ Error: {e}")
        return state

def node_embedding_memory_store(state: PipelineState) -> PipelineState:
    """Agent 4: Store embeddings"""
    try:
        embedding_packet = state["embedding_packet"]
        if not embedding_packet or not embedding_packet.get("percept"):
            return state
        
        percept = embedding_packet["percept"]
        vectors = {}
        
        for key in ["semantic_bind", "semantic_image", "semantic_audio", 
                    "sensor_dense", "semantic_video", "lexical_sparse"]:
            if key in percept and percept[key] is not None:
                vectors[key] = percept[key]
        
        hydro_voxel = {
            "percept_id": percept["percept_id"],
            "modality": percept.get("modality", "unknown"),
            "vectors": vectors,
            "context": percept.get("context", {}),
            "raw_ref": percept.get("raw_ref", {}),
            "ingested_at": time.time(),
        }
        
        with MEM_LOCK:
            EMBEDDING_MEMORY[percept["percept_id"]] = hydro_voxel
        
        logger.info(f"[AGENT4] ✅ Stored in memory | percept_id={percept['percept_id']}")
        state["completed_agents"].append("AGENT4")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT4: {str(e)}")
        logger.error(f"[AGENT4] ❌ Error: {e}")
        return state

def node_vector_db_store(state: PipelineState) -> PipelineState:
    """Agent 5: Store in Qdrant"""
    try:
        ensure_collection()
        
        embedding_packet = state["embedding_packet"]
        if not embedding_packet or not embedding_packet.get("percept_id"):
            return state
        
        percept_id = embedding_packet["percept_id"]
        
        with MEM_LOCK:
            hydro_voxel = EMBEDDING_MEMORY.get(percept_id)
        
        if not hydro_voxel:
            return state
        
        point = voxel_to_point(hydro_voxel)
        upsert_points([point])
        
        state["embedding_packet"]["vector_stored"] = True
        
        logger.info(f"[AGENT5] ✅ Stored in Qdrant | point_id={point.id}")
        state["completed_agents"].append("AGENT5")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT5: {str(e)}")
        logger.error(f"[AGENT5] ❌ Error: {e}")
        return state

def node_sensor_ingest(state: PipelineState) -> PipelineState:
    """Agent 1: Sensor ingestion"""
    try:
        agent1_state = Agent1State(raw=None, clean=None)
        agent1_state = consume_raw(agent1_state)
        agent1_state = preprocess(agent1_state)
        agent1_state = publish_clean(agent1_state)
        
        state["sensor_packet"] = {
            "raw_data": agent1_state.get("raw"),
            "cleaned_data": agent1_state.get("clean"),
            "spike_event": None,
            "semantic_event": None
        }
        
        if agent1_state.get("clean"):
            logger.info(f"[AGENT1] ✅ Sensor ingested")
            state["completed_agents"].append("AGENT1")
        
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT1: {str(e)}")
        logger.warning(f"[AGENT1] ⚠️ Sensor ingestion skipped: {e}")
        return state

def node_spike_detection(state: PipelineState) -> PipelineState:
    """Agent 2: Spike detection"""
    try:
        sensor_packet = state["sensor_packet"]
        if not sensor_packet or not sensor_packet.get("cleaned_data"):
            return state
        
        cleaned_data = sensor_packet["cleaned_data"]
        source_id = cleaned_data.get("source_id")
        timestamp_str = cleaned_data.get("timestamp", "")
        readings = cleaned_data.get("readings", {})
        
        try:
            timestamp = parse_iso(timestamp_str)
        except:
            timestamp = time.time()
        
        history = STM[source_id]
        if not history:
            for metric, value in readings.items():
                if value is not None:
                    history.append((timestamp, value))
            return state
        
        anomalies = []
        if len(history) >= MIN_POINTS:
            for metric, current_val in readings.items():
                if current_val is None:
                    continue
                
                history_vals = [r.get(metric) for (_, r) in history 
                               if r.get(metric) is not None]
                
                if len(history_vals) >= MIN_POINTS:
                    baseline = history_vals[:-1] if len(history_vals) > 1 else history_vals
                    mean, std, z = compute_z(baseline, float(current_val))
                    
                    if abs(z) >= Z_THRESH:
                        anomalies.append({
                            "metric": metric,
                            "value": float(current_val),
                            "z_score": z,
                            "mean": mean,
                            "std": std
                        })
        
        if anomalies:
            spike = max(anomalies, key=lambda x: abs(x["z_score"]))
            semantic_text = build_semantic_text(
                source_id, spike["metric"], spike["value"], spike["z_score"]
            )
            
            state["sensor_packet"]["spike_event"] = spike
            state["sensor_packet"]["semantic_event"] = {"text": semantic_text}
            
            logger.info(f"[AGENT2] ✅ Spike detected | metric={spike['metric']}")
        
        state["completed_agents"].append("AGENT2")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT2: {str(e)}")
        logger.warning(f"[AGENT2] ⚠️ Spike detection error: {e}")
        return state

def route_to_liquid_memory(state: PipelineState) -> str:
    """Conditional routing"""
    has_embedding = (state["embedding_packet"] and 
                     state["embedding_packet"].get("vector_stored"))
    has_sensor = (state["sensor_packet"] and 
                  state["sensor_packet"].get("cleaned_data"))
    
    if has_embedding or has_sensor:
        return "retriever"
    else:
        return "end"

def node_liquid_memory_retrieval(state: PipelineState) -> PipelineState:
    """Agent 7: Liquid Memory Retrieval"""
    try:
        query_vector = None
        if state["embedding_packet"] and state["embedding_packet"].get("embeddings"):
            query_vector = state["embedding_packet"]["embeddings"].get("semantic_bind")
        
        if not query_vector:
            state["liquid_memory"] = {
                "query_vector": None,
                "retrieved_items": [],
                "num_items": 0,
                "avg_score": 0.0
            }
            state["completed_agents"].append("AGENT7")
            return state
        
        retriever = LiquidRetriever()
        
        results = retriever.retrieve_with_liquid_memory(
            query_vector=query_vector,
            vector_name=VECTOR_FALLBACK,
            alpha=0.7,
            beta=0.3,
            top_k=SIM_TOPK,
            decay_scale="14d",
            decay_factor=0.5
        )
        
        filtered = [r for r in results if r["score"] >= SIM_SEARCH_GATE]
        
        reranked = retriever.mmr_reranking(
            query_vector=query_vector,
            candidates=filtered,
            lambda_mult=0.5,
            k=SIM_TOPK
        )
        
        retrieval_scores = [r["score"] for r in reranked]
        avg_score = np.mean(retrieval_scores) if retrieval_scores else 0.0
        
        state["liquid_memory"] = {
            "query_vector": query_vector,
            "retrieved_items": reranked,
            "num_items": len(reranked),
            "avg_score": float(avg_score)
        }
        
        logger.info(f"[AGENT7] ✅ Liquid Memory Retrieval | retrieved={len(reranked)}")
        state["completed_agents"].append("AGENT7")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT7: {str(e)}")
        state["liquid_memory"] = {
            "query_vector": None,
            "retrieved_items": [],
            "num_items": 0,
            "avg_score": 0.0
        }
        return state

def node_forecasting(state: PipelineState) -> PipelineState:
    """Agent 6: Forecasting"""
    try:
        liquid_memory = state["liquid_memory"]
        if not liquid_memory or liquid_memory.get("num_items") == 0:
            return state
        
        sensor_packet = state["sensor_packet"]
        if not sensor_packet or not sensor_packet.get("cleaned_data"):
            return state
        
        source_id = sensor_packet["cleaned_data"].get("source_id")
        if not source_id:
            return state
        
        forecast_result = forecast(
            well_id=source_id,
            window=DEFAULT_WINDOW,
            horizon=DEFAULT_HORIZON,
            mode=DEFAULT_MODE
        )
        
        if not state["analysis"]:
            state["analysis"] = {
                "forecast_result": None,
                "qa_result": None,
                "recommendations": None,
                "priority_actions": None,
            }
        
        state["analysis"]["forecast_result"] = forecast_result
        
        risk_level = forecast_result.get("risk_forecast", {}).get("risk_level", "UNKNOWN")
        
        logger.info(f"[AGENT6] ✅ Forecast complete | risk_level={risk_level}")
        state["completed_agents"].append("AGENT6")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT6: {str(e)}")
        logger.warning(f"[AGENT6] ⚠️ Forecasting error: {e}")
        return state

def node_qa_rag(state: PipelineState) -> PipelineState:
    """Agent 8: QA/RAG"""
    try:
        liquid_memory = state["liquid_memory"]
        if not liquid_memory or liquid_memory.get("num_items") == 0:
            return state
        
        user_input = state["user_input"]
        if not user_input or user_input["modality"] != InputModality.TEXT:
            return state
        
        query = user_input.get("content")
        if not query:
            return state
        
        retrieved_context = liquid_memory.get("retrieved_items", [])[:3]
        
        if not state["analysis"]:
            state["analysis"] = {
                "forecast_result": None,
                "qa_result": None,
                "recommendations": None,
                "priority_actions": None,
            }
        
        state["analysis"]["qa_result"] = {
            "query": query,
            "answer": "RAG answer based on retrieved incidents",
            "evidence_count": len(retrieved_context),
            "avg_evidence_score": liquid_memory.get("avg_score", 0.0)
        }
        
        logger.info(f"[AGENT8] ✅ QA/RAG complete")
        state["completed_agents"].append("AGENT8")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT8: {str(e)}")
        logger.warning(f"[AGENT8] ⚠️ QA/RAG error: {e}")
        return state

def node_recommendations(state: PipelineState) -> PipelineState:
    """Agent 9: Recommendations"""
    try:
        liquid_memory = state["liquid_memory"]
        analysis = state["analysis"]
        
        if not liquid_memory or not analysis:
            return state
        
        forecast_result = analysis.get("forecast_result")
        
        recommendations = []
        priority_actions = []
        
        if forecast_result:
            risk_forecast = forecast_result.get("risk_forecast", {})
            risk_level = risk_forecast.get("risk_level")
            
            if risk_level == "HIGH":
                recommendations.append({
                    "recommendation": "🚨 IMMEDIATE ACTION: Contact water management",
                    "priority": "CRITICAL"
                })
                priority_actions.append("Alert government team")
                
            elif risk_level == "MEDIUM":
                recommendations.append({
                    "recommendation": "⚠️ Schedule water quality testing",
                    "priority": "HIGH"
                })
                priority_actions.append("Schedule diagnostic tests")
        
        if liquid_memory.get("num_items", 0) > 0:
            recommendations.append({
                "recommendation": f"📚 Review {liquid_memory['num_items']} similar incidents",
                "priority": "MEDIUM"
            })
        
        state["analysis"]["recommendations"] = recommendations
        state["analysis"]["priority_actions"] = priority_actions
        
        logger.info(f"[AGENT9] ✅ Recommendations generated")
        state["completed_agents"].append("AGENT9")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT9: {str(e)}")
        logger.warning(f"[AGENT9] ⚠️ Recommendations error: {e}")
        return state

async def node_alert_engine(state: PipelineState) -> PipelineState:
    """
    ALERT ENGINE: Send alerts to government if risk is HIGH or CRITICAL
    Triggered after forecasting and recommendations
    """
    try:
        user_input = state["user_input"]
        analysis = state["analysis"]
        
        if not user_input or not analysis:
            state["alert"] = {
                "alert_sent": False,
                "alert_level": None,
                "recipient_org": None,
                "alert_time": None
            }
            return state
        
        forecast_result = analysis.get("forecast_result")
        if not forecast_result:
            state["alert"] = {
                "alert_sent": False,
                "alert_level": None,
                "recipient_org": None,
                "alert_time": None
            }
            return state
        
        risk_forecast = forecast_result.get("risk_forecast", {})
        risk_level = risk_forecast.get("risk_level", "LOW")
        risk_score = risk_forecast.get("risk_score", 0.0)
        
        recommendations = [
            rec.get("recommendation", "")
            for rec in analysis.get("recommendations", [])
        ]
        
        alert_sent = await _alert_service.send_alert(
            risk_level=risk_level,
            risk_score=risk_score,
            location=user_input["location"],
            source_id=user_input["source_id"],
            channel=user_input["channel"].value,
            recommendations=recommendations,
            sensor_data=state["sensor_packet"].get("cleaned_data") if state["sensor_packet"] else None
        )
        
        # Get org info for logging
        org = _alert_service.get_nearest_org_by_location(user_input["location"])
        
        state["alert"] = {
            "alert_sent": alert_sent,
            "alert_level": risk_level if alert_sent else None,
            "recipient_org": org["name"] if alert_sent else None,
            "alert_time": time.time() if alert_sent else None
        }
        
        if alert_sent:
            logger.info(
                f"[ALERT ENGINE] ✅ Alert sent to {org['name']} | "
                f"risk={risk_level} | score={risk_score:.2%}"
            )
        
        state["completed_agents"].append("ALERT_ENGINE")
        return state
        
    except Exception as e:
        state["errors"].append(f"ALERT_ENGINE: {str(e)}")
        logger.error(f"[ALERT_ENGINE] ❌ Error: {e}")
        state["alert"] = {
            "alert_sent": False,
            "alert_level": None,
            "recipient_org": None,
            "alert_time": None
        }
        return state

# ==================== BUILD GRAPH ====================

def build_pipeline_graph() -> StateGraph:
    """Build complete pipeline with alert engine"""
    graph = StateGraph(PipelineState)
    
    graph.add_node("input", node_user_input)
    graph.add_node("sensor_request", node_request_sensor_data)  # NEW: Request sensor data
    
    graph.add_node("perception", node_perception_embedding)
    graph.add_node("embed_store", node_embedding_memory_store)
    graph.add_node("vector_store", node_vector_db_store)
    
    graph.add_node("sensor_ingest", node_sensor_ingest)
    graph.add_node("spike_detect", node_spike_detection)
    
    graph.add_node("retriever", node_liquid_memory_retrieval)
    
    graph.add_node("forecast", node_forecasting)
    graph.add_node("qa_rag", node_qa_rag)
    graph.add_node("recommend", node_recommendations)
    
    graph.add_node("alert_engine", node_alert_engine)  # NEW: Alert engine
    
    graph.set_entry_point("input")
    
    # Request sensor data immediately
    graph.add_edge("input", "sensor_request")
    
    # Split to both paths after sensor request
    graph.add_edge("sensor_request", "perception")
    graph.add_edge("sensor_request", "sensor_ingest")
    
    # Path A: User content
    graph.add_edge("perception", "embed_store")
    graph.add_edge("embed_store", "vector_store")
    
    # Path B: Sensor stream
    graph.add_edge("sensor_ingest", "spike_detect")
    
    # Convergence
    graph.add_conditional_edges(
        "vector_store",
        route_to_liquid_memory,
        {"retriever": "retriever", "end": END}
    )
    
    graph.add_conditional_edges(
        "spike_detect",
        route_to_liquid_memory,
        {"retriever": "retriever", "end": END}
    )
    
    # Parallel analysis
    graph.add_edge("retriever", "forecast")
    graph.add_edge("retriever", "qa_rag")
    
    # Alert engine after analysis
    graph.add_edge("forecast", "alert_engine")
    graph.add_edge("qa_rag", "recommend")
    graph.add_edge("recommend", "alert_engine")
    
    graph.add_edge("alert_engine", END)
    
    return graph.compile()

# ==================== EXECUTION ====================

def run_pipeline(user_input: UserInputPacket) -> PipelineState:
    """Execute complete pipeline with alerts"""
    logger.info("\n" + "=" * 100)
    logger.info("🚀 WATERWATCH PIPELINE STARTED - WITH ALERT ENGINE")
    logger.info("=" * 100 + "\n")
    
    state = PipelineState(
        pipeline_id=str(uuid.uuid4()),
        created_at=time.time(),
        user_input=user_input,
        embedding_packet=None,
        sensor_packet=None,
        liquid_memory=None,
        analysis=None,
        alert=None,
        completed_agents=[],
        errors=[]
    )
    
    graph = build_pipeline_graph()
    result = graph.invoke(state)
    
    logger.info("\n" + "=" * 100)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 100)
    logger.info(f"Agents: {' → '.join(result['completed_agents'])}")
    logger.info(f"Alert Sent: {result['alert'].get('alert_sent') if result['alert'] else False}")
    if result["alert"] and result["alert"].get("alert_sent"):
        logger.info(f"Recipient: {result['alert']['recipient_org']}")
    logger.info("=" * 100 + "\n")
    
    return result

# ==================== MAIN ====================

if __name__ == "__main__":
    # Example 1: Direct input - High risk
    direct_input = UserInputPacket(
        input_id=str(uuid.uuid4()),
        channel=InputChannel.DIRECT,
        modality=InputModality.IMAGE,
        content="images/water_sample.jpg",
        source_id="well_001",
        location="North Delhi",  # Will trigger alert if HIGH risk
        metadata={"location": "North Delhi"}
    )
    
    logger.info("\n" + "=" * 100)
    logger.info("EXAMPLE 1: DIRECT INPUT WITH ALERT")
    logger.info("=" * 100)
    result = run_pipeline(direct_input)
    
    # Print results
    if result["alert"] and result["alert"]["alert_sent"]:
        print("\n🚨 ALERT NOTIFICATION SENT")
        print(f"Organization: {result['alert']['recipient_org']}")
        print(f"Risk Level: {result['alert']['alert_level']}")
    
    if result["analysis"] and result["analysis"]["recommendations"]:
        print("\n💡 RECOMMENDATIONS:")
        for rec in result["analysis"]["recommendations"]:
            print(f"  • {rec.get('recommendation')} [{rec.get('priority')}]")
    
    print("\n" + "=" * 100 + "\n")