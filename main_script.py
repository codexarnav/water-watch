import asyncio
import threading
import time
import logging
import json
import subprocess
from typing import Dict, Any, Optional, List, TypedDict, Literal, Union
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
    DIRECT = "direct"          # Direct user upload
    WHATSAPP = "whatsapp"      # WhatsApp Web message

# ==================== STATE SCHEMA ====================
class UserInputPacket(TypedDict):
    """User-provided content (from direct input or WhatsApp)"""
    input_id: str
    channel: InputChannel      # NEW: Track source (direct or whatsapp)
    modality: InputModality
    content: str
    source_id: str
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
    """Retrieval results from Agent 7 (liquid memory)"""
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

class PipelineState(TypedDict):
    """Complete pipeline state"""
    # Metadata
    pipeline_id: str
    created_at: float
    
    # Input
    user_input: Optional[UserInputPacket]
    
    # Parallel Path A: User Content
    embedding_packet: Optional[EmbeddingPacket]
    
    # Parallel Path B: Sensor Stream
    sensor_packet: Optional[SensorDataPacket]
    
    # CENTRAL HUB: Liquid Memory Retrieval (Agent 7)
    liquid_memory: Optional[LiquidMemoryPacket]
    
    # Downstream Analysis
    analysis: Optional[AnalysisPacket]
    
    # Tracking
    completed_agents: List[str]
    errors: List[str]

# ==================== WHATSAPP INTEGRATION ====================

def parse_whatsapp_message(message_data: Dict[str, Any]) -> UserInputPacket:
    """
    Convert WhatsApp message data to UserInputPacket
    
    Message format from scraper.py:
    {
        "modality": "text|image|audio|video",
        "payload": {...},
        "context": {
            "timestamp": ISO timestamp,
            "source": "whatsapp",
            "geohash": "unknown"
        }
    }
    """
    modality_str = message_data.get("modality", "text").lower()
    try:
        modality = InputModality[modality_str.upper()]
    except KeyError:
        modality = InputModality.TEXT
    
    payload = message_data.get("payload", {})
    context = message_data.get("context", {})
    
    # Extract content based on modality
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
        metadata={
            "source_channel": "whatsapp",
            "timestamp": context.get("timestamp"),
            "geohash": context.get("geohash"),
            "whatsapp_original": message_data
        }
    )

def start_whatsapp_listener() -> threading.Thread:
    """
    Start WhatsApp scraper in background thread
    Monitors messages.jsonl file for new messages
    """
    def listen_whatsapp():
        """Listen to WhatsApp scraper output"""
        try:
            processor_script = WHATSAPP_DIR / "processor.py"
            logger.info(f"🔌 Starting WhatsApp listener: {processor_script}")
            
            # Run processor.py which handles scraper communication
            process = subprocess.Popen(
                ["python", "-u", str(processor_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(WHATSAPP_DIR)
            )
            
            logger.info("⏳ WhatsApp listener running (Awaiting messages...)")
            
            # Keep process alive
            while True:
                stdout_line = process.stdout.readline()
                if stdout_line:
                    logger.info(f"[WhatsApp] {stdout_line.strip()}")
                
                if process.poll() is not None:
                    break
            
            logger.warning("⚠️ WhatsApp listener stopped")
            
        except Exception as e:
            logger.error(f"❌ WhatsApp listener error: {e}")
    
    # Start in daemon thread
    thread = threading.Thread(target=listen_whatsapp, daemon=True)
    thread.start()
    return thread

def poll_whatsapp_messages() -> Optional[UserInputPacket]:
    """
    Poll messages.jsonl for new WhatsApp messages
    Returns None if no new messages
    """
    try:
        messages_file = WHATSAPP_DIR / "messages.jsonl"
        
        if not messages_file.exists():
            return None
        
        # Read last line (most recent message)
        with open(messages_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        # Parse last message
        try:
            last_message_data = json.loads(lines[-1].strip())
            user_packet = parse_whatsapp_message(last_message_data)
            logger.info(f"📱 WhatsApp message received: {user_packet['modality'].value}")
            return user_packet
        except json.JSONDecodeError:
            return None
            
    except Exception as e:
        logger.warning(f"⚠️ Error polling WhatsApp messages: {e}")
        return None

# ==================== PIPELINE NODES ====================

def node_user_input(state: PipelineState) -> PipelineState:
    """
    Entry point: Validate and log user input (from direct or WhatsApp)
    """
    try:
        user_input = state["user_input"]
        if not user_input:
            state["errors"].append("No user input provided")
            return state
        
        channel_emoji = "📱" if user_input["channel"] == InputChannel.WHATSAPP else "📥"
        
        logger.info(
            f"[USER_INPUT] {channel_emoji} [{user_input['channel'].value}] "
            f"Received {user_input['modality'].value} | source_id={user_input['source_id']}"
        )
        return state
        
    except Exception as e:
        state["errors"].append(f"USER_INPUT: {str(e)}")
        logger.error(f"[USER_INPUT] ❌ Error: {e}")
        return state

def node_perception_embedding(state: PipelineState) -> PipelineState:
    """
    Agent 3: Multimodal Perception & Embedding
    Works for both direct input and WhatsApp messages
    """
    try:
        user_input = state["user_input"]
        if not user_input:
            return state
        
        modality = user_input["modality"]
        content = user_input["content"]
        
        # Build routed signal for Agent 3
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
                "metadata": user_input.get("metadata", {})
            }
        }
        
        # Call Agent 3: Unified perception
        percept = agent_b_perceive(routed_signal)
        
        if percept is None:
            state["errors"].append("Perception failed")
            logger.warning("[AGENT3] ⚠️ Perception returned None")
            return state
        
        # Extract embeddings
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
        
        logger.info(
            f"[AGENT3] ✅ Embeddings generated | modality={modality.value} | "
            f"channel={user_input['channel'].value}"
        )
        state["completed_agents"].append("AGENT3")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT3: {str(e)}")
        logger.error(f"[AGENT3] ❌ Error: {e}")
        return state

def node_embedding_memory_store(state: PipelineState) -> PipelineState:
    """
    Agent 4: Store embeddings in thread-safe shared memory
    """
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
    """
    Agent 5: Persist embeddings to Qdrant Vector DB
    """
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
    """
    Agent 1: Consume and preprocess sensor data from Kafka
    (Separate from WhatsApp/Direct input - different data source)
    """
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
            source_id = agent1_state["clean"].get("source_id", "unknown")
            logger.info(f"[AGENT1] ✅ Sensor ingested | source_id={source_id}")
            state["completed_agents"].append("AGENT1")
        
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT1: {str(e)}")
        logger.warning(f"[AGENT1] ⚠️ Sensor ingestion skipped: {e}")
        return state

def node_spike_detection(state: PipelineState) -> PipelineState:
    """
    Agent 2: Detect anomalies in sensor stream using Z-score
    """
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
        
        # Update STM
        history = STM[source_id]
        if not history:
            for metric, value in readings.items():
                if value is not None:
                    history.append((timestamp, value))
            return state
        
        # Detect spikes
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
            
            logger.info(
                f"[AGENT2] ✅ Spike detected | metric={spike['metric']} | z={spike['z_score']:.2f}"
            )
        else:
            logger.info(f"[AGENT2] ℹ️  No spikes detected")
        
        state["completed_agents"].append("AGENT2")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT2: {str(e)}")
        logger.warning(f"[AGENT2] ⚠️ Spike detection error: {e}")
        return state

# ==================== CONDITIONAL ROUTING ====================

def route_to_liquid_memory(state: PipelineState) -> str:
    """
    Conditional edge: Route to liquid memory retrieval if data available
    """
    has_embedding = (state["embedding_packet"] and 
                     state["embedding_packet"].get("vector_stored"))
    has_sensor = (state["sensor_packet"] and 
                  state["sensor_packet"].get("cleaned_data"))
    
    if has_embedding or has_sensor:
        logger.info("🔀 [ROUTING] Data ready → Liquid Memory Retrieval (Agent 7)")
        return "liquid_memory"
    else:
        logger.warning("🔀 [ROUTING] No data available → End")
        return "end"

# ==================== AGENT 7: LIQUID MEMORY RETRIEVAL ====================

def node_liquid_memory_retrieval(state: PipelineState) -> PipelineState:
    """
    Agent 7: LIQUID MEMORY RETRIEVAL (Central Hub)
    Feeds Forecasting, QA/RAG, and Recommendations
    """
    try:
        query_vector = None
        if state["embedding_packet"] and state["embedding_packet"].get("embeddings"):
            query_vector = state["embedding_packet"]["embeddings"].get("semantic_bind")
        
        if not query_vector:
            logger.warning("[AGENT7] ⚠️ No query vector - skipping retrieval")
            state["liquid_memory"] = {
                "query_vector": None,
                "retrieved_items": [],
                "num_items": 0,
                "avg_score": 0.0
            }
            state["completed_agents"].append("AGENT7")
            return state
        
        retriever = LiquidRetriever()
        
        # Step 1: Hybrid search with liquid memory formula
        results = retriever.retrieve_with_liquid_memory(
            query_vector=query_vector,
            vector_name=VECTOR_FALLBACK,
            alpha=0.7,
            beta=0.3,
            top_k=SIM_TOPK,
            decay_scale="14d",
            decay_factor=0.5
        )
        
        # Step 2: Filter by similarity gate
        filtered = [r for r in results if r["score"] >= SIM_SEARCH_GATE]
        
        # Step 3: MMR reranking
        reranked = retriever.mmr_reranking(
            query_vector=query_vector,
            candidates=filtered,
            lambda_mult=0.5,
            k=SIM_TOPK
        )
        
        # Calculate metrics
        retrieval_scores = [r["score"] for r in reranked]
        avg_score = np.mean(retrieval_scores) if retrieval_scores else 0.0
        
        state["liquid_memory"] = {
            "query_vector": query_vector,
            "retrieved_items": reranked,
            "num_items": len(reranked),
            "avg_score": float(avg_score)
        }
        
        logger.info(
            f"[AGENT7] ✅ Liquid Memory Retrieval | retrieved={len(reranked)} items | "
            f"avg_score={avg_score:.3f}"
        )
        logger.info(
            f"[AGENT7] 📊 CENTRAL HUB → Feeding Forecast, QA/RAG, Recommendations"
        )
        
        state["completed_agents"].append("AGENT7")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT7: {str(e)}")
        logger.error(f"[AGENT7] ❌ Error: {e}")
        state["liquid_memory"] = {
            "query_vector": None,
            "retrieved_items": [],
            "num_items": 0,
            "avg_score": 0.0
        }
        return state

# ==================== DOWNSTREAM ANALYSIS ====================

def node_forecasting(state: PipelineState) -> PipelineState:
    """
    Agent 6: Risk Forecasting (fed by Agent 7 retrieval)
    """
    try:
        liquid_memory = state["liquid_memory"]
        if not liquid_memory or liquid_memory.get("num_items") == 0:
            logger.info("[AGENT6] ℹ️  No retrieval results - skipping forecast")
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
    """
    Agent 8: QA/RAG Chatbot (fed by Agent 7 retrieval)
    """
    try:
        liquid_memory = state["liquid_memory"]
        if not liquid_memory or liquid_memory.get("num_items") == 0:
            logger.info("[AGENT8] ℹ️  No retrieval results - skipping QA")
            return state
        
        user_input = state["user_input"]
        if not user_input or user_input["modality"] != InputModality.TEXT:
            logger.info("[AGENT8] ℹ️  Skipping QA (non-text input)")
            return state
        
        query = user_input.get("content")
        if not query:
            return state
        
        retrieved_context = liquid_memory.get("retrieved_items", [])[:3]
        context_text = "\n".join([
            f"- {item.get('payload', {}).get('raw_ref', {})}"
            for item in retrieved_context
        ])
        
        prompt = f"""
Based on these similar water quality incidents:

{context_text}

Answer: {query}

Provide concise, evidence-based answer.
"""
        
        response = llm.generate_content(prompt)
        qa_answer = response.text if response else "Unable to generate answer"
        
        if not state["analysis"]:
            state["analysis"] = {
                "forecast_result": None,
                "qa_result": None,
                "recommendations": None,
                "priority_actions": None,
            }
        
        state["analysis"]["qa_result"] = {
            "query": query,
            "answer": qa_answer,
            "evidence_count": len(retrieved_context),
            "avg_evidence_score": liquid_memory.get("avg_score", 0.0)
        }
        
        logger.info(f"[AGENT8] ✅ QA/RAG complete | evidence={len(retrieved_context)}")
        state["completed_agents"].append("AGENT8")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT8: {str(e)}")
        logger.warning(f"[AGENT8] ⚠️ QA/RAG error: {e}")
        return state

def node_recommendations(state: PipelineState) -> PipelineState:
    """
    Agent 9: Recommendations Engine (fed by Agent 7 + Agent 6)
    """
    try:
        liquid_memory = state["liquid_memory"]
        analysis = state["analysis"]
        
        if not liquid_memory or not analysis:
            return state
        
        retrieved_items = liquid_memory.get("retrieved_items", [])
        forecast_result = analysis.get("forecast_result")
        
        recommendations = []
        priority_actions = []
        
        if forecast_result:
            risk_forecast = forecast_result.get("risk_forecast", {})
            risk_level = risk_forecast.get("risk_level")
            
            if risk_level == "HIGH":
                recommendations.append({
                    "recommendation": "🚨 IMMEDIATE ACTION: Contact water management team",
                    "priority": "CRITICAL",
                    "based_on": "High risk forecast"
                })
                priority_actions.append("Alert water management team immediately")
                
            elif risk_level == "MEDIUM":
                recommendations.append({
                    "recommendation": "⚠️ MEDIUM RISK: Schedule water quality testing",
                    "priority": "HIGH",
                    "based_on": "Medium risk forecast"
                })
                priority_actions.append("Schedule diagnostic tests")
        
        if retrieved_items:
            recommendations.append({
                "recommendation": f"📚 Review {len(retrieved_items)} similar historical incidents",
                "priority": "MEDIUM",
                "based_on": "Historical patterns"
            })
        
        state["analysis"]["recommendations"] = recommendations
        state["analysis"]["priority_actions"] = priority_actions
        
        logger.info(
            f"[AGENT9] ✅ Recommendations generated | "
            f"recs={len(recommendations)} | actions={len(priority_actions)}"
        )
        state["completed_agents"].append("AGENT9")
        return state
        
    except Exception as e:
        state["errors"].append(f"AGENT9: {str(e)}")
        logger.warning(f"[AGENT9] ⚠️ Recommendations error: {e}")
        return state

# ==================== BUILD GRAPH ====================

def build_pipeline_graph() -> StateGraph:
    """
    Build LangGraph with WhatsApp + Direct input + Sensor processing
    Both input channels follow the same pipeline!
    """
    graph = StateGraph(PipelineState)
    
    # Entry point
    graph.add_node("input", node_user_input)
    graph.set_entry_point("input")
    
    # Parallel paths A & B
    graph.add_node("perception", node_perception_embedding)
    graph.add_node("embed_store", node_embedding_memory_store)
    graph.add_node("vector_store", node_vector_db_store)
    
    graph.add_node("sensor_ingest", node_sensor_ingest)
    graph.add_node("spike_detect", node_spike_detection)
    
    # Central Hub
    graph.add_node("liquid_memory", node_liquid_memory_retrieval)
    
    # Downstream Analysis
    graph.add_node("forecast", node_forecasting)
    graph.add_node("qa_rag", node_qa_rag)
    graph.add_node("recommend", node_recommendations)
    
    # ========== EDGES ==========
    # Split inputs
    graph.add_edge("input", "perception")
    graph.add_edge("input", "sensor_ingest")
    
    # Path A: User content (both direct and WhatsApp)
    graph.add_edge("perception", "embed_store")
    graph.add_edge("embed_store", "vector_store")
    
    # Path B: Sensor stream
    graph.add_edge("sensor_ingest", "spike_detect")
    
    # Convergence
    graph.add_conditional_edges(
        "vector_store",
        route_to_liquid_memory,
        {"liquid_memory": "liquid_memory", "end": END}
    )
    
    graph.add_conditional_edges(
        "spike_detect",
        route_to_liquid_memory,
        {"liquid_memory": "liquid_memory", "end": END}
    )
    
    # Parallel analysis from Agent 7
    graph.add_edge("liquid_memory", "forecast")
    graph.add_edge("liquid_memory", "qa_rag")
    graph.add_edge("forecast", "recommend")
    graph.add_edge("qa_rag", "recommend")
    
    # End
    graph.add_edge("recommend", END)
    
    return graph.compile()

# ==================== EXECUTION ====================

def run_pipeline(user_input: UserInputPacket) -> PipelineState:
    """Execute the complete pipeline"""
    logger.info("\n" + "=" * 90)
    logger.info("🚀 WATERWATCH PIPELINE STARTED")
    logger.info("=" * 90 + "\n")
    
    state = PipelineState(
        pipeline_id=str(uuid.uuid4()),
        created_at=time.time(),
        user_input=user_input,
        embedding_packet=None,
        sensor_packet=None,
        liquid_memory=None,
        analysis=None,
        completed_agents=[],
        errors=[]
    )
    
    graph = build_pipeline_graph()
    result = graph.invoke(state)
    
    logger.info("\n" + "=" * 90)
    logger.info("✅ PIPELINE COMPLETE")
    logger.info("=" * 90)
    logger.info(f"Agents Executed: {' → '.join(result['completed_agents'])}")
    logger.info(f"Errors: {len(result['errors'])}")
    logger.info("=" * 90 + "\n")
    
    return result

# # ==================== EXAMPLE USAGE ====================

# if __name__ == "__main__":
#     # Example 1: Direct user input
#     direct_input = UserInputPacket(
#         input_id=str(uuid.uuid4()),
#         channel=InputChannel.DIRECT,
#         modality=InputModality.IMAGE,
#         content="images/water_sample.jpg",
#         source_id="well_001",
#         metadata={"location": "site_A"}
#     )
    
#     logger.info("\n" + "=" * 90)
#     logger.info("EXAMPLE 1: DIRECT USER INPUT (IMAGE)")
#     logger.info("=" * 90)
#     result = run_pipeline(direct_input)
    
#     # Example 2: WhatsApp message (simulated)
#     whatsapp_input = UserInputPacket(
#         input_id=str(uuid.uuid4()),
#         channel=InputChannel.WHATSAPP,
#         modality=InputModality.TEXT,
#         content="Water pH level reported as abnormal",
#         source_id="whatsapp_geohash_123",
#         metadata={
#             "source_channel": "whatsapp",
#             "sender": "field_agent",
#             "chat_name": "Water Quality Reports"
#         }
#     )
    
#     logger.info("\n" + "=" * 90)
#     logger.info("EXAMPLE 2: WHATSAPP INPUT (TEXT MESSAGE)")
#     logger.info("=" * 90)
#     result = run_pipeline(whatsapp_input)
    
#     # Pretty print results
#     if result["analysis"]:
#         print("\n" + "=" * 90)
#         print("📊 RESULTS")
#         print("=" * 90)
        
#         analysis = result["analysis"]
#         if analysis.get("forecast_result"):
#             print(f"⚠️  Risk Level: {analysis['forecast_result'].get('risk_forecast', {}).get('risk_level')}")
        
#         if analysis.get("recommendations"):
#             print(f"💡 Recommendations: {len(analysis['recommendations'])}")
#             for rec in analysis["recommendations"]:
#                 print(f"  [{rec.get('priority')}] {rec.get('recommendation')}")
        
#         print("=" * 90 + "\n")