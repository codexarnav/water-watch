import json
import time
import logging
from typing import TypedDict, Optional, Dict, Any, List, Literal
from collections import deque
from datetime import datetime, timedelta
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from kafka import KafkaConsumer, KafkaProducer

# ===== Import from existing files =====
# From kafka_stream.py
from agents.agent1_sensor_data_ingestion import preprocess as kafka_preprocess_state

# From agent2.py
import agents.agent2 as agent2
from agents.agent2 import (
    STM,  # Short-term memory dict
    parse_iso,
    compute_z,
    build_semantic_text,
    spike_detection,
    update_stm,
    MIN_POINTS,
    Z_THRESH
)

# From memory.py
from agents.agent4 import sensor_event_to_routed_signal, store_percept, create_voxel_from_percept

# From citizen files
from citizen_input import agent_a_route
from agents.agent3 import agent_b_perceive

# From kernel
from agents.agent5 import ensure_collection, store_hydro_voxel

# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("orchestrator")

# =========================================================
# CONFIGURATION
# =========================================================

@dataclass
class Config:
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_RAW_TOPIC: str = "sensor.raw"
    KAFKA_CLEAN_TOPIC: str = "sensor.cleaned"
    KAFKA_EVENT_TOPIC: str = "events.queue"
    
    # Processing
    MAX_RETRIES: int = 3
    RETRY_DELAYS: List[int] = None
    
    # Circuit breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 10
    CIRCUIT_BREAKER_WINDOW_SEC: int = 60
    
    # Dead letter queue
    DLQ_PATH: str = "dead_letter_queue.jsonl"
    
    def __post_init__(self):
        if self.RETRY_DELAYS is None:
            self.RETRY_DELAYS = [1, 5, 15]

config = Config()

# =========================================================
# STATE SCHEMA
# =========================================================

class OrchestratorState(TypedDict):
    """Master state for dual-stream orchestration."""
    
    # === Stream Selection ===
    active_stream: Literal["signal", "citizen", "idle"]
    
    # === Signal Stream State ===
    signal_state: Optional[Dict[str, Any]]
    signal_status: Literal["idle", "preprocessing", "spike_detection", "embedding", "storing", "done", "error"]
    signal_error: Optional[str]
    signal_retry_count: int
    
    # === Citizen Stream State ===
    citizen_state: Optional[Dict[str, Any]]
    citizen_status: Literal["idle", "routing", "embedding", "voxel_creation", "storing", "done", "error"]
    citizen_error: Optional[str]
    citizen_retry_count: int
    
    # === Shared Storage State ===
    storage_queue: List[Dict[str, Any]]
    storage_status: Literal["idle", "writing", "done", "error"]
    storage_error: Optional[str]
    
    # === Metrics & Monitoring ===
    processed_count: Dict[str, int]
    error_count: Dict[str, int]
    last_processed_at: Optional[float]

# =========================================================
# CIRCUIT BREAKER
# =========================================================

class CircuitBreaker:
    """Prevent cascading failures."""
    
    def __init__(self, failure_threshold=10, window_seconds=60):
        self.failure_threshold = failure_threshold
        self.window = timedelta(seconds=window_seconds)
        self.failures = deque()
        self.is_open = False
    
    def record_failure(self):
        now = datetime.utcnow()
        self.failures.append(now)
        
        while self.failures and (now - self.failures[0]) > self.window:
            self.failures.popleft()
        
        if len(self.failures) >= self.failure_threshold:
            self.is_open = True
            logger.error(f"⚠️ Circuit breaker opened! {len(self.failures)} failures")
    
    def record_success(self):
        if self.failures:
            self.failures.clear()
        self.is_open = False
    
    def should_process(self) -> bool:
        return not self.is_open

# Global circuit breakers
signal_breaker = CircuitBreaker(config.CIRCUIT_BREAKER_THRESHOLD, config.CIRCUIT_BREAKER_WINDOW_SEC)
citizen_breaker = CircuitBreaker(config.CIRCUIT_BREAKER_THRESHOLD, config.CIRCUIT_BREAKER_WINDOW_SEC)

# =========================================================
# DEAD LETTER QUEUE
# =========================================================

def send_to_dead_letter_queue(stream: str, data: dict, error: str):
    """Log failed messages for manual inspection."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "stream": stream,
        "error": error,
        "data": data
    }
    
    with open(config.DLQ_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    
    logger.error(f"📮 DLQ Entry: {stream} - {error}")

# =========================================================
# KAFKA HELPERS
# =========================================================

_kafka_consumers = {}
_kafka_producer = None

def get_kafka_consumer(topic: str, group_id: str) -> KafkaConsumer:
    """Get or create Kafka consumer for topic."""
    key = f"{topic}:{group_id}"
    if key not in _kafka_consumers:
        _kafka_consumers[key] = KafkaConsumer(
            topic,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            consumer_timeout_ms=100
        )
    return _kafka_consumers[key]

def get_kafka_producer() -> KafkaProducer:
    """Get or create Kafka producer."""
    global _kafka_producer
    if _kafka_producer is None:
        _kafka_producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            retries=5
        )
    return _kafka_producer

def publish_to_kafka(topic: str, key: str, value: dict):
    """Publish message to Kafka topic."""
    producer = get_kafka_producer()
    producer.send(topic, key=key, value=value)
    producer.flush()

# =========================================================
# SIGNAL STREAM HELPERS (using existing functions)
# =========================================================

def preprocess_sensor_data_wrapper(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for kafka_stream.preprocess() which expects state format.
    Converts to/from state format.
    """
    # Create state format expected by kafka_stream.preprocess
    state = {"raw": raw, "clean": None}
    
    # Call existing function
    result_state = kafka_preprocess_state(state)
    
    # Extract clean data
    return result_state["clean"]

def detect_spike_wrapper(clean: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Wrapper for agent2 spike detection using existing functions.
    """
    well_id = clean["source_id"]
    ts = parse_iso(clean["timestamp"])
    readings = clean["readings"]
    
    # Update STM using agent2's update_stm logic
    state_for_stm = {
        "clean_msg": clean,
        "well_id": well_id,
        "spike": None,
        "semantic_event": None
    }
    update_stm(state_for_stm)
    
    # Check if enough points
    if len(STM[well_id]) < MIN_POINTS:
        return None
    
    # Use agent2's spike_detection logic
    spike_state = spike_detection(state_for_stm)
    
    if spike_state.get("spike") is None:
        return None
    
    # Build semantic event (using agent2 logic from push_to_event_queue)
    spike = spike_state["spike"]
    severity = min(1.0, abs(spike["z"]) / 6.0)
    semantic_text = build_semantic_text(well_id, spike["metric"], spike["value"], spike["z"])
    
    semantic_event = {
        "meta": {
            "event_id": str(int(time.time() * 1000)),
            "source_id": well_id,
            "timestamp": clean["timestamp"],
            "producer": "agent2_stm_spike",
            "confidence": 0.85
        },
        "event_type": "sensor",
        "derived": {
            "severity": severity,
            "tags": ["spike_detected", spike["metric"]],
            "semantic_text": semantic_text,
            "reliability_score": 0.9
        },
        "stats": spike
    }
    
    return semantic_event

# =========================================================
# CITIZEN INPUT SOURCE
# =========================================================

citizen_input_queue = []

def has_citizen_input() -> bool:
    """Check if citizen input is available."""
    return len(citizen_input_queue) > 0

def get_citizen_input() -> Optional[Dict[str, Any]]:
    """Get next citizen input from queue/API."""
    if citizen_input_queue:
        return citizen_input_queue.pop(0)
    return None

def add_citizen_input(input_data: Dict[str, Any]):
    """Add citizen input to queue (for testing)."""
    citizen_input_queue.append(input_data)

# =========================================================
# INPUT AVAILABILITY CHECKS
# =========================================================

def has_signal_input() -> bool:
    """Check if signal data is available in Kafka."""
    try:
        consumer = get_kafka_consumer(config.KAFKA_RAW_TOPIC, "orchestrator-peek")
        for msg in consumer:
            return True
        return False
    except:
        return False

# =========================================================
# ORCHESTRATOR NODES
# =========================================================

# === ROUTER NODE ===

def stream_router(state: OrchestratorState) -> OrchestratorState:
    """Determines which stream has data available."""
    if not signal_breaker.should_process() and not citizen_breaker.should_process():
        logger.warning("⚠️ Both circuit breakers are open!")
        return {**state, "active_stream": "idle"}
    
    # Citizen priority
    if has_citizen_input() and citizen_breaker.should_process():
        logger.info("🎯 Routing to CITIZEN stream")
        return {**state, "active_stream": "citizen", "citizen_status": "idle"}
    
    # Signal
    if has_signal_input() and signal_breaker.should_process():
        logger.info("🎯 Routing to SIGNAL stream")
        return {**state, "active_stream": "signal", "signal_status": "idle"}
    
    return {**state, "active_stream": "idle"}

# === SIGNAL STREAM NODES ===

def signal_intake_node(state: OrchestratorState) -> OrchestratorState:
    """Consume ONE message from Kafka sensor.raw topic."""
    try:
        consumer = get_kafka_consumer(config.KAFKA_RAW_TOPIC, "orchestrator-signal")
        msg = next(iter(consumer))
        raw = msg.value
        
        logger.info(f"📥 [SIGNAL] Consumed | well={raw.get('source_id')}")
        
        return {
            **state,
            "signal_state": {"raw": raw},
            "signal_status": "preprocessing",
            "signal_error": None
        }
    except StopIteration:
        return {**state, "active_stream": "idle"}
    except Exception as e:
        logger.error(f"❌ [SIGNAL] Intake failed: {e}")
        return {
            **state,
            "signal_status": "error",
            "signal_error": f"Intake failed: {str(e)}",
            "signal_retry_count": state["signal_retry_count"] + 1
        }

def signal_preprocess_node(state: OrchestratorState) -> OrchestratorState:
    """Preprocess sensor data using kafka_stream.preprocess()."""
    try:
        raw = state["signal_state"]["raw"]
        
        # Use existing preprocess function
        clean = preprocess_sensor_data_wrapper(raw)
        
        # Publish to sensor.cleaned
        publish_to_kafka(config.KAFKA_CLEAN_TOPIC, clean["source_id"], clean)
        
        logger.info(f"✅ [SIGNAL] Preprocessed | well={clean['source_id']}")
        
        return {
            **state,
            "signal_state": {**state["signal_state"], "clean": clean},
            "signal_status": "spike_detection",
            "signal_error": None
        }
    except Exception as e:
        logger.error(f"❌ [SIGNAL] Preprocessing failed: {e}")
        return {
            **state,
            "signal_status": "error",
            "signal_error": f"Preprocessing failed: {str(e)}",
            "signal_retry_count": state["signal_retry_count"] + 1
        }

def signal_spike_detection_node(state: OrchestratorState) -> OrchestratorState:
    """Perform spike detection using agent2 functions."""
    try:
        clean = state["signal_state"]["clean"]
        
        # Use existing spike detection functions
        spike_event = detect_spike_wrapper(clean)
        
        if spike_event:
            # Publish to events.queue
            publish_to_kafka(config.KAFKA_EVENT_TOPIC, spike_event["meta"]["source_id"], spike_event)
            
            logger.info(f"🔔 [SIGNAL] Spike detected | well={clean['source_id']}")
            
            return {
                **state,
                "signal_state": {**state["signal_state"], "event": spike_event},
                "signal_status": "embedding",
                "signal_error": None
            }
        else:
            logger.info(f"✅ [SIGNAL] No spike | well={clean['source_id']}")
            signal_breaker.record_success()
            
            return {
                **state,
                "signal_status": "done",
                "processed_count": {
                    **state["processed_count"],
                    "signal": state["processed_count"]["signal"] + 1
                },
                "last_processed_at": time.time()
            }
            
    except Exception as e:
        logger.error(f"❌ [SIGNAL] Spike detection failed: {e}")
        return {
            **state,
            "signal_status": "error",
            "signal_error": f"Spike detection failed: {str(e)}",
            "signal_retry_count": state["signal_retry_count"] + 1
        }

def signal_embed_node(state: OrchestratorState) -> OrchestratorState:
    """Generate embeddings using memory.py functions."""
    try:
        event = state["signal_state"]["event"]
        
        # Use existing sensor_event_to_routed_signal from memory.py
        routed_signal = sensor_event_to_routed_signal(event)
        
        # Use existing agent_b_perceive from citizen_embed.py
        percept = agent_b_perceive(routed_signal)
        
        if percept is None:
            raise ValueError("Embedding generation returned None")
        
        # Use create_voxel_from_percept from memory.py (canonical voxel creation)
        voxel = create_voxel_from_percept(percept)
        
        # Add to storage queue
        storage_queue = state["storage_queue"] + [voxel]
        
        logger.info(f"🧠 [SIGNAL] Embedded | percept_id={percept['percept_id']}")
        
        return {
            **state,
            "signal_state": {**state["signal_state"], "percept": percept, "voxel": voxel},
            "signal_status": "storing",
            "storage_queue": storage_queue,
            "signal_error": None
        }
        
    except Exception as e:
        logger.error(f"❌ [SIGNAL] Embedding failed: {e}")
        return {
            **state,
            "signal_status": "error",
            "signal_error": f"Embedding failed: {str(e)}",
            "signal_retry_count": state["signal_retry_count"] + 1
        }

# === CITIZEN STREAM NODES ===

def citizen_intake_node(state: OrchestratorState) -> OrchestratorState:
    """Receive citizen input."""
    try:
        raw_input = get_citizen_input()
        
        if raw_input is None:
            return {**state, "active_stream": "idle"}
        
        logger.info(f"📥 [CITIZEN] Received | source={raw_input.get('source')}")
        
        return {
            **state,
            "citizen_state": {"raw": raw_input},
            "citizen_status": "routing",
            "citizen_error": None
        }
    except Exception as e:
        logger.error(f"❌ [CITIZEN] Intake failed: {e}")
        return {
            **state,
            "citizen_status": "error",
            "citizen_error": f"Intake failed: {str(e)}",
            "citizen_retry_count": state["citizen_retry_count"] + 1
        }

def citizen_route_node(state: OrchestratorState) -> OrchestratorState:
    """Route citizen input using citizen_input.agent_a_route()."""
    try:
        raw = state["citizen_state"]["raw"]
        
        # Use existing agent_a_route from citizen_input.py
        routed_signals = agent_a_route(raw)
        
        if not routed_signals:
            raise ValueError("No routed signals generated")
        
        logger.info(f"🔀 [CITIZEN] Routed | signals={len(routed_signals)}")
        
        return {
            **state,
            "citizen_state": {
                **state["citizen_state"],
                "routed_signals": routed_signals
            },
            "citizen_status": "embedding",
            "citizen_error": None
        }
        
    except Exception as e:
        logger.error(f"❌ [CITIZEN] Routing failed: {e}")
        return {
            **state,
            "citizen_status": "error",
            "citizen_error": f"Routing failed: {str(e)}",
            "citizen_retry_count": state["citizen_retry_count"] + 1
        }

def citizen_embed_node(state: OrchestratorState) -> OrchestratorState:
    """Generate embeddings using citizen_embed.agent_b_perceive()."""
    try:
        routed_signals = state["citizen_state"]["routed_signals"]
        
        # Use existing agent_b_perceive from citizen_embed.py
        percepts = []
        for signal in routed_signals:
            percept = agent_b_perceive(signal)
            if percept:
                percepts.append(percept)
        
        if not percepts:
            raise ValueError("No percepts generated")
        
        logger.info(f"🧠 [CITIZEN] Embedded | percepts={len(percepts)}")
        
        return {
            **state,
            "citizen_state": {
                **state["citizen_state"],
                "percepts": percepts
            },
            "citizen_status": "voxel_creation",
            "citizen_error": None
        }
        
    except Exception as e:
        logger.error(f"❌ [CITIZEN] Embedding failed: {e}")
        return {
            **state,
            "citizen_status": "error",
            "citizen_error": f"Embedding failed: {str(e)}",
            "citizen_retry_count": state["citizen_retry_count"] + 1
        }

def citizen_voxel_creation_node(state: OrchestratorState) -> OrchestratorState:
    """Create voxel structures (similar to memory.store_percept())."""
    try:
        percepts = state["citizen_state"]["percepts"]
        
        # Create voxels using canonical function from memory.py
        voxels = [create_voxel_from_percept(percept) for percept in percepts]
        
        # Add to storage queue
        storage_queue = state["storage_queue"] + voxels
        
        logger.info(f"📦 [CITIZEN] Voxels created | count={len(voxels)}")
        
        return {
            **state,
            "citizen_state": {
                **state["citizen_state"],
                "voxels": voxels
            },
            "citizen_status": "storing",
            "storage_queue": storage_queue,
            "citizen_error": None
        }
        
    except Exception as e:
        logger.error(f"❌ [CITIZEN] Voxel creation failed: {e}")
        return {
            **state,
            "citizen_status": "error",
            "citizen_error": f"Voxel creation failed: {str(e)}",
            "citizen_retry_count": state["citizen_retry_count"] + 1
        }

# === STORAGE NODE ===

def storage_writer_node(state: OrchestratorState) -> OrchestratorState:
    """Write voxels to Qdrant using kernel.store_hydro_voxel()."""
    try:
        if not state["storage_queue"]:
            return state
        
        # Use existing ensure_collection from kernel.py
        ensure_collection()
        
        stored_count = 0
        errors = []
        
        # Use existing store_hydro_voxel from kernel.py
        for voxel in state["storage_queue"]:
            try:
                store_hydro_voxel(voxel)
                stored_count += 1
            except Exception as e:
                errors.append(f"Voxel {voxel['percept_id']}: {str(e)}")
        
        if errors:
            logger.error(f"⚠️ [STORAGE] Partial failure: {len(errors)}/{len(state['storage_queue'])}")
            return {
                **state,
                "storage_queue": [],
                "storage_status": "error",
                "storage_error": f"Partial storage failure: {'; '.join(errors[:3])}",
                "error_count": {
                    **state["error_count"],
                    "storage": state["error_count"].get("storage", 0) + len(errors)
                }
            }
        
        # Success
        active = state["active_stream"]
        logger.info(f"💾 [STORAGE] Stored {stored_count} voxels | stream={active}")
        
        if active == "signal":
            signal_breaker.record_success()
        elif active == "citizen":
            citizen_breaker.record_success()
        
        return {
            **state,
            "storage_queue": [],
            "storage_status": "done",
            "storage_error": None,
            f"{active}_status": "done",
            "processed_count": {
                **state["processed_count"],
                active: state["processed_count"][active] + 1
            },
            "last_processed_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ [STORAGE] Failed: {e}")
        return {
            **state,
            "storage_status": "error",
            "storage_error": f"Storage failed: {str(e)}"
        }

# === ERROR HANDLER NODE ===

def error_handler_node(state: OrchestratorState) -> OrchestratorState:
    """Handle errors with retry logic and DLQ."""
    active = state["active_stream"]
    
    if active == "signal":
        retry_count = state["signal_retry_count"]
        error = state["signal_error"]
        breaker = signal_breaker
    elif active == "citizen":
        retry_count = state["citizen_retry_count"]
        error = state["citizen_error"]
        breaker = citizen_breaker
    else:
        logger.error(f"⚠️ [ERROR] Storage error: {state.get('storage_error')}")
        return {**state, "active_stream": "idle"}
    
    logger.warning(f"⚠️ [{active.upper()}] Error (retry {retry_count}/{config.MAX_RETRIES}): {error}")
    
    if retry_count < config.MAX_RETRIES:
        delay = config.RETRY_DELAYS[min(retry_count, len(config.RETRY_DELAYS)-1)]
        logger.info(f"⏱️ [{active.upper()}] Retrying in {delay}s...")
        time.sleep(delay)
        
        if active == "signal":
            return {**state, "signal_status": "preprocessing"}
        else:
            return {**state, "citizen_status": "routing"}
    else:
        logger.error(f"💀 [{active.upper()}] Max retries exceeded, DLQ")
        send_to_dead_letter_queue(active, state[f"{active}_state"], error)
        breaker.record_failure()
        
        return {
            **state,
            f"{active}_status": "done",
            f"{active}_state": None,
            f"{active}_error": None,
            f"{active}_retry_count": 0,
            "error_count": {
                **state["error_count"],
                active: state["error_count"].get(active, 0) + 1
            },
            "active_stream": "idle"
        }

# === WAIT NODE ===

def wait_node(state: OrchestratorState) -> OrchestratorState:
    """Wait briefly when no data available."""
    time.sleep(0.1)
    return state

# =========================================================
# ROUTING LOGIC
# =========================================================

def should_continue(state: OrchestratorState) -> str:
    """Routing logic for the graph."""
    active = state["active_stream"]
    
    if active == "idle":
        return END
    
    # Signal routing
    if active == "signal":
        status = state["signal_status"]
        if status == "idle":
            return "signal_intake"
        elif status == "preprocessing":
            return "signal_preprocess"
        elif status == "spike_detection":
            return "signal_spike"
        elif status == "embedding":
            return "signal_embed"
        elif status == "storing":
            return "storage_writer"
        elif status == "error":
            return "error_handler"
        elif status == "done":
            return END
    
    # Citizen routing
    if active == "citizen":
        status = state["citizen_status"]
        if status == "idle":
            return "citizen_intake"
        elif status == "routing":
            return "citizen_route"
        elif status == "embedding":
            return "citizen_embed"
        elif status == "voxel_creation":
            return "citizen_voxel"
        elif status == "storing":
            return "storage_writer"
        elif status == "error":
            return "error_handler"
        elif status == "done":
            return END
    
    if state["storage_status"] == "error":
        return "error_handler"
    
    return END

# =========================================================
# GRAPH CONSTRUCTION
# =========================================================

def build_orchestration_graph(recursion_limit=1000):
    """
    Build the master orchestration graph.
    
    Args:
        recursion_limit: Maximum number of graph iterations before stopping.
                        Use high value (1000+) for continuous operation,
                        or low value (10-50) for testing single messages.
    """
    graph = StateGraph(OrchestratorState)
    
    # Add all nodes
    graph.add_node("router", stream_router)
    graph.add_node("signal_intake", signal_intake_node)
    graph.add_node("signal_preprocess", signal_preprocess_node)
    graph.add_node("signal_spike", signal_spike_detection_node)
    graph.add_node("signal_embed", signal_embed_node)
    graph.add_node("citizen_intake", citizen_intake_node)
    graph.add_node("citizen_route", citizen_route_node)
    graph.add_node("citizen_embed", citizen_embed_node)
    graph.add_node("citizen_voxel", citizen_voxel_creation_node)
    graph.add_node("storage_writer", storage_writer_node)
    graph.add_node("error_handler", error_handler_node)
    graph.add_node("wait", wait_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional edges
    graph.add_conditional_edges("router", should_continue)
    graph.add_conditional_edges("signal_intake", should_continue)
    graph.add_conditional_edges("signal_preprocess", should_continue)
    graph.add_conditional_edges("signal_spike", should_continue)
    graph.add_conditional_edges("signal_embed", should_continue)
    graph.add_conditional_edges("citizen_intake", should_continue)
    graph.add_conditional_edges("citizen_route", should_continue)
    graph.add_conditional_edges("citizen_embed", should_continue)
    graph.add_conditional_edges("citizen_voxel", should_continue)
    graph.add_conditional_edges("storage_writer", should_continue)
    graph.add_conditional_edges("error_handler", should_continue)
    graph.add_conditional_edges("wait", should_continue)
    
    # Compile graph (recursion_limit will be passed in invoke/stream config)
    return graph.compile()

# =========================================================
# INITIAL STATE
# =========================================================

def create_initial_state() -> OrchestratorState:
    """Create initial orchestrator state."""
    return {
        "active_stream": "idle",
        "signal_state": None,
        "signal_status": "idle",
        "signal_error": None,
        "signal_retry_count": 0,
        "citizen_state": None,
        "citizen_status": "idle",
        "citizen_error": None,
        "citizen_retry_count": 0,
        "storage_queue": [],
        "storage_status": "idle",
        "storage_error": None,
        "processed_count": {"signal": 0, "citizen": 0},
        "error_count": {"signal": 0, "citizen": 0},
        "last_processed_at": None
    }

# =========================================================
# MAIN ORCHESTRATOR LOOP
# =========================================================

def run_orchestrator():
    """Run the orchestrator continuously."""
    orchestrator = build_orchestration_graph()
    state = create_initial_state()
    
    logger.info("🚀 Water-Watch Orchestration Agent Started (Refactored)")
    logger.info(f"📊 Using functions from: kafka_stream, agent2, memory, citizen_input, citizen_embed, kernel")
    
    try:
        while True:
            state = orchestrator.invoke(state)
            
            if state["processed_count"]["signal"] % 10 == 0 or state["processed_count"]["citizen"] % 10 == 0:
                logger.info(
                    f"📈 Metrics | "
                    f"signal={state['processed_count']['signal']} "
                    f"citizen={state['processed_count']['citizen']} "
                    f"errors={state['error_count'].get('signal', 0) + state['error_count'].get('citizen', 0)}"
                )
            
            if state["active_stream"] == "idle":
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        logger.info("⏹️ Orchestrator stopped by user")
    except Exception as e:
        logger.error(f"💥 Orchestrator crashed: {e}")
        raise

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    ensure_collection()
    run_orchestrator()
