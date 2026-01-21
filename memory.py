import json
import time
import threading
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from kafka import KafkaConsumer

# ✅ use your existing embedder
from embeddings import agent_b_perceive

SENSOR_TOPIC = "events.queue"

# -----------------------------
# Parallel Execution Config
# -----------------------------
MAX_WORKERS = 6            # load balancing factor
MAX_INFLIGHT = 200         # avoid infinite inflight futures


# -----------------------------
# Thread-safe In-Memory Embedding Store
# -----------------------------
EMBEDDING_MEMORY: Dict[str, Dict[str, Any]] = {}
MEM_LOCK = threading.Lock()


def make_consumer():
    return KafkaConsumer(
        SENSOR_TOPIC,
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="agent5-memory-parallel",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )


def sensor_event_to_routed_signal(sensor_event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Agent2 semantic event (sensor) into routed_signal for agent_b_perceive.
    """
    return {
        "modality": "text",  # sensor spike is textualized -> treat as text
        "payload": {"text": sensor_event["derived"]["semantic_text"]},
        "context": {
            "source_id": sensor_event["meta"]["source_id"],
            "timestamp": sensor_event["meta"]["timestamp"],
            "severity": sensor_event["derived"]["severity"],
            "tags": sensor_event["derived"]["tags"],
            "producer": sensor_event["meta"].get("producer", "agent2_stm_spike"),
            "event_id": sensor_event["meta"].get("event_id"),
        }
    }


def store_percept(percept: Dict[str, Any]) -> None:
    """
    Store percept in memory store (voxel-ready).
    """
    record = {
        "percept_id": percept["percept_id"],
        "modality": percept["modality"],
        "vectors": {
            "semantic_bind": percept["semantic_bind"],
            **({"lexical_sparse": percept["lexical_sparse"]} if "lexical_sparse" in percept else {})
        },
        "context": percept.get("context", {}),
        "raw_ref": percept.get("raw_ref", {}),
        "ingested_at": time.time(),
    }

    with MEM_LOCK:
        EMBEDDING_MEMORY[percept["percept_id"]] = record


def embed_and_store(sensor_event: Dict[str, Any]) -> Optional[str]:
    """
    Worker task:
    - convert kafka event -> routed_signal
    - call agent_b_perceive() (your embedder)
    - store percept
    """
    try:
        routed_signal = sensor_event_to_routed_signal(sensor_event)
        percept = agent_b_perceive(routed_signal)

        if percept is None:
            return None

        store_percept(percept)
        return percept["percept_id"]

    except Exception as e:
        print("[AGENT5] ❌ embedding failed:", e)
        return None


def run_agent5_parallel():
    """
    Main loop:
    - consume kafka sensor events
    - submit embedding tasks into executor pool
    - load balance tasks across workers
    - control inflight futures
    """
    consumer = make_consumer()

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    inflight = set()

    print(f"[AGENT5] ✅ Parallel Trinity Memory Agent running | workers={MAX_WORKERS}")

    for msg in consumer:
        event = msg.value

        # safety: only sensor events
        if event.get("event_type") != "sensor":
            continue

        # backpressure: don't allow too many inflight tasks
        while len(inflight) >= MAX_INFLIGHT:
            done = {f for f in inflight if f.done()}
            inflight -= done
            time.sleep(0.01)

        # submit task
        fut = executor.submit(embed_and_store, event)
        inflight.add(fut)

        # print completion logs (non-blocking)
        done_now = {f for f in inflight if f.done()}
        for f in done_now:
            inflight.remove(f)
            pid = f.result()
            if pid:
                with MEM_LOCK:
                    ctx = EMBEDDING_MEMORY[pid]["context"]
                print(f"[AGENT5] ✅ Stored percept | id={pid} well={ctx.get('source_id')}")


if __name__ == "__main__":
    run_agent5_parallel()