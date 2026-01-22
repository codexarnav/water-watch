import json
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Any, Optional, TypedDict, List

import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from langgraph.graph import StateGraph, END


INPUT_TOPIC = "sensor.cleaned"
OUTPUT_TOPIC = "events.queue"

STM_SECONDS = 1800    
MIN_POINTS = 10            
Z_THRESH = 3.0             


STM: Dict[str, deque] = defaultdict(deque)

class Agent2State(TypedDict):
    clean_msg: Optional[Dict[str, Any]]
    well_id: Optional[str]
    spike: Optional[Dict[str, Any]]
    semantic_event: Optional[Dict[str, Any]]


def parse_iso(ts: str) -> float:
    """
    Converts ISO timestamp to epoch seconds.
    """
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def compute_z(history: List[float], current: float):
    """
    Computes z-score using rolling history.
    """
    mean = float(np.mean(history))
    std = float(np.std(history))
    if std < 1e-6:
        std = 1e-6
    z = (current - mean) / std
    return mean, std, float(z)


def build_semantic_text(well_id: str, metric: str, value: float, z: float) -> str:
    return (
        f"Spike detected in {metric} for well {well_id}. "
        f"Current={value:.2f}, z-score={z:.2f}. "
        f"Possible contamination episode."
    )


def make_consumer():
    return KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="agent2-stm-spike",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None
    )


def make_producer():
    return KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        retries=5
    )


def load_from_kafka(state: Agent2State) -> Agent2State:
    """
    Node 1: Load ONE message from sensor.cleaned
    """
    consumer = make_consumer()
    msg = next(iter(consumer))         
    consumer.close()

    clean_msg = msg.value
    well_id = clean_msg["source_id"]

    print(f"[AGENT2] Loaded clean message | well={well_id}")

    return {
        "clean_msg": clean_msg,
        "well_id": well_id,
        "spike": None,
        "semantic_event": None
    }


def update_stm(state: Agent2State) -> Agent2State:
    """
    Node 2: Update 1-hour STM per well
    """
    clean_msg = state["clean_msg"]
    well_id = state["well_id"]

    if clean_msg is None or well_id is None:
        return state
    
    ts = parse_iso(clean_msg["timestamp"])
    readings = clean_msg["readings"]
    STM[well_id].append((ts, readings))
    while STM[well_id] and (ts - STM[well_id][0][0] > STM_SECONDS):
        STM[well_id].popleft()

    return state


def spike_detection(state: Agent2State) -> Agent2State:
    """
    Node 3: spike detection based on rolling z-score
    """
    clean_msg = state["clean_msg"]
    well_id = state["well_id"]

    if clean_msg is None or well_id is None:
        return state

    if len(STM[well_id]) < MIN_POINTS:
        # not enough points to detect
        return state

    readings = clean_msg["readings"]
    metrics = ["ph", "dissolved_oxygen_mgL", "water_temp_c", "salinity_ppt"]

    anomalies = []
    for metric in metrics:
        current_val = readings.get(metric)
        if current_val is None:
            continue

        history_vals = [
            r.get(metric) for (_, r) in STM[well_id]
            if r.get(metric) is not None
        ]

        if len(history_vals) < MIN_POINTS:
            continue

        baseline = history_vals[:-1] if len(history_vals) > 1 else history_vals
        mean, std, z = compute_z(baseline, float(current_val))

        if abs(z) >= Z_THRESH:
            anomalies.append({
                "metric": metric,
                "value": float(current_val),
                "z": z,
                "mean": mean,
                "std": std
            })

    if not anomalies:
        return state

    # strongest anomaly
    spike = max(anomalies, key=lambda x: abs(x["z"]))
    state["spike"] = spike

    print(f"[AGENT2] Spike Detected | well={well_id} metric={spike['metric']} z={spike['z']:.2f}")
    return state


def push_to_event_queue(state: Agent2State) -> Agent2State:
    """
    Node 4: Create semantic event + push to events.queue
    """
    spike = state["spike"]
    clean_msg = state["clean_msg"]
    well_id = state["well_id"]

    if spike is None or clean_msg is None or well_id is None:
        # no spike → no event
        return state


    severity = min(1.0, abs(spike["z"]) / 6.0)

    semantic_text = build_semantic_text(well_id, spike["metric"], spike["value"], spike["z"])

    semantic_event = {
        "meta": {
            "event_id": str(int(time.time() * 1000)),
            "source_id": well_id,
            "timestamp": clean_msg["timestamp"],
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

    producer = make_producer()
    producer.send(OUTPUT_TOPIC, key=well_id, value=semantic_event)
    producer.flush()
    producer.close()

    print(f"[AGENT2] Event pushed to {OUTPUT_TOPIC} | well={well_id}")
    state["semantic_event"] = semantic_event
    return state


def build_agent2_graph():
    g = StateGraph(Agent2State)

    g.add_node("load", load_from_kafka)
    g.add_node("stm", update_stm)
    g.add_node("spike", spike_detection)
    g.add_node("event", push_to_event_queue)

    g.set_entry_point("load")

    g.add_edge("load", "stm")
    g.add_edge("stm", "spike")
    g.add_edge("spike", "event")
    g.add_edge("event", END)

    return g.compile()


def run_agent2():
    agent2 = build_agent2_graph()
    print("[AGENT2] Running STM + Spike Detection Agent... CTRL+C to stop")

    while True:
        agent2.invoke({
            "clean_msg": None,
            "well_id": None,
            "spike": None,
            "semantic_event": None
        })
        time.sleep(0.05)


# if __name__ == "__main__":
#     run_agent2()
