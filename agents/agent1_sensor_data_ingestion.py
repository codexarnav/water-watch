import json
import time
from datetime import datetime, timezone
from typing import TypedDict, Dict, Any, Optional

import pandas as pd
from kafka import KafkaProducer, KafkaConsumer

from langgraph.graph import StateGraph, END

RAW_TOPIC = "sensor.raw"
CLEAN_TOPIC = "sensor.cleaned"
CSV_PATH = "water.csv"

def run_producer():
    data = pd.read_csv(CSV_PATH)

    producer = KafkaProducer(
        bootstrap_servers="127.0.0.1:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        retries=5
    )

    start_time = datetime.now(timezone.utc)

    print(f"[PRODUCER] Streaming {len(data)} rows to {RAW_TOPIC} ...")

    for i, row in data.iterrows():
        well_id = str(row["Site_Id"])
        ts = start_time.timestamp() + i
        iso_ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        msg = {
            "source_id": well_id,
            "timestamp": iso_ts,
            "readings": {
                "salinity_ppt": float(row["salinity (ppt)"]) if pd.notna(row["salinity (ppt)"]) else None,
                "dissolved_oxygen_mgL": float(row["Dissolved  Oxygen (mg/L)"]) if pd.notna(row["Dissolved  Oxygen (mg/L)"]) else None,
                "ph": float(row["pH (standard units)"]) if pd.notna(row["pH (standard units)"]) else None,
                "secchi_depth_m": float(row["Secchi Depth (m)"]) if pd.notna(row["Secchi Depth (m)"]) else None,
                "water_depth_m": float(row["Water Depth (m)"]) if pd.notna(row["Water Depth (m)"]) else None,
                "water_temp_c": float(row["Water Temp (°C)"]) if pd.notna(row["Water Temp (°C)"]) else None,
            }
        }

        producer.send(topic=RAW_TOPIC, key=well_id, value=msg)
        producer.flush()

        print(f"[PRODUCER] sent -> {well_id} @ {iso_ts}")
        time.sleep(1)  # ✅ per second stream

    producer.close()
    print("[PRODUCER] finished ✅")

class Agent1State(TypedDict):
    raw: Optional[Dict[str, Any]]
    clean: Optional[Dict[str, Any]]


def consume_raw(state: Agent1State) -> Agent1State:
    """
    Pull ONE message from Kafka RAW_TOPIC.
    """
    consumer = KafkaConsumer(
        RAW_TOPIC,
        bootstrap_servers="127.0.0.1:9092",
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="agent1-preprocess",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None
    )

    msg = next(iter(consumer))  # ✅ consume one record
    raw = msg.value

    print(f"[AGENT1] consumed raw from {RAW_TOPIC} | well={raw.get('source_id')}")

    consumer.close()
    return {"raw": raw, "clean": None}


def preprocess(state: Agent1State) -> Agent1State:
    """
    Preprocess raw record:
    - ensure numeric types
    - replace missing values (optional)
    - add quality flags
    """
    raw = state["raw"]
    if raw is None:
        return {"raw": None, "clean": None}

    readings = raw.get("readings", {})

    def to_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    clean_readings = {k: to_float(v) for k, v in readings.items()}

    # Quality Flags
    missing_count = sum(v is None for v in clean_readings.values())
    quality_flags = {
        "missing_count": missing_count,
        "missing_present": missing_count > 0,
        "schema_ok": True
    }

    clean = {
        "source_id": raw.get("source_id"),
        "timestamp": raw.get("timestamp"),
        "readings": clean_readings,
        "quality_flags": quality_flags
    }

    print(f"[AGENT1] preprocessed | missing={missing_count}")

    return {"raw": raw, "clean": clean}


def publish_clean(state: Agent1State) -> Agent1State:
    """
    Publish cleaned record to CLEAN_TOPIC.
    """
    clean = state["clean"]
    if clean is None:
        return state

    producer = KafkaProducer(
        bootstrap_servers="127.0.0.1:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        retries=5
    )

    well_id = clean["source_id"]
    producer.send(topic=CLEAN_TOPIC, key=well_id, value=clean)
    producer.flush()
    producer.close()

    print(f"[AGENT1] published clean -> {CLEAN_TOPIC} | well={well_id}")
    return state


def build_agent1_graph():
    graph = StateGraph(Agent1State)

    graph.add_node("consume_raw", consume_raw)
    graph.add_node("preprocess", preprocess)
    graph.add_node("publish_clean", publish_clean)

    graph.set_entry_point("consume_raw")
    graph.add_edge("consume_raw", "preprocess")
    graph.add_edge("preprocess", "publish_clean")
    graph.add_edge("publish_clean", END)

    return graph.compile()


def run_agent1_loop():
    """
    Runs Agent1 continuously:
    consume raw -> preprocess -> publish clean -> repeat
    """
    agent1 = build_agent1_graph()
    print("[AGENT1] running loop... CTRL+C to stop")

    while True:
        agent1.invoke({"raw": None, "clean": None})
        time.sleep(0.1)


# if __name__ == "__main__":
   
#     run_producer()

#     # 2) Start Agent1 in another terminal
#     #run_agent1_loop()
