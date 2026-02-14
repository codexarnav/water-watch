import json
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from kafka import KafkaConsumer


from agents.agent3 import agent_b_perceive

SENSOR_TOPIC = "events.queue"


MAX_WORKERS = 6
MAX_INFLIGHT = 200


def run_agent4_parallel(embedding_memory: Dict[str, Dict[str, Any]], mem_lock):
    """
    Agent4 callable wrapper:
    - Consumes sensor events from Kafka
    - Embeds in parallel using agent_b_perceive
    - Stores hydro-voxels into provided embedding_memory (thread-safe)

    This function is designed to be called directly by Agent5.
    """

    # -----------------------------
    # Kafka Consumer
    # -----------------------------
    def make_consumer():
        return KafkaConsumer(
            SENSOR_TOPIC,
            bootstrap_servers="localhost:9092",
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="agent5-memory-parallel",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

    # -----------------------------
    # Convert sensor event -> routed signal
    # -----------------------------
    def sensor_event_to_routed_signal(sensor_event: Dict[str, Any], kafka_meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "modality": "sensor",
            "payload": {
                # raw sensor values if present
                "sensor_values": sensor_event.get("payload", {}).get("sensor_values"),

                # semantic summary from Agent2
                "text": sensor_event.get("derived", {}).get("semantic_text", ""),

                # keep severity/tags
                "severity": sensor_event.get("derived", {}).get("severity"),
                "tags": sensor_event.get("derived", {}).get("tags", []),
            },
            "context": {
                "source_id": sensor_event["meta"].get("source_id"),
                "timestamp": sensor_event["meta"].get("timestamp"),
                "severity": sensor_event.get("derived", {}).get("severity"),
                "tags": sensor_event.get("derived", {}).get("tags", []),
                "producer": sensor_event["meta"].get("producer", "agent2_stm_spike"),
                "event_id": sensor_event["meta"].get("event_id"),

                # ✅ ordering safety
                "kafka_partition": kafka_meta["partition"],
                "kafka_offset": kafka_meta["offset"],
            },
            "raw_ref": {
                "original_event": sensor_event
            }
        }

    # -----------------------------
    # Store percept -> hydro-voxel
    # -----------------------------
    def store_percept(percept: Dict[str, Any]) -> None:
        """
        Convert percept (Agent3 output) into hydro-voxel format,
        then store it in the shared memory dict.
        """

        vectors = {}

        # ✅ store ALL embeddings if available
        for key in [
            "semantic_bind",
            "sensor_dense",
            "semantic_image",
            "semantic_audio",
            "semantic_video",  # 2048 dims (Agent5 splits)
        ]:
            if key in percept and percept[key] is not None:
                vectors[key] = percept[key]

        # ✅ sparse lexical stored for hybrid memory
        if "lexical_sparse" in percept and percept["lexical_sparse"] is not None:
            vectors["lexical_sparse"] = percept["lexical_sparse"]

        record = {
            "percept_id": percept["percept_id"],
            "modality": percept.get("modality", "unknown"),
            "vectors": vectors,
            "context": percept.get("context", {}),
            "raw_ref": percept.get("raw_ref", {}),
            "ingested_at": time.time(),
        }

        with mem_lock:
            embedding_memory[percept["percept_id"]] = record

    # -----------------------------
    # Worker task: embed + store
    # -----------------------------
    def embed_and_store(sensor_event: Dict[str, Any], kafka_meta: Dict[str, Any]) -> Optional[str]:
        try:
            routed_signal = sensor_event_to_routed_signal(sensor_event, kafka_meta)
            percept = agent_b_perceive(routed_signal)

            if percept is None:
                return None

            store_percept(percept)
            return percept["percept_id"]

        except Exception as e:
            print("[AGENT4] ❌ embedding failed:", e)
            return None

    # -----------------------------
    # Main consume loop
    # -----------------------------
    consumer = make_consumer()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    inflight = set()

    print(f"[AGENT4] ✅ Running | Kafka={SENSOR_TOPIC} | workers={MAX_WORKERS}")

    for msg in consumer:
        event = msg.value

        # safety: only sensor events
        if event.get("event_type") != "sensor":
            continue

        kafka_meta = {"partition": msg.partition, "offset": msg.offset}

        # backpressure: control inflight futures
        while len(inflight) >= MAX_INFLIGHT:
            done = {f for f in inflight if f.done()}
            inflight -= done
            time.sleep(0.01)

        fut = executor.submit(embed_and_store, event, kafka_meta)
        inflight.add(fut)

        # completion logs (non-blocking)
        done_now = {f for f in inflight if f.done()}
        for f in done_now:
            inflight.remove(f)
            pid = f.result()

            if pid:
                with mem_lock:
                    ctx = embedding_memory[pid].get("context", {})
                    vectors = embedding_memory[pid].get("vectors", {})

                print(
                    f"[AGENT4] ✅ Stored percept | id={pid} "
                    f"well={ctx.get('source_id')} "
                    f"offset={ctx.get('kafka_offset')} "
                    f"vectors={list(vectors.keys())}"
                )
