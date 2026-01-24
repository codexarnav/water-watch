import time
import uuid
import threading
from typing import Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    BinaryQuantization,
    BinaryQuantizationConfig,
    PointStruct,
    PayloadSchemaType,
)

from agents.agent4 import run_agent4_parallel


EMBEDDING_MEMORY: Dict[str, Dict[str, Any]] = {}
MEM_LOCK = threading.Lock()

QDRANT_URL = "http://localhost:6333"
COLLECTION = "water_memory"
client = QdrantClient(url=QDRANT_URL)

BATCH_SIZE = 100
FLUSH_INTERVAL_SEC = 2
MAX_RETRY = 3



def ensure_collection():
    if client.collection_exists(COLLECTION):
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            # Core multimodal embeddings
            "semantic_bind": VectorParams(size=512, distance=Distance.COSINE),
            "semantic_image": VectorParams(size=512, distance=Distance.COSINE),
            "semantic_audio": VectorParams(size=512, distance=Distance.COSINE),
            "sensor_dense": VectorParams(size=256, distance=Distance.COSINE),

            # Video embeddings (512×4 → split + BQ)
            "semantic_video_0": VectorParams(
                size=512,
                distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
                on_disk=True,
            ),
            "semantic_video_1": VectorParams(
                size=512,
                distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
                on_disk=True,
            ),
            "semantic_video_2": VectorParams(
                size=512,
                distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
                on_disk=True,
            ),
            "semantic_video_3": VectorParams(
                size=512,
                distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
                on_disk=True,
            ),
        },
    )

    # ✅ Payload indexes for fast filtering / retrieval
    client.create_payload_index(COLLECTION, "source_id", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "modality", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "event_id", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "timestamp", PayloadSchemaType.DATETIME)
    client.create_payload_index(COLLECTION, "severity", PayloadSchemaType.FLOAT)

    print(f"[AGENT5] ✅ Qdrant collection created: {COLLECTION}")


# -------------------------------------------------
# 2) Video split utility (2048 -> 4*512)
# -------------------------------------------------
def split_video_embedding(video_vec: List[float]):
    if len(video_vec) != 2048:
        raise ValueError("semantic_video must be exactly 2048 dims (512×4)")
    return {
        "semantic_video_0": video_vec[0:512],
        "semantic_video_1": video_vec[512:1024],
        "semantic_video_2": video_vec[1024:1536],
        "semantic_video_3": video_vec[1536:2048],
    }


# -------------------------------------------------
# 3) Convert Hydro-Voxel -> Qdrant Point
# -------------------------------------------------
def voxel_to_point(hydro_voxel: Dict[str, Any]) -> PointStruct:
    vectors = {}

    payload = {
        **hydro_voxel.get("context", {}),
        "percept_id": hydro_voxel.get("percept_id"),
        "modality": hydro_voxel.get("modality", "unknown"),
        "ingested_at": hydro_voxel.get("ingested_at", time.time()),
        "raw_ref": hydro_voxel.get("raw_ref", {}),

        # Liquid Memory fields
        "reliability_score": hydro_voxel.get("context", {}).get("reliability_score", 1.0),
        "intervention_outcome": hydro_voxel.get("context", {}).get("intervention_outcome"),
    }

    v = hydro_voxel.get("vectors", {})

    # ✅ Core named vectors
    for key in ("semantic_bind", "semantic_image", "semantic_audio", "sensor_dense"):
        if key in v and v[key] is not None:
            vectors[key] = v[key]

    # ✅ Video split + quantization (collection config applies)
    if "semantic_video" in v and v["semantic_video"] is not None:
        vectors.update(split_video_embedding(v["semantic_video"]))

    # ✅ Sparse lexical stored in payload
    if "lexical_sparse" in v:
        payload["lexical_sparse"] = v["lexical_sparse"]

    return PointStruct(
        id=hydro_voxel.get("percept_id") or str(uuid.uuid4()),
        vector=vectors,
        payload=payload,
    )


# -------------------------------------------------
# 4) Snapshot & clear memory safely
# -------------------------------------------------
def snapshot_and_clear_memory() -> List[Dict[str, Any]]:
    with MEM_LOCK:
        voxels = list(EMBEDDING_MEMORY.values())
        EMBEDDING_MEMORY.clear()
    return voxels


# -------------------------------------------------ī
# 5) Batch upsert into Qdrant
# -------------------------------------------------
def upsert_points(points: List[PointStruct]):
    if not points:
        return

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,  # deterministic, stable in dev
    )


# -------------------------------------------------
# 6) Qdrant flush loop
# -------------------------------------------------
def qdrant_flush_loop():
    ensure_collection()
    print("[AGENT5] ✅ Qdrant flush loop started...")

    while True:
        time.sleep(FLUSH_INTERVAL_SEC)

        voxels = snapshot_and_clear_memory()
        if not voxels:
            continue

        # convert voxels -> points
        points: List[PointStruct] = []
        skipped = 0

        for voxel in voxels:
            try:
                points.append(voxel_to_point(voxel))
            except Exception as e:
                skipped += 1
                print("[AGENT5] ❌ voxel->point error:", e)

        # batch writes
        written = 0
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]

            for attempt in range(1, MAX_RETRY + 1):
                try:
                    upsert_points(batch)
                    written += len(batch)
                    break
                except Exception as e:
                    print(f"[AGENT5] ❌ Qdrant upsert failed attempt={attempt}: {e}")
                    time.sleep(0.5 * attempt)

        print(f"[AGENT5] ✅ Flush complete | written={written} skipped={skipped}")


# -------------------------------------------------
# 7) Main Orchestrator
# -------------------------------------------------
def run_system():
    """
    Runs Agent4 + Agent5 together safely.
    """
    print("[AGENT5] 🚀 Starting Hydro-Kernel Memory System...")

    # 1) start Agent4 ingestion loop in a background thread
    t_ingest = threading.Thread(
        target=run_agent4_parallel,
        args=(EMBEDDING_MEMORY, MEM_LOCK),
        daemon=True,
    )
    t_ingest.start()
    print("[AGENT5] ✅ Agent4 started (ingestion + embedding)")

    # 2) run Qdrant flush loop on main thread
    qdrant_flush_loop()


if __name__ == "__main__":
    run_system()
