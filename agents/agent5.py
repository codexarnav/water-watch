"""
Qdrant Hydro-Voxel Writer
------------------------
• Does NOT modify hydro-voxel creation
• Selective Binary Quantization (video only)
• Named vectors
• Liquid Memory ready (payload-driven)
"""

from typing import Dict, Any
import uuid
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    BinaryQuantization,
    BinaryQuantizationConfig,
    PointStruct,
)

# -------------------------------------------------
# Qdrant Configuration
# -------------------------------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "water_memory"

client = QdrantClient(url=QDRANT_URL)


# -------------------------------------------------
# Collection Schema (ONE-TIME)
# -------------------------------------------------
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

            # Video embeddings (512×4 → split, BQ)
            "semantic_video_0": VectorParams(
                size=512, distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
                on_disk=True
            ),
            "semantic_video_1": VectorParams(
                size=512, distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
                on_disk=True
            ),
            "semantic_video_2": VectorParams(
                size=512, distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
                on_disk=True
            ),
            "semantic_video_3": VectorParams(
                size=512, distance=Distance.COSINE,
                quantization_config=BinaryQuantization(
                    binary=BinaryQuantizationConfig(always_ram=True)
                ),
                on_disk=True
            ),
        }
    )


# -------------------------------------------------
# Video Split Utility (NO mutation)
# -------------------------------------------------
def split_video_embedding(video_vec):
    """
    Expects exactly 2048-dim vector.
    """
    return {
        "semantic_video_0": video_vec[0:512],
        "semantic_video_1": video_vec[512:1024],
        "semantic_video_2": video_vec[1024:1536],
        "semantic_video_3": video_vec[1536:2048],
    }


# -------------------------------------------------
# Store ONE Hydro-Voxel
# -------------------------------------------------
def store_hydro_voxel(hydro_voxel: Dict[str, Any]):
    """
    hydro_voxel = entry from EMBEDDING_MEMORY
    """

    vectors = {}
    payload = {
        **hydro_voxel.get("context", {}),
        "percept_id": hydro_voxel["percept_id"],
        "modality": hydro_voxel["modality"],
        "ingested_at": hydro_voxel.get("ingested_at", time.time()),
        "raw_ref": hydro_voxel.get("raw_ref", {}),
        # Liquid Memory fields
        "reliability_score": hydro_voxel.get("context", {}).get("reliability_score", 1.0),
        "intervention_outcome": hydro_voxel.get("context", {}).get("intervention_outcome"),
    }

    v = hydro_voxel.get("vectors", {})

    # -----------------------------
    # Core embeddings (no quant)
    # -----------------------------
    for key in ("semantic_bind", "semantic_image", "semantic_audio", "sensor_dense"):
        if key in v:
            vectors[key] = v[key]

    # -----------------------------
    # Video embedding (BQ only)
    # -----------------------------
    if "semantic_video" in v:
        video_vec = v["semantic_video"]

        if len(video_vec) != 2048:
            raise ValueError("semantic_video must be exactly 512×4 (2048 dims)")

        vectors.update(split_video_embedding(video_vec))

    # -----------------------------
    # Sparse lexical (payload only)
    # -----------------------------
    if "lexical_sparse" in v:
        payload["lexical_sparse"] = v["lexical_sparse"]

    point = PointStruct(
        id=hydro_voxel["percept_id"] or str(uuid.uuid4()),
        vector=vectors,
        payload=payload,
    )

    client.upsert(
        collection_name=COLLECTION,
        points=[point],
        wait=False,
    )


# -------------------------------------------------
# Bulk Flush (Multi-Agent Safe)
# -------------------------------------------------
def flush_embedding_memory(embedding_memory: Dict[str, Dict[str, Any]]):
    ensure_collection()

    for voxel in embedding_memory.values():
        store_hydro_voxel(voxel)


if __name__ == "__main__":
    import numpy as np
    
    print("\\n[KERNEL] Running Component Test...")
    
    # ensure db exists
    ensure_collection()
    
    # 1. Create Dummy Hydro-Voxel (Simulating output from memory.py for Text)
    # Matches 'semantic_bind' (512 dims)
    '''
    "lexical_sparse": {
                "indices": [101, 2034, 4521],
                "values": [0.5, 0.8, 0.2]
            }
    '''
    dummy_text_voxel = {
        "percept_id": str(uuid.uuid4()),
        "modality": "text",
        "vectors": {
            "semantic_bind": np.random.rand(512).tolist(),
            "lexical_sparse": {
                "indices": [101, 2034, 4521],
                "values": [0.5, 0.8, 0.2]
            }
        },
        "context": {
            "source_id": "Well_Test_fairness",
            "event_id": "evt_demo_fair",
            "timestamp": "2024-01-01T12:00:00Z", 
            "reliability_score": 0.95,
            "geohash": "ttnwv6d"
        },
        "raw_ref": {
            "text": "Correct fairness test voxel."
        },
        "ingested_at": time.time()
    }

    print(f"1. Generated Dummy Voxel: {dummy_text_voxel['percept_id']}")

    # 2. Store it
    try:
        store_hydro_voxel(dummy_text_voxel)
        print("2. Upsert Successful!")
    except Exception as e:
        print(f"2. Upsert Failed: {e}")
        import traceback
        traceback.print_exc()
        
    # 3. Verify Retrieval
    print("3. Verifying storage...")
    try:
        res = client.retrieve(
            collection_name=COLLECTION,
            ids=[dummy_text_voxel["percept_id"]],
            with_payload=True,
            with_vectors=True
        )
        
        if res:
            print(f"   Found point: {res[0].id}")
            print(f"   Modality: {res[0].payload.get('modality')}")
            print("   Vectors retrieved: ", list(res[0].vector.keys()) if res[0].vector else "None")
        else:
            print("   Point not found immediately (might be async).")
    except Exception as e:
        print(f"   Retrieval Failed: {e}")
