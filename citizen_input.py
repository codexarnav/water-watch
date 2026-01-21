"""
Agent A: Human Signal Intake & Routing

Responsibility:
- Normalize raw human inputs (WhatsApp, App, Portal)
- Preserve temporal + spatial context
- Split multimodal input into per-modality routed signals
- Perform ZERO perception or interpretation

Downstream:
- Outputs are consumed by Layer B Perception Agents
"""

from typing import Dict, List, Optional
from datetime import datetime
import geohash2
import uuid

# -----------------------------
# Data Schemas (Lightweight)
# -----------------------------

RawHumanSignal = Dict
RoutedSignal = Dict


# -----------------------------
# Utility
# -----------------------------

def compute_geohash(
    location: Optional[Dict[str, float]],
    precision: int = 7
) -> Optional[str]:
    """
    Convert lat/lon into geohash.
    """
    if not location:
        return None

    lat = location.get("lat")
    lon = location.get("lon")

    if lat is None or lon is None:
        return None

    return geohash2.encode(lat, lon, precision=precision)


# -----------------------------
# Agent A Core Function
# -----------------------------

def agent_a_route(raw_signal: RawHumanSignal) -> List[RoutedSignal]:

    routed_signals: List[RoutedSignal] = []

    source = raw_signal.get("source")
    timestamp = raw_signal.get("timestamp")
    location = raw_signal.get("location")
    content = raw_signal.get("content", {})

    if isinstance(timestamp, (int, float)):
        timestamp = datetime.utcfromtimestamp(timestamp).isoformat()

    event_id = str(uuid.uuid4())  # 🔑 ONE event identity

    context = {
        "event_id": event_id,
        "timestamp": timestamp,
        "geohash": compute_geohash(location),
        "source": source
    }

    if "text" in content and content["text"]:
        routed_signals.append({
            "modality": "text",
            "payload": {"text": content["text"]},
            "context": context
        })

    if "image_uri" in content and content["image_uri"]:
        routed_signals.append({
            "modality": "image",
            "payload": {"image_uri": content["image_uri"]},
            "context": context
        })

    if "audio_uri" in content and content["audio_uri"]:
        routed_signals.append({
            "modality": "audio",
            "payload": {"audio_uri": content["audio_uri"]},
            "context": context
        })

    if "video_uri" in content and content["video_uri"]:
        routed_signals.append({
            "modality": "video",
            "payload": {"video_uri": content["video_uri"]},
            "context": context
        })

    return routed_signals



# -----------------------------
# Example (for testing only)
# -----------------------------

if __name__ == "__main__":
    example_input = {
        "source": "whatsapp",
        "timestamp": 1712345678,
        "sender_id": "9198xxxxxx",
        "location": {
            "lat": 28.6139,
            "lon": 77.2090
        },
        "content": {
            "text": "Water is brown after rain",
            "image_uri": "/data/well12.jpg"
        }
    }

    outputs = agent_a_route(example_input)
    for o in outputs:
        print(o)




