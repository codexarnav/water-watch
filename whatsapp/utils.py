import os
import time
import json
from datetime import datetime
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"
VIDEOS_DIR = BASE_DIR / "videos"
AUDIO_DIR = BASE_DIR / "audio"

# Ensure directories exist
for d in [IMAGES_DIR, VIDEOS_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def get_timestamp_iso():
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_filename(prefix="msg", extension="txt"):
    """Generates a unique filename with timestamp."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def save_metadata(data, filename="metadata.json"):
    """Saves metadata to a JSON lines file and prints it."""
    # Print to stdout
    print(json.dumps(data), flush=True)
    
    # Save to file
    messages_file = BASE_DIR / "whatsapp" / "messages.jsonl"
    with open(messages_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
