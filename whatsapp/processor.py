import os
import time
import json
import logging
import base64
import requests
import subprocess
import threading
from pathlib import Path

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
# We still use the file for backup/persistence, but we process from stdout
MESSAGES_FILE = BASE_DIR / "whatsapp" / "messages.jsonl"
API_URL = "http://localhost:8000/api/multimodal/upload"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WhatsAppProcessor")

def get_base64_content(file_path):
    """Reads a file and returns its base64 encoded string."""
    try:
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            logger.error(f"File not found: {full_path}")
            return None
            
        with open(full_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding file {file_path}: {e}")
        return None

def process_message_data(data):
    """Processes a message data dict."""
    try:
        modality = data.get("modality")
        payload = data.get("payload", {})
        context = data.get("context", {})
        
        logger.info(f"Processing new {modality} message...")
        
        api_payload = {
            "metadata": {
                "source": "whatsapp",
                "timestamp": context.get("timestamp"),
                "geohash": context.get("geohash")
            }
        }
        
        content = None
        
        if modality == "text":
            api_payload["type"] = "text"
            api_payload["content"] = payload.get("text")
            
        elif modality == "image":
            api_payload["type"] = "image"
            uri = payload.get("image_uri")
            api_payload["content"] = get_base64_content(uri)
            api_payload["metadata"]["filename"] = uri
            
        elif modality == "video":
            api_payload["type"] = "video"
            uri = payload.get("video_uri")
            api_payload["content"] = get_base64_content(uri)
            api_payload["metadata"]["filename"] = uri

        elif modality == "audio":
            api_payload["type"] = "audio"
            uri = payload.get("audio_uri")
            api_payload["content"] = get_base64_content(uri)
            api_payload["metadata"]["filename"] = uri
            
        else:
            logger.warning(f"Unknown modality: {modality}")
            return

        if not api_payload.get("content"):
            logger.warning(f"No content extracted for {modality}")
            return

        # Send to API
        logger.info(f"Sending {modality} to Backend API...")
        try:
            response = requests.post(API_URL, json=api_payload)
            if response.status_code == 200:
                logger.info(f"SUCCESS: {response.json()}")
            else:
                logger.error(f"API Error ({response.status_code}): {response.text}")
        except Exception as e:
            logger.error(f"Connection Error: {e}")

    except Exception as e:
        logger.error(f"Error processing message data: {e}")

def run_scraper_and_process():
    """Runs the scraper and processes its stdout."""
    scraper_script = BASE_DIR / "whatsapp" / "scraper.py"
    
    logger.info(f"Launching scraper: {scraper_script}")
    
    # Run unbuffered
    process = subprocess.Popen(
        ["python", "-u", str(scraper_script)], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        cwd=BASE_DIR / "whatsapp"
    )

    # Helper to read stderr in a separate thread to prevent blocking
    def read_stderr(pipe):
        for line in pipe:
            print(f"[Scraper Log] {line.strip()}")
            
    stderr_thread = threading.Thread(target=read_stderr, args=(process.stderr,), daemon=True)
    stderr_thread.start()
    
    try:
        # Read stdout line by line
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is JSON (our metadata)
            try:
                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    # Check if it has the expected keys to be a message
                    if "modality" in data and "payload" in data:
                        process_message_data(data)
                    else:
                        print(f"[Scraper Output] {line}")
                else:
                    # Normal print output
                    print(f"[Scraper Output] {line}")
            except json.JSONDecodeError:
                print(f"[Scraper Output] {line}")
                
        process.wait()
        logger.info(f"Scraper process exited with code {process.returncode}")
        
    except KeyboardInterrupt:
        logger.info("Stopping scraper...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    run_scraper_and_process()
