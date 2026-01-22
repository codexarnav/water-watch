# Water-Watch Orchestration Agent

## Overview

The **Water-Watch Orchestration Agent** is a LangGraph-based system that coordinates two independent data processing pipelines:

1. **Signal Stream**: IoT sensor data from Kafka
2. **Citizen Stream**: Multimodal citizen reports (text, images, audio, video)

Both streams are processed, embedded, and stored in Qdrant vector database with comprehensive error handling.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              ORCHESTRATOR                        │
│                                                  │
│  ┌─────────────┐         ┌──────────────┐      │
│  │   SIGNAL    │         │   CITIZEN    │      │
│  │   STREAM    │         │   STREAM     │      │
│  └─────────────┘         └──────────────┘      │
│        │                        │               │
│        ▼                        ▼               │
│   Kafka Input              API/Queue Input      │
│        │                        │               │
│        ▼                        ▼               │
│   Agent1: Preprocess      AgentA: Route         │
│        │                        │               │
│        ▼                        ▼               │
│   Agent2: Spike Detect    AgentB: Embed         │
│        │                        │               │
│        ▼                        ▼               │
│   Agent5: Embed          Memory: Voxel          │
│        │                        │               │
│        └────────┬───────────────┘               │
│                 ▼                                │
│           Kernel: Storage                        │
│                 │                                │
│                 ▼                                │
│          Qdrant Vector DB                        │
└─────────────────────────────────────────────────┘
```

---

## Features

### ✅ Dual-Stream Processing
- Independent signal and citizen pipelines
- Dynamic stream prioritization (citizen > signal)
- Isolated state management prevents cross-contamination

### ✅ Error Handling
- **Retry Logic**: Exponential backoff (1s, 5s, 15s)
- **Circuit Breaker**: Stops processing if error rate > threshold
- **Dead Letter Queue**: Failed messages logged for manual recovery
- **Max Retries**: 3 attempts per message

### ✅ Monitoring & Logging
- Structured logging with stream/stage context
- Metrics tracking (processed count, error count)
- Circuit breaker status monitoring

### ✅ Idempotent Storage
- Uses `percept_id` as Qdrant point ID
- Prevents duplicates on retry
- At-least-once delivery guarantee

### ✅ Context Preservation
**From citizen inputs:**
- ✅ Geohash (location)
- ✅ Timestamp
- ✅ Event metadata
- ✅ Raw references

**From sensor data:**
- ✅ Source ID (well ID)
- ✅ Timestamp
- ✅ Spike severity
- ✅ Quality flags

---

## Prerequisites

```bash
# Install dependencies
pip install langgraph kafka-python qdrant-client

# Required services
- Kafka (localhost:9092)
- Qdrant (localhost:6333)
```

---

## Configuration

Edit the `Config` dataclass in `orchestration_agent.py`:

```python
@dataclass
class Config:
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_RAW_TOPIC: str = "sensor.raw"
    KAFKA_CLEAN_TOPIC: str = "sensor.cleaned"
    KAFKA_EVENT_TOPIC: str = "events.queue"
    
    # Processing
    MAX_RETRIES: int = 3
    RETRY_DELAYS: List[int] = [1, 5, 15]
    
    # Circuit breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 10
    CIRCUIT_BREAKER_WINDOW_SEC: int = 60
    
    # Dead letter queue
    DLQ_PATH: str = "dead_letter_queue.jsonl"
```

---

## Usage

### Basic Usage

```bash
python orchestration_agent.py
```

### With Citizen Input (Programmatic)

```python
from orchestration_agent import add_citizen_input

# Add a citizen report
citizen_report = {
    "source": "whatsapp",
    "timestamp": 1706092800,
    "sender_id": "919812345678",
    "location": {
        "lat": 28.6139,
        "lon": 77.2090
    },
    "content": {
        "text": "Water is brown after rain",
        "image_uri": "/path/to/image.jpg"
    }
}

add_citizen_input(citizen_report)
```

### Check Dead Letter Queue

```bash
# View failed messages
cat dead_letter_queue.jsonl | jq
```

---

## Data Flow

### Signal Stream Flow

```
1. Kafka: sensor.raw
   ↓
2. signal_intake_node
   → Consumes one message
   ↓
3. signal_preprocess_node
   → Validates, cleans, adds quality flags
   → Publishes to sensor.cleaned
   ↓
4. signal_spike_detection_node
   → STM-based rolling z-score
   → If spike: create semantic event
   → If no spike: done
   ↓
5. signal_embed_node
   → Convert event to routed_signal
   → Generate embeddings (CLIP text + SPLADE++)
   → Create voxel structure
   ↓
6. storage_writer_node
   → Write to Qdrant
   ✓ Done
```

### Citizen Stream Flow

```
1. Citizen Input Queue/API
   ↓
2. citizen_intake_node
   → Get next input
   ↓
3. citizen_route_node
   → Split by modality (text, image, audio, video)
   → Preserve context (geohash, timestamp, event_id)
   ↓
4. citizen_embed_node
   → For each modality:
     - Text: CLIP + SPLADE++
     - Image: CLIP
     - Audio: CLAP
     - Video: CLIP (frame sampling)
   ↓
5. citizen_voxel_creation_node
   → Create voxel structures
   → Preserve context from routing
   ↓
6. storage_writer_node
   → Write to Qdrant
   ✓ Done
```

---

## Error Handling

### Retry Behavior

| Attempt | Delay | Action |
|---------|-------|--------|
| 1st | 0s | Immediate |
| 2nd | 1s | Retry after 1s |
| 3rd | 5s | Retry after 5s |
| 4th | 15s | Retry after 15s |
| Failed | - | Send to DLQ |

### Circuit Breaker

- **Opens when**: 10 failures in 60 seconds
- **Effect**: Stream processing halted
- **Recovery**: Automatically resets on next success

### Dead Letter Queue Format

```json
{
  "timestamp": "2026-01-22T11:23:45.123456",
  "stream": "citizen",
  "error": "Embedding failed: CUDA out of memory",
  "data": {
    "raw": {...},
    "routed_signals": [...]
  }
}
```

---

## Monitoring

### Log Format

```
2026-01-22 11:23:45 - orchestrator - INFO - 🎯 Routing to SIGNAL stream
2026-01-22 11:23:45 - orchestrator - INFO - 📥 [SIGNAL] Consumed raw message | well=Well_123
2026-01-22 11:23:45 - orchestrator - INFO - ✅ [SIGNAL] Preprocessed | well=Well_123
2026-01-22 11:23:45 - orchestrator - INFO - 🔔 [SIGNAL] Spike detected | well=Well_123
2026-01-22 11:23:46 - orchestrator - INFO - 🧠 [SIGNAL] Embedded | percept_id=abc-123
2026-01-22 11:23:46 - orchestrator - INFO - 💾 [STORAGE] Stored 1 voxels | stream=signal
```

### Metrics Log

```
2026-01-22 11:24:00 - orchestrator - INFO - 📈 Metrics | signal=10 citizen=5 errors_signal=0 errors_citizen=1
```

---

## State Schema

```python
{
    "active_stream": "signal" | "citizen" | "idle",
    
    # Signal stream
    "signal_state": {
        "raw": {...},
        "clean": {...},
        "event": {...},
        "percept": {...},
        "voxel": {...}
    },
    "signal_status": "idle" | "preprocessing" | "spike_detection" | "embedding" | "storing" | "done" | "error",
    "signal_error": "Error message",
    "signal_retry_count": 2,
    
    # Citizen stream
    "citizen_state": {
        "raw": {...},
        "routed_signals": [...],
        "percepts": [...],
        "voxels": [...]
    },
    "citizen_status": "idle" | "routing" | "embedding" | "voxel_creation" | "storing" | "done" | "error",
    "citizen_error": None,
    "citizen_retry_count": 0,
    
    # Storage
    "storage_queue": [...],
    "storage_status": "idle",
    "storage_error": None,
    
    # Metrics
    "processed_count": {"signal": 10, "citizen": 5},
    "error_count": {"signal": 0, "citizen": 1},
    "last_processed_at": 1706092846.123
}
```

---

## Testing

See `test_orchestrator.py` for examples:

```bash
python test_orchestrator.py
```

---

## File Dependencies

The orchestrator imports from:

| File | Purpose |
|------|---------|
| `citizen_input.py` | `agent_a_route()` - Routes citizen input by modality |
| `citizen_embed.py` | `agent_b_perceive()` - Generates multimodal embeddings |
| `kernel.py` | `ensure_collection()`, `store_hydro_voxel()` - Qdrant storage |

**Note**: Signal processing logic (`preprocess_sensor_data`, `detect_spike`) is embedded directly in the orchestrator to avoid circular dependencies.

---

## Troubleshooting

### "No Kafka brokers available"
```bash
# Check Kafka is running
docker ps | grep kafka

# Or start Kafka
docker-compose up -d kafka
```

### "Qdrant connection refused"
```bash
# Check Qdrant is running
curl http://localhost:6333

# Or start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### "Circuit breaker opened"
```bash
# Check dead letter queue for errors
cat dead_letter_queue.jsonl | tail -n 20 | jq

# Check logs for root cause
# Fix the issue and restart orchestrator
```

### "Embedding failed: CUDA out of memory"
```python
# In citizen_embed.py, switch to CPU
device = "cpu"  # instead of "cuda"
```

---

## Performance

### Expected Throughput

- **Signal Stream**: 100-200 messages/second
- **Citizen Stream**: 50-100 messages/second
- **Combined**: 150-300 messages/second

### Latency (p99)

- **Signal Processing**: < 500ms
- **Citizen Processing**: < 2s (including embedding)
- **Storage Write**: < 100ms

---

## Next Steps

1. **Add citizen input API**: Replace in-memory queue with REST API or Kafka topic
2. **Add Prometheus metrics**: For production monitoring
3. **Add health checks**: `/health` endpoint
4. **Deploy to Kubernetes**: Use provided manifests in `deployment/`

---

## License

MIT
