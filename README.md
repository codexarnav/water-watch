# Water Watch System

An intelligent, multi-agent AI system for real-time water quality monitoring, risk forecasting, multimodal input processing, and automated government alerts.

---

## 🏗️ System Architecture

Water Watch is built on a **9-Agent Pipeline** that processes water quality data through multiple stages of analysis, enrichment, and alert generation. The system supports both **sensor streams** and **multimodal user inputs** (text, images, audio, video).

### 📊 Data Flow Overview

```
Input Sources (Sensor/WhatsApp/Direct)
            ↓
    [AGENT 1] Ingestion & Preprocessing
            ↓
    [AGENT 2] Spike Detection (Z-Score Analysis)
            ↓
    [AGENT 3] Perception & Feature Extraction
    [AGENT 4] Embedding Generation (Multimodal)
    [AGENT 5] Vector Memory Storage (Qdrant)
            ↓
    [AGENT 7] Liquid Memory Retrieval (MMR-Reranking)
            ↓
    [AGENT 6] Forecasting & Risk Scoring
    [AGENT 8] QA/RAG (Contextual Q&A)
    [AGENT 9] Recommendations & Actions
            ↓
    [ALERT ENGINE] → Government Alert Dispatch
```

---

## 📂 Project Structure

| Directory / File | Purpose |
| :--- | :--- |
| **`main_script.py`** | **Master Orchestrator**: LangGraph-based pipeline that coordinates all 9 agents. Executes the complete flow from input to alert dispatch. |
| **`backend/main.py`** | FastAPI application for HTTP API endpoints. Minimal bootstrap file for mounting routes. |
| **`backend/routers/app.py`** | API handlers for multimodal uploads, text inputs, and system endpoints. |
| **`agents/`** | Agent implementations: <br> • `agent1_sensor_data_ingestion.py`: Kafka consumption & CSV parsing <br> • `agent2.py`: Z-score spike detection, semantic text building <br> • `agent3.py`: Perception module (feature extraction) <br> • `agent5.py`: Embedding memory & Qdrant upsert <br> • `agent6_forecasting.py`: LSTM/ML risk forecasting <br> • `agent7_retriever.py`: Retrieval-augmented memory (MMR) <br> • `qa_agent8.py`: RAG chatbot with context <br> • `agent9_recommendation.py`: Action recommendations <br> • `agent4.py`: Legacy embedding service |
| **`backend/services/`** | Service layer: <br> • `kafka_service.py`: Kafka producer/consumer <br> • `qdrant_service.py`: Vector DB operations <br> • `embedding_service.py`: Multimodal embeddings (CLIP) <br> • `rag_service.py`: RAG chatbot logic <br> • `trust_service.py`: Reporter trust scoring <br> • `redis_throttle.py`: Alert deduplication <br> • `smtp_service.py`: Email notifications <br> • `audit_logger.py`: Audit trail logging |
| **`backend/db/`** | Database models and SQLite/PostgreSQL connections |
| **`backend/models/`** | SQLAlchemy data models (User, Alert, Forecasting, etc.) |
| **`whatsapp/`** | WhatsApp integration: <br> • `scraper.py`: Message scraper & JSONL output <br> • `processor.py`: Converts messages to multimodal packets <br> • `utils.py`: Helper functions |
| **`docker-compose.yml`** | Infrastructure stack: Kafka, Zookeeper, Qdrant |
| **`requirements.txt`** | Python dependencies (FastAPI, LangGraph, Torch, etc.) |
| **`water.csv`** | Sample sensor dataset for simulation/testing |

---

## 🧠 AI Models & Components

Water Watch runs a **hybrid AI stack** combining local + cloud models:

| Component | Model | Purpose |
| :--- | :--- | :--- |
| **LLM Reasoning** | `gemini-2.0-flash` | Risk analysis, forecasting explanations, RAG answers |
| **Multimodal Embeddings** | `openai/clip-vit-base-patch32` | Encodes images, text, and sensor data into 512-dim vectors |
| **Audio Transcription** | `openai-whisper-base` | Converts audio reports to text |
| **Sparse Retrieval** | `naver/splade-cocondenser-v3` | Keyword-aware ranking in vector search |
| **Time Series Forecasting** | LSTM/ML models | 6-hour water quality risk predictions |

> Models are auto-downloaded via HuggingFace on first run.

---

## 🚀 Getting Started

### Prerequisites
- **Docker Desktop** (running Kafka, Zookeeper, Qdrant)
- **Python 3.8+**
- **Google Gemini API Key** (for LLM reasoning)
- Optional: **Redis** (for alert deduplication)
- Optional: **SMTP credentials** (for email alerts)

### ✅ Quick Setup

#### 1. Clone & Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Configure Environment
Create a `.env` file in the root directory:

```ini
# === AI & LLM ===
GEMINI_API_KEY=your_google_gemini_api_key

# === Infrastructure ===
KAFKA_BROKERS=localhost:9092
QDRANT_URL=http://localhost:6333
REDIS_HOST=localhost
REDIS_PORT=6379

# === Email Alerts (Optional) ===
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587

# === Risk Thresholds ===
RISK_HIGH_THRESHOLD=0.7
RISK_CRITICAL_THRESHOLD=0.9
```

#### 3. Start Infrastructure
```bash
docker-compose up -d
```
> ⏳ **Wait 30-60 seconds** for Kafka and Qdrant to fully initialize.

Verify status:
```bash
docker-compose logs -f
```

#### 4. Run the System
```bash
python main_script.py
```

---

## 📊 Complete Pipeline Flow

### The 9-Agent Architecture

When you run `python main_script.py`, the system executes this **LangGraph-based pipeline**:

```
INPUT LAYER
├─ AGENT 1: Ingestion (agent1_sensor_data_ingestion.py)
│  └─ Kafka consumption → CSV parsing → Raw sensor data
│
├─ AGENT 2: Spike Detection (agent2.py)
│  └─ Z-score analysis → Detects anomalies → Semantic text encoding
│
├─ AGENT 3: Perception (agent3.py)
│  └─ Feature extraction → Context enrichment
│
├─ AGENT 4: 🎯 Embedding Generation (agent4.py)
│  └─ Input → CLIP embeddings → 512-dim vectors
│
└─ AGENT 5: Vector Storage (agent5.py)
   └─ Qdrant insert → Memory persistence
   
   ↓
   
MEMORY & RETRIEVAL LAYER
├─ AGENT 7: Liquid Memory (agent7_retriever.py)
│  └─ Query → MMR reranking → TopK retrieval
│  └─ Combines semantic + keyword search
│
└─ Returns: [Similar incidents, evidence scores]

   ↓

ANALYSIS & DECISION LAYER
├─ AGENT 6: Forecasting (agent6_forecasting.py)
│  └─ LSTM/ML model → Risk prediction → "Why" explanation
│  └─ Output: risk_level (LOW/MEDIUM/HIGH/CRITICAL)
│
├─ AGENT 8: QA/RAG (qa_agent8.py)
│  └─ Retriever + Gemini → Contextual Q&A
│  └─ Answers based on retrieved evidence
│
└─ AGENT 9: Recommendations (agent9_recommendation.py)
   └─ Risk level → Action recommendations
   └─ Priority escalation
   
   ↓
   
ALERT LAYER
└─ ALERT ENGINE: Automated Dispatch
   └─ If risk >= HIGH:
      • Trust scoring (individual/NGO/government)
      • Location-based government org matching
      • Email alert via SMTP
      • Redis deduplication (prevent duplicates)
      • Audit log entry
```

### Example Flow: WhatsApp Image Input

**User sends a photo to WhatsApp:**
1. `whatsapp/scraper.py` → Detects new image
2. `whatsapp/processor.py` → Converts to JSONL packet
3. Creates `UserInputPacket` with modality=IMAGE
4. **Pipeline Start:**
   - AGENT 4 embeds image via CLIP
   - AGENT 5 stores in Qdrant
   - AGENT 7 retrieves similar historical incidents
   - AGENT 6 forecasts risk based on image features + historical data
   - If risk >= HIGH → ALERT ENGINE triggers
5. **Sends email to regional government authority** with recommendations

---

## 🔗 API Usage

### FastAPI Backend

The backend runs on `http://localhost:8000`:

#### Multimodal Upload
```bash
curl -X POST http://localhost:8000/api/multimodal/upload \
  -H "Content-Type: application/json" \
  -d '{
    "type": "image",
    "content": "<base64_image>",
    "metadata": {
      "source": "whatsapp",
      "location": "North Delhi"
    }
  }'
```

#### Health Check
```bash
curl http://localhost:8000/health
```

---

## 🚨 Alert System

### How Alerts Work

1. **Risk Scoring**: AGENT 6 computes risk_score (0.0 - 1.0)
2. **Trust Multiplier**: Applied based on reporter type
   - Individual: 0.4x multiplier
   - NGO: 0.75x multiplier
   - Government: 0.9x multiplier
3. **Geographic Routing**: `get_nearest_org_by_location()` matches to regional authority
4. **Deduplication**: Redis prevents duplicate alerts within 1 hour
5. **Email Dispatch**: SMTP sends detailed alert with:
   - Risk level & score
   - Location & source ID
   - Sensor data readings
   - AI-generated recommendations
   - Audit trail

### Alert Severity Levels

| Level | Risk Score | Action |
| :--- | :--- | :--- |
| **LOW** | < 0.4 | Monitor, no alert |
| **MEDIUM** | 0.4 - 0.7 | Log & store |
| **HIGH** | 0.7 - 0.9 | 🚨 Alert government |
| **CRITICAL** | ≥ 0.9 | 🔴 Escalate immediately |

---

## 🎯 Key Features

### ✅ Multimodal Support
- **Text**: WhatsApp messages, direct reports
- **Images**: Water quality photos → CLIP embeddings
- **Audio**: Voice reports → Whisper transcription → Text
- **Video**: Frame extraction + analysis

### ✅ Memory System
- **Qdrant Vector DB**: 3 separate vector spaces
  - Semantic embeddings (dense)
  - Fallback embeddings (dense alternative)
  - Structured sensor data (sparse + dense)
- **MMR Reranking**: Balances semantic similarity with diversity
- **Similarity Gating**: Only retrieve events with score ≥ 0.55

### ✅ Forecasting Engine
- Historical incident patterns
- Time-series LSTM predictions
- 6-hour risk horizon
- "Why" explanations via Gemini

---

## 🔧 Configuration & Tuning

### Environment Variables
Key settings in `.env`:

```ini
# Vector Search
SIM_SEARCH_GATE=0.55          # Min similarity to retrieve
SIM_TOPK=30                   # Retrieve top 30 similar incidents
SIM_SEVERITY_MIN=0.4          # Min severity to consider

# Time Windows
DEFAULT_WINDOW=24h            # Historical lookback
DEFAULT_HORIZON=6h            # Forecast period

# Risk Thresholds
RISK_HIGH_THRESHOLD=0.7
RISK_CRITICAL_THRESHOLD=0.9

# Forecasting Mode
DEFAULT_MODE=risk+evidence+why  # Include explanations
```

---

## 📈 Monitoring & Debugging

### View Logs
```bash
# Backend logs
docker-compose logs -f redis
docker-compose logs -f qdrant
docker-compose logs -f kafka
```

### Check Agent Status
The pipeline logs each agent's completion:
```
✅ AGENT1: Ingestion complete
✅ AGENT2: Spike detection complete
✅ AGENT7: Liquid memory retrieval | retrieved=5
✅ AGENT6: Forecast complete | risk_level=HIGH
🚨 ALERT ENGINE: Alert sent to Northern Region Water Authority
```

### Database Inspection
```bash
# Query Qdrant
curl http://localhost:6333/collections/water_memory/points
```

---

## 🚀 Running Different Components

### Run Just the Streaming Pipeline
```bash
python main_script.py
# Select Option 1: End-to-End Pipeline
```

### Run WhatsApp Processor
```bash
python whatsapp/processor.py
```

### Run Backend API Only
```bash
uvicorn backend.main:app --reload --port 8000
```

---

## 📦 Deployment

For production:
1. Use environment-specific `.env` files
2. Enable Redis for alert deduplication
3. Configure proper SMTP credentials
4. Use external Qdrant instance (not container)
5. Enable audit logging to external system
6. Set up proper authentication on API endpoints

---

## 🛠️ Technology Stack

### Core Framework
- **LangGraph**: Agent orchestration & state management
- **FastAPI**: REST API framework
- **Pydantic**: Data validation

### AI/ML
- **Google Gemini 2.0**: LLM reasoning & RAG
- **Torch**: Deep learning framework
- **HuggingFace Transformers**: CLIP, Whisper, SPLADE
- **Sentence-Transformers**: Embeddings

### Infrastructure
- **Kafka**: Event streaming & data pipelines
- **Qdrant**: Vector database for semantic search
- **Redis**: Alert deduplication & caching
- **SQLAlchemy**: ORM for structured data

### Data Processing
- **Pandas**: Data analysis
- **NumPy**: Numerical computing

---

## ❓ Troubleshooting

### Kafka Connection Issues
```bash
# Check Kafka status
docker-compose logs kafka

# Restart Kafka
docker-compose restart kafka

# Verify Kafka is listening
nc -zv localhost 9092
```

### Qdrant Not Responding
```bash
# Check Qdrant status
docker-compose logs qdrant

# Verify collection exists
curl http://localhost:6333/collections/water_memory

# Restart Qdrant
docker-compose restart qdrant
```

### CLIP Model Download Failure
If you see `OSError: Can't find the requested file`:
```bash
# Download models manually
python -c "from transformers import CLIPVisionModel; CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')"
```

### Redis Connection Error
If Redis is optional and you see warnings:
- This is normal in development
- Alerts will still send but without deduplication
- For production, configure Redis in `.env`

### Memory Issues
If system runs out of memory:
- Reduce `SIM_TOPK` from 30 to 10
- Reduce chunk size in AGENT 5
- Use lighter embedding model (e.g., `all-MiniLM-L6-v2`)

---

## 📞 Support & Debugging

### Enable Debug Logging
Add to `.env`:
```ini
LOG_LEVEL=DEBUG
```

### Check System Status
```bash
python -c "
from main_script import *
print('✅ Kafka:', test_kafka())
print('✅ Qdrant:', test_qdrant())
print('✅ Gemini:', test_gemini())
"
```

### View Sample Alerts
```bash
# Last 10 alerts from Qdrant
curl -s http://localhost:6333/collections/water_memory/points?limit=10 | jq '.result[] | select(.payload.alert_sent==true)'
```

---

## 📄 License

[See LICENSE file]

---

**Built with ❤️ for water quality monitoring and environmental protection.**
