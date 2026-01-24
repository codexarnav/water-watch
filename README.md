# Water Watch System

An intelligent, multi-agent AI system for real-time water quality monitoring, risk forecasting, and multimodal analysis.

---

## 📂 1. Project Structure & Architecture

This project uses a mesh of intelligent agents to continually monitor data, detect anomalies, store memories, and forecast risks.

| Directory / File | Description |
| :--- | :--- |
| **`agents/`** | Contains the logic for the 11-agent mesh system: <br> • **Ingestion**: `agent1*.py` (Reads/Cleans `water.csv`) <br> • **Analysis**: `agent2*.py` (Spike Refinement/Z-Score) <br> • **Memory**: `agent4/5.py` (Embeddings & Qdrant Storage) <br> • **Forecasting**: `agent6*.py` (Risk Prediction) <br> • **Chatbot**: `agent9.py` / `qa_agent8.py` (RAG & Multimodal RAG) |
| **`backend/`** | Backend service logic for API integrations (if applicable). |
| **`main_script.py`** | **The Master Orchestrator.** This script validates infrastructure, manages all agents, runs the forecasting pipeline, and hosts the interactive CLI. |
| **`input_agent.py`** | Handles specific input processing tasks for the agent mesh. |
| **`docker-compose.yml`** | Defines the critical infrastructure services: <br> • **Kafka (9092)**: Real-time event streaming. <br> • **Zookeeper (2181)**: Kafka coordination. <br> • **Qdrant (6333)**: Vector database for AI memory. |
| **`.env`** | Configuration file for API keys, database hosts, and SMTP settings. |
| **`water.csv`** | The raw sensor dataset used for simulation. |

---

## 🧠 2. AI Engine & Models

Water Watch doesn't just "call an API"—it runs a sophisticated local + cloud hybrid AI mesh. When you run the system, the following models are verified and loaded:

| Role | Model / Transformer | Purpose |
| :--- | :--- | :--- |
| **Reasoning & Chat** | **`gemini-2.0-flash`** | The "Cortex" of the system. Used for complex risk forecasting, "Why" analysis, and RAG chatbot answers. |
| **Vision & Text** | **`openai/clip-vit-base-patch32`** | **Multimodal Embeddings.** Converts sensor logs, images, and text descriptions into shared 512-dim vector space for Qdrant. |
| **Audio Analysis** | **`whisper-base`** | **Transcription.** Converts audio reports (e.g., "water rushing noise") into text for processing. |
| **Keyword Search** | **`naver/splade-cocondenser...`** | **Sparse Retrieval.** Generates lexical weights to ensure specific keywords (chemicals, locations) aren't lost in vector search. |

> **Note:** The local models (CLIP, Whisper, SPLADE) will be downloaded automatically by HuggingFace `transformers` on the first run.

---

## 🚀 3. Setup & Running Guide

### 🧠 The Orchestrator: `main_script.py`
The entire system is managed by `main_script.py`. Instead of manually running 11 separate agent files, this script acts as a unified commander that:

1.  **Validates Infrastructure**: Automatically checks if Docker, Kafka, and Qdrant are active before starting.
2.  **Initializes Resources**: Creates required Kafka topics (`sensor.raw`, `sensor.cleaned`) and Qdrant collections (`water_memory` with 3 vector spaces) if they don't exist.
3.  **Manages Threading**: Launches the ingestion, preprocessing, spike detection, and forecasting agents in parallel background threads.
4.  **Provides a UI**: Offers a clean interactive CLI for users to control the system.

---

### ✅ Prerequisites
*   **Docker Desktop** (Must be running for Kafka/Qdrant)
*   **Python 3.8+**

### Step A: Configure Environment
Create a `.env` file in the root directory.

```ini
# --- AI & LLM Configuration ---
GEMINI_API_KEY=your_google_gemini_api_key

# --- Infrastructure ---
KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333

# --- Email Alerts ---
ENABLE_SMTP_ALERTS=true
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_TO=admin@waterwatch.com

# --- Tuning ---
RISK_HIGH_THRESHOLD=0.7
```

### Step B: Start Infrastructure
```bash
docker-compose up -d
```
> **⏳ Important:** Wait 30-60 seconds after this command for Kafka to fully initialize.

### Step C: Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Step D: Execution Modes

You can run the system in **Interactive Mode** (recommended) or **Headless Mode** (for automation).

#### 1. 🎮 Interactive Mode
The easiest way to use Water Watch.
```bash
python main_script.py
```
**What happens next?**
The script performs a self-check and presents a menu:
*   **Option 1: End-to-End Pipeline**
    *   Starts streaming `water.csv` simulated data.
    *   Agents clean data -> Detect Spikes -> Generate Embeddings -> Store in Qdrant.
    *   Every 30s, the Forecaster predicts risk and sends an email if it exceeds `0.7`.
*   **Option 2: Chatbot**
    *   Enters a loop where you can ask questions like *"How many high-risk events happened today?"*.
    *   The system uses specific vector retrieval to answer based on the sensor history.

#### 2. 🤖 Headless / CLI Mode
Use these flags to run specific components directly without user input:

**Run the Forecasting Pipeline:**
```bash
# Run the full pipeline (Ingestion -> AI -> Alerts)
python main_script.py --mode forecasting

# Run with a limit (e.g., process only the first 50 rows of data)
python main_script.py --mode forecasting --data-limit 50
```

**Run the Chatbot Only:**
```bash
# Skip the menu and go straight to chat
python main_script.py --mode chatbot
```

**Verify System Status:**
```bash
# Check if Kafka/Qdrant are reachable without running agents
python main_script.py --dry-run
```
