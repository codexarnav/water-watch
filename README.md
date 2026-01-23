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

## 🚀 2. Setup & Running Guide (`main_script.py`)

The `main_script.py` is the central entry point. It handles environment validation, infrastructure checks, and pipeline execution.

### ✅ Prerequisites
*   **Docker Desktop** (must be running)
*   **Python 3.8+**

---

### Step A: Configure Environment
Create a `.env` file in the root directory. Copy the following template and fill in your details:

```ini
# --- AI & LLM Configuration ---
GEMINI_API_KEY=your_google_gemini_api_key

# --- Infrastructure (Docker Default) ---
KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
KAFKA_RAW_TOPIC=sensor.raw
KAFKA_CLEAN_TOPIC=sensor.cleaned
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_COLLECTION=water_watch_vectors

# --- Alerting (SMTP) ---
# Enable/Disable email alerts (true/false)
ENABLE_SMTP_ALERTS=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM=waterwatch@system.com
SMTP_TO=admin@waterwatch.com

# --- System Tuning ---
RISK_HIGH_THRESHOLD=0.7
LOG_LEVEL=INFO
```

### Step B: Start Infrastructure
The system requires Kafka and Qdrant to be running.
```bash
docker-compose up -d
```
> **Note:** Wait about 30 seconds for Kafka to fully initialize before starting the script.

### Step C: Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step D: Run the System
Execute the main orchestrator script:
```bash
python main_script.py
```

---

## 🎮 Interactive Menu

When you run `python main_script.py`, you will see the following menu:

### 1. 📈 Run Forecasting Pipeline (End-to-End)
*   **What it does:**
    1.  **Ingests** data from `water.csv`.
    2.  **Streams** it through Kafka.
    3.  **Detects** spikes and anomalies.
    4.  **Vectorizes** data into Qdrant memory.
    5.  **Forecasts** future risks based on history.
    6.  **Alerts** via Email if risk > 0.7.
*   **Options:** You can limit the number of rows processed (e.g., test with just 50 rows).

### 2. 💬 Start Chatbot Interface
*   **What it does:** Opens a terminal-based chat interface.
*   **Features:**
    *   Ask questions about water quality (e.g., *"What is the risk level at the Bay?"*).
    *   Uses RAG (Retrieval Augmented Generation) to fetch real data from Qdrant.

### 3. Closes the application.

---

## 🔧 Advanced Usage (CLI Flags)

You can also run the script with arguments to bypass the menu:

```bash
# Run the forecasting pipeline directly
python main_script.py --mode forecasting

# Run the chatbot directly
python main_script.py --mode chatbot

# Run with a data limit (e.g., only process 100 rows)
python main_script.py --mode forecasting --data-limit 100

# Dry Run (Check connections without processing data)
python main_script.py --dry-run
```
