#!/usr/bin/env python3
"""
Water Watch - End-to-End Integration Script
============================================

Orchestrates the complete Water Watch system with real integrations:
- Forecasting Flow: CSV → Kafka → Preprocessing → Spike Detection → Embeddings → Qdrant → Risk Forecasting → SMTP
- Chatbot Flow: Multimodal Inputs → Embeddings → Memory → Qdrant → RAG → Responses

Author: Water Watch Team
"""

import os
import sys
import time
import json
import signal
import argparse
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Kafka
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# SMTP
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("WaterWatch")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Centralized configuration from environment variables"""
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
    KAFKA_RAW_TOPIC = os.getenv("KAFKA_RAW_TOPIC", "sensor.raw")
    KAFKA_CLEAN_TOPIC = os.getenv("KAFKA_CLEAN_TOPIC", "sensor.cleaned")
    KAFKA_EVENT_TOPIC = os.getenv("KAFKA_EVENT_TOPIC", "events.queue")
    
    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "water_watch_vectors")
    QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
    
    # SMTP
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    SMTP_FROM = os.getenv("SMTP_FROM", "waterwatch@gmail.com")
    SMTP_TO = os.getenv("SMTP_TO", "admin@waterwatch.com")
    
    # Risk Thresholds
    RISK_HIGH_THRESHOLD = float(os.getenv("RISK_HIGH_THRESHOLD", "0.7"))
    RISK_MEDIUM_THRESHOLD = float(os.getenv("RISK_MEDIUM_THRESHOLD", "0.4"))
    
    # Data
    CSV_PATH = "water.csv"
    
    # Script Config
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DATA_BATCH_SIZE = int(os.getenv("DATA_BATCH_SIZE", "10"))
    ENABLE_SMTP_ALERTS = os.getenv("ENABLE_SMTP_ALERTS", "true").lower() == "true"
    CHATBOT_ENABLED = os.getenv("CHATBOT_ENABLED", "true").lower() == "true"

config = Config()

# =============================================================================
# GLOBAL STATE
# =============================================================================

class SystemState:
    """Global system state for orchestration"""
    def __init__(self):
        self.running = True
        self.threads: List[threading.Thread] = []
        self.kafka_producer: Optional[KafkaProducer] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.stats = {
            "rows_produced": 0,
            "rows_preprocessed": 0,
            "spikes_detected": 0,
            "embeddings_created": 0,
            "qdrant_stored": 0,
            "forecasts_made": 0,
            "emails_sent": 0,
            "chatbot_queries": 0,
            "errors": 0
        }
        self.lock = threading.Lock()

state = SystemState()

# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

def signal_handler(signum, frame):
    """Graceful shutdown on SIGINT/SIGTERM"""
    logger.info("🛑 Shutdown signal received. Cleaning up...")
    state.running = False
    
    # Wait for threads to finish
    for thread in state.threads:
        if thread.is_alive():
            thread.join(timeout=5)
    
    # Close connections
    if state.kafka_producer:
        state.kafka_producer.close()
    
    logger.info("✅ Shutdown complete")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# INFRASTRUCTURE VERIFICATION
# =============================================================================

def verify_docker_services() -> bool:
    """Verify Docker services (Kafka, Qdrant) are running"""
    logger.info("🔍 Verifying Docker services...")
    
    # Check Kafka
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            request_timeout_ms=5000
        )
        admin_client.close()
        logger.info("✅ Kafka is running on %s", config.KAFKA_BOOTSTRAP_SERVERS)
    except Exception as e:
        logger.error("❌ Kafka not available: %s", e)
        logger.error("   Please run: docker-compose up -d")
        return False
    
    # Check Qdrant
    try:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        client.get_collections()
        logger.info("✅ Qdrant is running on %s:%s", config.QDRANT_HOST, config.QDRANT_PORT)
    except Exception as e:
        logger.error("❌ Qdrant not available: %s", e)
        logger.error("   Please run: docker-compose up -d")
        return False
    
    return True

def verify_environment() -> bool:
    """Verify environment variables are set"""
    logger.info("🔍 Verifying environment configuration...")
    
    required = {
        "GEMINI_API_KEY": config.GEMINI_API_KEY,
        "SMTP_USER": config.SMTP_USER,
        "SMTP_PASSWORD": config.SMTP_PASSWORD,
    }
    
    missing = [k for k, v in required.items() if not v]
    
    if missing:
        logger.error("❌ Missing required environment variables: %s", ", ".join(missing))
        logger.error("   Please check your .env file")
        return False
    
    logger.info("✅ Environment configuration valid")
    return True

def verify_data_file() -> bool:
    """Verify water.csv exists"""
    logger.info("🔍 Verifying data file...")
    
    if not os.path.exists(config.CSV_PATH):
        logger.error("❌ Data file not found: %s", config.CSV_PATH)
        return False
    
    logger.info("✅ Data file found: %s", config.CSV_PATH)
    return True

# =============================================================================
# KAFKA SETUP
# =============================================================================

def create_kafka_topics():
    """Create Kafka topics if they don't exist"""
    logger.info("📋 Creating Kafka topics...")
    
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            request_timeout_ms=10000
        )
        
        topics = [
            NewTopic(name=config.KAFKA_RAW_TOPIC, num_partitions=3, replication_factor=1),
            NewTopic(name=config.KAFKA_CLEAN_TOPIC, num_partitions=3, replication_factor=1),
            NewTopic(name=config.KAFKA_EVENT_TOPIC, num_partitions=3, replication_factor=1),
        ]
        
        try:
            admin_client.create_topics(new_topics=topics, validate_only=False)
            logger.info("✅ Created Kafka topics")
        except TopicAlreadyExistsError:
            logger.info("✅ Kafka topics already exist")
        
        admin_client.close()
        return True
        
    except Exception as e:
        logger.error("❌ Failed to create Kafka topics: %s", e)
        return False

def init_kafka_producer() -> Optional[KafkaProducer]:
    """Initialize Kafka producer"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8") if k else None,
            retries=5,
            max_in_flight_requests_per_connection=5
        )
        logger.info("✅ Kafka producer initialized")
        return producer
    except Exception as e:
        logger.error("❌ Failed to initialize Kafka producer: %s", e)
        return None

# =============================================================================
# QDRANT SETUP
# =============================================================================

def init_qdrant_collection() -> bool:
    """Initialize Qdrant collection with proper schema"""
    logger.info("📊 Initializing Qdrant collection...")
    
    try:
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        state.qdrant_client = client
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if "water_memory" in collection_names:
            logger.info("✅ Qdrant collection 'water_memory' already exists")
            return True
        
        # Create collection with multiple vectors (as per Agent 5 schema)
        from qdrant_client.models import VectorParams, Distance
        
        client.create_collection(
            collection_name="water_memory",
            vectors_config={
                "semantic_bind": VectorParams(size=384, distance=Distance.COSINE),
                "sensor_dense": VectorParams(size=384, distance=Distance.COSINE),
                "lexical_sparse": VectorParams(size=384, distance=Distance.COSINE),
            }
        )
        
        logger.info("✅ Created Qdrant collection 'water_memory'")
        return True
        
    except Exception as e:
        logger.error("❌ Failed to initialize Qdrant collection: %s", e)
        return False

# =============================================================================
# SMTP VERIFICATION
# =============================================================================

def verify_smtp_connection() -> bool:
    """Verify SMTP connection (without sending email)"""
    logger.info("📧 Verifying SMTP connection...")
    
    try:
        server = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=10)
        server.starttls()
        server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        server.quit()
        logger.info("✅ SMTP connection verified")
        return True
    except Exception as e:
        logger.error("❌ SMTP connection failed: %s", e)
        logger.error("   Check SMTP credentials in .env")
        return False

# =============================================================================
# FORECASTING PIPELINE
# =============================================================================

def run_forecasting_pipeline(data_limit: Optional[int] = None):
    """
    Run the complete forecasting pipeline:
    CSV → Kafka → Preprocessing → Spike Detection → Embeddings → Qdrant → Forecasting → SMTP
    """
    logger.info("🚀 Starting Forecasting Pipeline...")
    
    # Import agents
    try:
        from agents.agent1_sensor_data_ingestion import run_producer, run_agent1_loop
        from agents.agent2 import run_agent2
        from agents.agent5 import run_system as run_agent5_system
        from agents.agent6_forecasting import forecast
        from agents.agent_10_smtp import check_all_sources
    except ImportError as e:
        logger.error("❌ Failed to import agents: %s", e)
        logger.error("   Make sure all agent files are present in the agents/ directory")
        return
    
    # Stage 1: Data Producer
    logger.info("📤 Stage 1: Producing data from water.csv to Kafka...")
    producer_thread = threading.Thread(target=run_producer, daemon=True, name="Producer")
    producer_thread.start()
    state.threads.append(producer_thread)
    
    # Wait for producer to start
    time.sleep(3)
    
    # Stage 2: Preprocessing (Agent 1)
    logger.info("🔧 Stage 2: Starting preprocessing agent...")
    preprocess_thread = threading.Thread(target=run_agent1_loop, daemon=True, name="Preprocessor")
    preprocess_thread.start()
    state.threads.append(preprocess_thread)
    
    # Stage 3: Spike Detection (Agent 2)
    logger.info("📊 Stage 3: Starting spike detection agent...")
    spike_thread = threading.Thread(target=run_agent2, daemon=True, name="SpikeDetector")
    spike_thread.start()
    state.threads.append(spike_thread)
    
    # Stage 4 & 5: Embeddings + Memory (Agent 4 & 5 combined)
    logger.info("🧠 Stage 4-5: Starting embedding and memory agents...")
    memory_thread = threading.Thread(target=run_agent5_system, daemon=True, name="MemorySystem")
    memory_thread.start()
    state.threads.append(memory_thread)
    
    # Monitor progress
    logger.info("⏳ Processing data... (Press Ctrl+C to stop)")
    logger.info("=" * 80)
    
    forecast_counter = 0
    last_forecast_time = time.time()
    FORECAST_INTERVAL = 30  # Run forecasting every 30 seconds
    
    try:
        while state.running:
            time.sleep(5)
            
            # Periodic status update
            if forecast_counter % 6 == 0:  # Every 30 seconds
                logger.info("📈 Pipeline Status:")
                logger.info("   Threads alive: %d/%d", 
                          sum(1 for t in state.threads if t.is_alive()), 
                          len(state.threads))
                logger.info("   Stats: %s", json.dumps(state.stats, indent=2))
            
            # Check if we should run forecasting and alerts
            current_time = time.time()
            if current_time - last_forecast_time >= FORECAST_INTERVAL:
                last_forecast_time = current_time
                
                # Stage 6: Forecasting (periodically)
                logger.info("🔮 Stage 6: Running forecasting for 'Bay' site...")
                try:
                    result = forecast("Bay", window="24h", horizon="6h", mode="risk+evidence")
                    state.stats["forecasts_made"] += 1
                    
                    risk_score = result.get("risk_score", 0)
                    risk_level_str = result.get("risk_level", "unknown")
                    
                    logger.info("   Risk Score: %.2f (%s)", risk_score, risk_level_str)
                    
                    # Stage 7: SMTP Alerts (if high risk)
                    if config.ENABLE_SMTP_ALERTS and risk_score > config.RISK_HIGH_THRESHOLD:
                        logger.warning("⚠️  HIGH RISK DETECTED!")
                        logger.info("📧 Stage 7: Sending SMTP alert...")
                        try:
                            check_all_sources()
                            state.stats["emails_sent"] += 1
                            logger.info("✅ Alert email sent successfully")
                        except Exception as email_error:
                            logger.error("❌ Failed to send email: %s", email_error)
                            state.stats["errors"] += 1
                        
                except Exception as e:
                    logger.error("❌ Forecasting error: %s", e)
                    state.stats["errors"] += 1
            
            forecast_counter += 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping forecasting pipeline...")
    finally:
        logger.info("=" * 80)
        logger.info("📊 Final Statistics:")
        logger.info(json.dumps(state.stats, indent=2))


# =============================================================================
# CHATBOT PIPELINE
# =============================================================================

def test_chatbot(query: str, modality: str = "text"):
    """
    Test the chatbot pipeline with a query
    
    Args:
        query: The user query
        modality: Input type (text/image/video/audio)
    """
    logger.info("💬 Testing Chatbot Pipeline...")
    logger.info("   Query: %s", query)
    logger.info("   Modality: %s", modality)
    
    try:
        from agents.qa_agent8 import build_qa_agent
        
        # Build and run QA agent
        qa_agent = build_qa_agent()
        
        result = qa_agent.invoke({
            "query": query,
            "query_vector": None,
            "policy": None,
            "retrieved": [],
            "evidence_by_modality": {},
            "answer": None
        })
        
        answer_data = result.get("answer", {})
        
        logger.info("✅ Chatbot Response:")
        logger.info("   Answer: %s", answer_data.get("text", "No answer"))
        logger.info("   Citations: %d", len(answer_data.get("citations", [])))
        logger.info("   Media: %d items", len(answer_data.get("media", [])))
        
        state.stats["chatbot_queries"] += 1
        return result
        
    except Exception as e:
        logger.error("❌ Chatbot error: %s", e)
        import traceback
        traceback.print_exc()
        state.stats["errors"] += 1
        return None

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_full_system(args):
    """Run the complete Water Watch system"""
    logger.info("=" * 80)
    logger.info("🌊 Water Watch - End-to-End Integration System")
    logger.info("=" * 80)
    
    # Infrastructure verification
    if not args.skip_docker_check:
        if not verify_docker_services():
            logger.error("❌ Docker services not available. Exiting.")
            return False
    
    if not verify_environment():
        logger.error("❌ Environment configuration invalid. Exiting.")
        return False
    
    if not verify_data_file():
        logger.error("❌ Data file not found. Exiting.")
        return False
    
    # Setup
    if not create_kafka_topics():
        logger.error("❌ Failed to create Kafka topics. Exiting.")
        return False
    
    if not init_qdrant_collection():
        logger.error("❌ Failed to initialize Qdrant. Exiting.")
        return False
    
    if config.ENABLE_SMTP_ALERTS and not args.dry_run:
        if not verify_smtp_connection():
            logger.warning("⚠️  SMTP verification failed. Email alerts will be disabled.")
            config.ENABLE_SMTP_ALERTS = False
    
    # Dry run check
    if args.dry_run:
        logger.info("✅ Dry run complete. All systems verified.")
        return True
    
    # Initialize Kafka producer
    state.kafka_producer = init_kafka_producer()
    if not state.kafka_producer:
        logger.error("❌ Failed to initialize Kafka producer. Exiting.")
        return False
    
    logger.info("=" * 80)
    logger.info("✅ All systems initialized. Starting pipelines...")
    logger.info("=" * 80)
    
    # Run based on mode
    if args.mode in ["forecasting", "full"]:
        run_forecasting_pipeline(data_limit=args.data_limit)
    
    if args.mode == "chatbot":
        # Interactive chatbot mode
        logger.info("💬 Chatbot mode. Enter queries (type 'exit' to quit):")
        while state.running:
            try:
                query = input("\n🤔 You: ")
                if query.lower() in ["exit", "quit"]:
                    break
                test_chatbot(query)
            except EOFError:
                break
    
    return True

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def interactive_menu():
    """Run interactive UI menu"""
    while True:
        print("\n" + "="*50)
        print("🌊  Water Watch - System Control")
        print("="*50)
        print("1. 📈 Run Forecasting Pipeline (End-to-End)")
        print("2. 💬 Start Chatbot Interface")
        print("3. ❌ Exit")
        print("-" * 50)
        
        choice = input("👉 Select an option (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Initializing Forecasting Pipeline...")
            limit_input = input("   Limit rows? (Enter number or press Enter for all): ").strip()
            data_limit = int(limit_input) if limit_input.isdigit() else None
            
            # Run initialization first
            if not run_full_system(argparse.Namespace(
                mode="forecasting", 
                data_limit=data_limit, 
                skip_docker_check=False, 
                dry_run=False
            )):
                print("❌ Failed to start pipeline.")
            
        elif choice == "2":
            print("\n💬 Starting Chatbot...")
            # Initialize system first (Kafka/Qdrant checks)
            if verify_docker_services() and verify_environment() and init_qdrant_collection():
                 print("\n" + "="*40)
                 print("🤖 Water Watch Chatbot")
                 print("Type 'exit' to return to menu")
                 print("="*40)
                 
                 while True:
                    try:
                        query = input("\n👤 User: ").strip()
                        if query.lower() in ["exit", "quit", "menu"]:
                            break
                        if not query:
                            continue
                            
                        # Default to text modality for now
                        test_chatbot(query, modality="text")
                        
                    except KeyboardInterrupt:
                        break
            else:
                 print("❌ System check failed. Please verify infrastructure.")

        elif choice == "3":
            print("👋 Exiting Water Watch...")
            sys.exit(0)
        else:
            print("❌ Invalid option. Please try again.")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Water Watch End-to-End Integration System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["forecasting", "chatbot", "full", "interactive"],
        default="interactive",
        help="Execution mode (default: interactive)"
    )
    
    parser.add_argument(
        "--data-limit",
        type=int,
        default=None,
        help="Limit number of rows from CSV for testing"
    )
    
    parser.add_argument(
        "--skip-docker-check",
        action="store_true",
        help="Skip Docker service verification"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without processing data"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    if args.mode == "interactive":
        interactive_menu()
    else:
        # Run system standard way
        success = run_full_system(args)
        if success:
            logger.info("✅ System completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ System failed")
            sys.exit(1)

if __name__ == "__main__":
    main()
