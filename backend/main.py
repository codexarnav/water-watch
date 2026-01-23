"""
Water Watch FastAPI Backend - Complete System
Sensor Data → Kafka → Embeddings → Qdrant → Risk Forecasting → SMTP + RAG
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from schemas import *
from services import (
    qdrant_service,
    embedding_service,
    kafka_service,
    smtp_service,
    risk_forecasting_service,
    rag_chatbot_service
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application startup time
startup_time = time.time()

# Alert history storage (in-memory)
alert_history: List[AlertHistoryItem] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Water Watch Backend starting...")
    logger.info(f"Kafka: {settings.KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    yield
    logger.info("👋 Water Watch Backend shutting down...")
    kafka_service.close()


# Create FastAPI app
app = FastAPI(
    title="Water Watch API",
    description="Water quality monitoring with Kafka, Qdrant, and RAG chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ SENSOR DATA ENDPOINTS ============

@app.post("/api/sensor/ingest", tags=["Sensor Data"])
async def ingest_sensor_data(reading: SensorReading):
    """
    Ingest sensor data: Send to Kafka → Generate embedding → Store in Qdrant
    """
    try:
        # Add timestamp if not provided
        if not reading.timestamp:
            reading.timestamp = datetime.utcnow()
        
        # Convert to dict
        data = reading.model_dump()
        
        # Step 1: Send to Kafka
        success = kafka_service.produce(
            topic=settings.KAFKA_RAW_TOPIC,
            key=reading.site_id,
            value=data
        )
        
        if not success:
            logger.warning("Kafka not available, proceeding without it")
        
        # Step 2: Generate embedding
        embedding = embedding_service.embed_sensor_data(data)
        
        # Step 3: Store in Qdrant with embedding
        if embedding:
            vector_id = f"sensor_{reading.site_id}_{int(time.time() * 1000)}"
            qdrant_service.store_vector(
                vector_id=vector_id,
                vector=embedding,
                payload={
                    **data,
                    "type": "sensor",
                    "timestamp": data['timestamp'].isoformat() if isinstance(data['timestamp'], datetime) else data['timestamp']
                }
            )
        
        return {
            "status": "success",
            "message": f"Sensor data ingested for site {reading.site_id}",
            "timestamp": reading.timestamp,
            "kafka_sent": success,
            "embedding_generated": len(embedding) > 0 if embedding else False
        }
    except Exception as e:
        logger.error(f"Error ingesting sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sensor/status", response_model=SensorStatus, tags=["Sensor Data"])
async def get_sensor_status():
    """Get sensor data status"""
    try:
        # Get Qdrant collection info
        collection_info = qdrant_service.get_collection_info()
        
        # Get Kafka lag
        lag = kafka_service.get_lag(settings.KAFKA_RAW_TOPIC, "sensor-processor")
        
        # Get unique sites from Qdrant (simplified)
        active_sites = ["Bay", "A", "B", "C", "D"]
        
        return SensorStatus(
            total_readings=collection_info.get('points_count', 0),
            last_reading=datetime.utcnow(),
            active_sites=active_sites,
            kafka_lag=lag
        )
    except Exception as e:
        logger.error(f"Error getting sensor status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sensor/bulk-ingest", tags=["Sensor Data"])
async def bulk_ingest_from_csv():
    """Bulk ingest sensor data from water.csv"""
    try:
        # Read CSV
        df = pd.read_csv('water.csv')
        
        ingested = 0
        errors = 0
        
        for _, row in df.head(100).iterrows():  # Limit for demo
            try:
                # Create sensor reading
                reading = {
                    "site_id": str(row.get('Site_Id', 'Unknown')),
                    "salinity": float(row['Salinity (ppt)']) if pd.notna(row.get('Salinity (ppt)')) else None,
                    "dissolved_oxygen": float(row['Dissolved Oxygen (mg/L)']) if pd.notna(row.get('Dissolved Oxygen (mg/L)')) else None,
                    "ph": float(row['pH (standard units)']) if pd.notna(row.get('pH (standard units)')) else None,
                    "water_temp": float(row['Water Temp (?C)']) if pd.notna(row.get('Water Temp (?C)')) else None,
                    "air_temp": float(row['AirTemp (C)']) if pd.notna(row.get('AirTemp (C)')) else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Send to Kafka
                kafka_service.produce(
                    topic=settings.KAFKA_RAW_TOPIC,
                    key=reading['site_id'],
                    value=reading
                )
                
                # Generate embedding and store
                embedding = embedding_service.embed_sensor_data(reading)
                if embedding:
                    vector_id = f"sensor_{reading['site_id']}_{ingested}"
                    qdrant_service.store_vector(
                        vector_id=vector_id,
                        vector=embedding,
                        payload={**reading, "type": "sensor"}
                    )
                
                ingested += 1
            except Exception as e:
                logger.error(f"Error processing row: {e}")
                errors += 1
        
        return {
            "status": "success",
            "ingested": ingested,
            "errors": errors,
            "message": f"Bulk ingested {ingested} sensor readings"
        }
    except Exception as e:
        logger.error(f"Error bulk ingesting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ MULTIMODAL INPUT ENDPOINTS ============

@app.post("/api/multimodal/upload", response_model=UploadResponse, tags=["Multimodal"])
async def upload_multimodal(upload: MultimodalUpload):
    """Upload image/video/audio/text → Generate embedding → Store in Qdrant"""
    try:
        upload_id = str(uuid.uuid4())[:8]
        
        # Generate embedding based on type
        embedding = []
        if upload.type == "text":
            embedding = embedding_service.embed_text(upload.content)
        elif upload.type == "image":
            embedding = embedding_service.embed_image(upload.content)
        elif upload.type == "audio":
            embedding = embedding_service.embed_audio(upload.content)
        elif upload.type == "video":
            embedding = embedding_service.embed_video(upload.content)
        
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
        
        # Store in Qdrant
        vector_id = f"{upload.type}_{upload_id}"
        payload = {
            "type": upload.type,
            "content": upload.content[:500] if upload.type == "text" else "binary_data",
            "metadata": upload.metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        success = qdrant_service.store_vector(
            vector_id=vector_id,
            vector=embedding,
            payload=payload
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store in Qdrant")
        
        return UploadResponse(
            upload_id=upload_id,
            status="success",
            message=f"{upload.type.capitalize()} uploaded successfully",
            embedding_id=vector_id
        )
    except Exception as e:
        logger.error(f"Error uploading multimodal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ RISK & FORECASTING ENDPOINTS ============

@app.get("/api/risk/forecast", response_model=RiskForecast, tags=["Risk"])
async def forecast_risk(site_id: str):
    """
    Get risk forecast: Retrieve from Qdrant → Calculate risk → Send alert if high
    """
    try:
        # Get latest sensor data (simplified - use sample)
        sample_data = {
            "site_id": site_id,
            "ph": 7.3,
            "dissolved_oxygen": 8.5,
            "salinity": 2.5,
            "water_temp": 18.0,
            "timestamp": datetime.utcnow()
        }
        
        # Forecast risk using retrieval-based approach
        forecast = await risk_forecasting_service.forecast_risk(site_id, sample_data)
        
        # If high risk, send alert via SMTP
        if forecast['risk_level'] == 'high':
            await smtp_service.send_alert(
                site_id=site_id,
                risk_level=forecast['risk_level'],
                risk_score=forecast['risk_score'],
                sensor_data=sample_data,
                recommendations=forecast['recommendations']
            )
            
            # Log alert
            alert_history.append(AlertHistoryItem(
                id=str(uuid.uuid4())[:8],
                timestamp=datetime.utcnow(),
                site_id=site_id,
                risk_level=forecast['risk_level'],
                sent_to=settings.SMTP_TO,
                status="sent"
            ))
        
        return RiskForecast(**forecast)
    except Exception as e:
        logger.error(f"Error forecasting risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risk/history", response_model=RiskHistory, tags=["Risk"])
async def get_risk_history(site_id: str, period: str = "7d"):
    """Get historical risk data"""
    try:
        history = [
            RiskHistoryItem(
                timestamp=datetime.utcnow(),
                risk_score=0.65,
                risk_level="medium"
            )
        ]
        
        return RiskHistory(
            site_id=site_id,
            period=period,
            history=history
        )
    except Exception as e:
        logger.error(f"Error getting risk history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ ALERT ENDPOINTS ============

@app.post("/api/alerts/send", tags=["Alerts"])
async def send_alert(alert: AlertRequest):
    """Manually trigger alert email via SMTP"""
    try:
        sensor_data = {
            "site_id": alert.site_id,
            "ph": 7.3,
            "dissolved_oxygen": 8.5,
            "salinity": 2.5,
            "timestamp": datetime.utcnow()
        }
        
        # Send email
        success = await smtp_service.send_alert(
            site_id=alert.site_id,
            risk_level=alert.risk_level,
            risk_score=0.8,
            sensor_data=sensor_data,
            recommendations=["Manual alert triggered"],
            recipient=alert.recipient
        )
        
        if success:
            alert_history.append(AlertHistoryItem(
                id=str(uuid.uuid4())[:8],
                timestamp=datetime.utcnow(),
                site_id=alert.site_id,
                risk_level=alert.risk_level,
                sent_to=alert.recipient or settings.SMTP_TO,
                status="sent"
            ))
            
            return {"status": "success", "message": "Alert sent successfully"}
        else:
            return {"status": "warning", "message": "Alert not sent (SMTP not configured)"}
    except Exception as e:
        logger.error(f"Error sending alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/history", response_model=AlertHistory, tags=["Alerts"])
async def get_alert_history():
    """Get alert history"""
    return AlertHistory(alerts=alert_history[-50:])


# ============ RAG CHATBOT ENDPOINTS ============

@app.post("/api/chat/query", response_model=ChatResponse, tags=["Chatbot"])
async def query_chatbot(query: ChatQuery):
    """
    Query RAG chatbot: Retrieve context from Qdrant → Generate response with Gemini
    """
    try:
        response = await rag_chatbot_service.query(
            user_query=query.query,
            context_limit=query.context_limit
        )
        return ChatResponse(**response)
    except Exception as e:
        logger.error(f"Error querying chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history", response_model=ChatHistory, tags=["Chatbot"])
async def get_chat_history():
    """Get chat history"""
    history = rag_chatbot_service.get_chat_history()
    return ChatHistory(conversations=history)


# ============ HEALTH & MONITORING ============

@app.get("/api/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """System health check"""
    kafka_status = "connected" if kafka_service.health_check() else "disconnected"
    qdrant_status = "connected" if qdrant_service.health_check() else "disconnected"
    smtp_status = "configured" if smtp_service.health_check() else "not_configured"
    
    overall_status = "healthy" if kafka_status == "connected" and qdrant_status == "connected" else "degraded"
    
    return HealthCheck(
        status=overall_status,
        services=ServiceStatus(
            kafka=kafka_status,
            qdrant=qdrant_status,
            smtp=smtp_status
        ),
        uptime=int(time.time() - startup_time)
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "Water Watch API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "architecture": "Sensor → Kafka → Embeddings → Qdrant → Risk → SMTP + RAG"
    }


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host=settings.FASTAPI_HOST,
#         port=settings.FASTAPI_PORT,
#         reload=True
#     )
