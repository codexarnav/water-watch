"""
Comprehensive test script for Water Watch system
Tests complete flow: Sensor → Kafka → Embedding → Qdrant → Risk → SMTP + RAG
"""
import httpx
import asyncio
import base64
from datetime import datetime
import time

FASTAPI_URL = "http://localhost:8000"


async def test_health():
    """Test 1: Health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{FASTAPI_URL}/api/health")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Overall Status: {data['status']}")
        print(f"Kafka: {data['services']['kafka']}")
        print(f"Qdrant: {data['services']['qdrant']}")
        print(f"SMTP: {data['services']['smtp']}")
        return response.status_code == 200


async def test_sensor_ingest():
    """Test 2: Sensor data ingestion (Kafka + Embedding + Qdrant)"""
    print("\n" + "="*60)
    print("TEST 2: Sensor Data Ingestion")
    print("="*60)
    data = {
        "site_id": "Bay",
        "ph": 7.3,
        "dissolved_oxygen": 8.5,
        "salinity": 2.5,
        "water_temp": 18.0,
        "air_temp": 20.0
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{FASTAPI_URL}/api/sensor/ingest", json=data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Message: {result['message']}")
        print(f"Kafka Sent: {result.get('kafka_sent', False)}")
        print(f"Embedding Generated: {result.get('embedding_generated', False)}")
        return response.status_code == 200


async def test_bulk_ingest():
    """Test 3: Bulk ingest from water.csv"""
    print("\n" + "="*60)
    print("TEST 3: Bulk Ingest from water.csv")
    print("="*60)
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{FASTAPI_URL}/api/sensor/bulk-ingest")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Ingested: {result['ingested']}")
        print(f"Errors: {result['errors']}")
        return response.status_code == 200


async def test_sensor_status():
    """Test 4: Sensor status (Qdrant collection info)"""
    print("\n" + "="*60)
    print("TEST 4: Sensor Status")
    print("="*60)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{FASTAPI_URL}/api/sensor/status")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Total Readings: {result['total_readings']}")
        print(f"Active Sites: {result['active_sites']}")
        print(f"Kafka Lag: {result['kafka_lag']}")
        return response.status_code == 200


async def test_multimodal_text():
    """Test 5: Multimodal text upload (Embedding + Qdrant)"""
    print("\n" + "="*60)
    print("TEST 5: Multimodal Text Upload")
    print("="*60)
    data = {
        "type": "text",
        "content": "Water appears murky with unusual odor at Site Bay. pH seems low.",
        "metadata": {
            "location": "Site Bay",
            "user_id": "test_user"
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{FASTAPI_URL}/api/multimodal/upload", json=data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Upload ID: {result['upload_id']}")
        print(f"Message: {result['message']}")
        return response.status_code == 200


async def test_risk_forecast():
    """Test 6: Risk forecasting (Qdrant retrieval + Risk calculation + SMTP)"""
    print("\n" + "="*60)
    print("TEST 6: Risk Forecasting")
    print("="*60)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{FASTAPI_URL}/api/risk/forecast?site_id=Bay")
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Risk Score: {result['risk_score']}")
        print(f"Predictions:")
        for key, value in result['predictions'].items():
            print(f"  - {key}: {value}")
        print(f"Recommendations:")
        for rec in result['recommendations']:
            print(f"  - {rec}")
        return response.status_code == 200


async def test_rag_chatbot():
    """Test 7: RAG chatbot (Qdrant retrieval + Gemini)"""
    print("\n" + "="*60)
    print("TEST 7: RAG Chatbot")
    print("="*60)
    data = {
        "query": "What is the current water quality at Site Bay?",
        "context_limit": 5
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{FASTAPI_URL}/api/chat/query", json=data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Query ID: {result['query_id']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Sources Retrieved: {len(result.get('sources', []))}")
        return response.status_code == 200


async def test_alert_send():
    """Test 8: Manual alert (SMTP)"""
    print("\n" + "="*60)
    print("TEST 8: Manual Alert Send")
    print("="*60)
    data = {
        "site_id": "Bay",
        "risk_level": "high"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{FASTAPI_URL}/api/alerts/send", json=data)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Message: {result['message']}")
        return response.status_code == 200


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 WATER WATCH SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("Testing complete flow:")
    print("Sensor → Kafka → Embeddings → Qdrant → Risk → SMTP + RAG")
    print("="*70)
    
    tests = [
        ("Health Check", test_health),
        ("Sensor Ingestion (Kafka + Embedding + Qdrant)", test_sensor_ingest),
        ("Bulk Ingest from CSV", test_bulk_ingest),
        ("Sensor Status (Qdrant Info)", test_sensor_status),
        ("Multimodal Text Upload (Embedding + Qdrant)", test_multimodal_text),
        ("Risk Forecasting (Retrieval + SMTP)", test_risk_forecast),
        ("RAG Chatbot (Qdrant + Gemini)", test_rag_chatbot),
        ("Manual Alert (SMTP)", test_alert_send),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            await asyncio.sleep(1)  # Small delay between tests
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed ({int(passed/total*100)}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System is fully functional!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - Check logs above")
    
    print("="*70)


if __name__ == "__main__":
    print("\n⏳ Starting tests in 3 seconds...")
    print("Make sure the backend is running: uvicorn main:app --reload")
    time.sleep(3)
    asyncio.run(run_all_tests())
