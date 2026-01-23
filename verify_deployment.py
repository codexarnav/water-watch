"""
Water Watch - Deployment Verification Script
Run this to verify that your system is ready for demo.
"""
import asyncio
import httpx
import sys
import time
from datetime import datetime

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

FASTAPI_URL = "http://localhost:8000"

async def check_step(step_name, func):
    print(f"{Colors.OKCYAN}⏳ Checking: {step_name}...{Colors.ENDC}", end="\r")
    try:
        success = await func()
        if success:
            print(f"{Colors.OKGREEN}✅ {step_name}: PASSED{Colors.ENDC}   ")
            return True
        else:
            print(f"{Colors.FAIL}❌ {step_name}: FAILED{Colors.ENDC}   ")
            return False
    except Exception as e:
        print(f"{Colors.FAIL}❌ {step_name}: ERROR ({str(e)}){Colors.ENDC}")
        return False

async def verify_system():
    print(f"{Colors.HEADER}{Colors.BOLD}🌊 Water Watch - System Verification{Colors.ENDC}")
    print("="*60)
    
    # 1. Health Check
    async def check_health():
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{FASTAPI_URL}/api/health", timeout=5)
            return resp.status_code == 200 and resp.json()['status'] in ['healthy', 'degraded']
    
    if not await check_step("Backend API Health", check_health):
        print(f"\n{Colors.WARNING}⚠️  CRITICAL: Backend seems down. Is 'uvicorn' running?{Colors.ENDC}")
        return

    # 2. Qdrant Connection
    async def check_qdrant():
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{FASTAPI_URL}/api/sensor/status", timeout=5)
            # If we get stats, Qdrant is talking to us
            return resp.status_code == 200 and "total_readings" in resp.json()
    
    await check_step("Qdrant Vector DB Connection", check_qdrant)

    # 3. Ingest Simulation (The "Write" Test)
    async def check_ingest():
        async with httpx.AsyncClient() as client:
            payload = {
                "site_id": "TEST_SITE_VERIFY",
                "ph": 7.5,
                "dissolved_oxygen": 8.0,
                "timestamp": datetime.utcnow().isoformat()
            }
            resp = await client.post(f"{FASTAPI_URL}/api/sensor/ingest", json=payload, timeout=5)
            return resp.status_code == 200
            
    await check_step("Sensor Data Ingestion Pipeline", check_ingest)

    # 4. Risk Forecast (The "Read/Compute" Test)
    async def check_forecast():
        async with httpx.AsyncClient() as client:
            # Short wait for ingest to index
            await asyncio.sleep(1)
            resp = await client.get(f"{FASTAPI_URL}/api/risk/forecast?site_id=TEST_SITE_VERIFY", timeout=10)
            data = resp.json()
            # Verify we got the new Rich Schema with history
            has_history = 'predictions' in data and 'ph_history' in data['predictions']
            return resp.status_code == 200 and has_history
            
    await check_step("AI Risk Forecasting & Trend Analysis", check_forecast)

    print("="*60)
    print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
    print("If all checks passed, you are ready to present.")
    print("1. Ensure 'docker-compose up' is running locally.")
    print("2. Run 'streamlit run dashboard/app.py' for the UI.")
    print("\n🚀 Good luck with the submission!")

if __name__ == "__main__":
    try:
        asyncio.run(verify_system())
    except KeyboardInterrupt:
        print("\nAborted.")
