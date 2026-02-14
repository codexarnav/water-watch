import asyncio
from backend.services.smtp_service import get_smtp_service

async def main():
    smtp = get_smtp_service()
    ok = await smtp.send_alert(
        site_id="KAVERI-RIVER",
        risk_level="high",
        risk_score=0.88,
        sensor_data={"timestamp": "2026-02-13 17:10"},
        recommendations=[]
    )
    print("EMAIL SENT:", ok)

asyncio.run(main())
