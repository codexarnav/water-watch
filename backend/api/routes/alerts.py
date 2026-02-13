from fastapi import APIRouter, HTTPException
from backend.schemas import AlertRequest
from backend.alerts.alert_router import route_alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.post("/trigger")
async def trigger_alert(payload: AlertRequest):
    result = await route_alert(
        site_id=payload.site_id,
        risk_level=payload.risk_level,
        risk_score=0.0,  # will come from model later
        sensor_data={"timestamp": None},
        source="api"
    )

    if result["status"] == "no_recipients":
        raise HTTPException(status_code=404, detail="No authorities configured")

    return result
