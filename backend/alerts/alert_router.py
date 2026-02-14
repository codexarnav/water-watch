"""
Alert routing layer for Water Watch

Responsibilities:
- Resolve WHICH authorities must be notified
- Generate deterministic alert IDs
- Enforce Redis-based deduplication
- Apply trust weighting
- Dispatch alerts via SMTP service
- Emit audit logs

This file contains NO infrastructure logic.
"""

from typing import Dict, List
import hashlib

from services.smtp_service import get_smtp_service
from services.redis_throttle import get_redis_client
from services.trust_service import get_trust_service
from services.audit_logger import get_audit_logger

# --------------------------------------------------
# Service Singletons
# --------------------------------------------------

smtp = get_smtp_service()
throttle = get_redis_client()
trust_service = get_trust_service()
audit_logger = get_audit_logger()

# --------------------------------------------------
# Authority Registry (Policy Layer)
# NOTE: This will move to DB later
# --------------------------------------------------

AUTHORITY_REGISTRY: Dict[str, Dict] = {
    "KAVERI": {
        "central": ["cwrc@gov.in"],
        "state": ["cmwa@tn.gov.in"],
        "local": ["localbody@tn.gov.in"],

        "rules": {
            "high": ["central", "state"],
            "medium": ["state"],
            "low": []
        }
    },

    "GANGA": {
        "central": ["cpcb@gov.in"],
        "state": ["state@gov.in"],

        "rules": {
            "high": ["central", "state"],
            "medium": ["state"],
            "low": []
        }
    }
}

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def make_alert_id(site_id: str, risk_level: str, recipient: str) -> str:
    """
    Deterministic alert ID used for Redis deduplication
    """
    raw = f"{site_id}:{risk_level}:{recipient}"
    return hashlib.sha256(raw.encode()).hexdigest()


def resolve_recipients(site_id: str, risk_level: str) -> List[str]:
    """
    Resolve authority emails based on site + risk policy
    """
    policy = AUTHORITY_REGISTRY.get(site_id)
    if not policy:
        return []

    roles = policy.get("rules", {}).get(risk_level, [])
    recipients: List[str] = []

    for role in roles:
        recipients.extend(policy.get(role, []))

    return list(set(recipients))

# --------------------------------------------------
# Core Alert Orchestrator
# --------------------------------------------------

async def route_alert(
    site_id: str,
    risk_level: str,
    risk_score: float,
    sensor_data: dict,
    reporter_type: str = "individual",
    source: str = "api"
) -> dict:
    """
    Main alert routing pipeline
    """

    # ---- Trust & Severity ----
    effective_score = trust_service.compute_effective_score(
        risk_score=risk_score,
        reporter_type=reporter_type
    )

    recipients = resolve_recipients(site_id, risk_level)

    if not recipients:
        audit_logger.log({
            "event": "no_recipients",
            "site_id": site_id,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "reporter_type": reporter_type,
        })
        return {
            "status": "no_recipients",
            "site_id": site_id,
            "risk_level": risk_level,
            "sent": []
        }

    sent_results = []

    # ---- Dispatch Loop ----
    for recipient in recipients:
        alert_id = make_alert_id(site_id, risk_level, recipient)

        # ---- Redis Deduplication ----
        if not throttle.allow(alert_id):
            audit_logger.log({
                "event": "deduplicated",
                "alert_id": alert_id,
                "recipient": recipient,
                "site_id": site_id,
            })
            continue

        subject = f"🚨 {site_id} Water Alert ({risk_level.upper()})"

        text_body = f"""
Water Quality Alert

Site: {site_id}
Risk Level: {risk_level}
Risk Score: {risk_score}
Effective Score: {effective_score}
Reporter Type: {reporter_type}
Source: {source}
"""

        html_body = f"""
<h2>🚨 Water Quality Alert</h2>
<ul>
  <li><b>Site:</b> {site_id}</li>
  <li><b>Risk Level:</b> {risk_level}</li>
  <li><b>Risk Score:</b> {risk_score}</li>
  <li><b>Effective Score:</b> {effective_score}</li>
  <li><b>Reporter:</b> {reporter_type}</li>
  <li><b>Source:</b> {source}</li>
</ul>
"""

        sent = await smtp.send_alert(
            subject=subject,
            text_body=text_body,
            html_body=html_body,
            recipient=recipient
        )

        # ---- Audit Log (Legally Required) ----
        audit_logger.log({
            "event": "alert_dispatched",
            "alert_id": alert_id,
            "site_id": site_id,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "effective_score": effective_score,
            "recipient": recipient,
            "sent": sent,
            "reporter_type": reporter_type,
            "source": source,
        })

        sent_results.append({
            "recipient": recipient,
            "sent": sent
        })

    return {
        "status": "processed",
        "site_id": site_id,
        "risk_level": risk_level,
        "effective_score": effective_score,
        "sent": sent_results
    }