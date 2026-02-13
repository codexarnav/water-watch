"""
Alert routing layer for Water Watch

Responsibilities:
- Decide WHICH authorities must be alerted
- Generate deterministic alert_ids
- Prepare alert content
- Call SMTPService (infrastructure layer)

This file contains NO SMTP logic.
"""

from typing import Dict, List
import hashlib

from backend.services.smtp_service import get_smtp_service

smtp = get_smtp_service()

# --------------------------------------------------
# Authority Registry (Policy Layer)
# --------------------------------------------------

AUTHORITY_REGISTRY: Dict[str, Dict] = {
    "KAVERI": {
        "central": ["cwrc@gov.in"],
        "state": ["cmwa@tn.gov.in"],
        "local": ["localbody@tn.gov.in"],
        "statutory": ["cpcb@gov.in"],

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
            "medium": ["state"]
        }
    }
}

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def make_alert_id(site_id: str, risk_level: str, authority_email: str) -> str:
    """
    Deterministic ID used for throttling & deduplication
    """
    raw = f"{site_id}:{risk_level}:{authority_email}"
    return hashlib.sha256(raw.encode()).hexdigest()


def resolve_recipients(site_id: str, risk_level: str) -> List[str]:
    """
    Resolve authority emails based on site + risk policy
    """
    river = AUTHORITY_REGISTRY.get(site_id)
    if not river:
        return []

    roles = river.get("rules", {}).get(risk_level, [])
    recipients: List[str] = []

    for role in roles:
        recipients.extend(river.get(role, []))

    return list(set(recipients))


# --------------------------------------------------
# Core Router Entry
# --------------------------------------------------

async def route_alert(
    site_id: str,
    risk_level: str,
    risk_score: float,
    sensor_data: dict,
    source: str = "sensor"
) -> dict:
    recipients = resolve_recipients(site_id, risk_level)

    if not recipients:
        return {
            "status": "no_recipients",
            "site_id": site_id,
            "risk_level": risk_level,
            "sent": []
        }

    sent_results = []

    for email in recipients:
        alert_id = make_alert_id(site_id, risk_level, email)

        subject = f"🚨 {site_id} Water Alert ({risk_level.upper()})"

        text = f"""
Site: {site_id}
Risk Level: {risk_level}
Risk Score: {risk_score}
Source: {source}
Timestamp: {sensor_data.get("timestamp")}
"""

        html = f"""
<h2>🚨 Water Quality Alert</h2>
<b>Site:</b> {site_id}<br>
<b>Risk:</b> {risk_level}<br>
<b>Score:</b> {risk_score}<br>
<b>Source:</b> {source}
"""

        sent = await smtp.send_alert(
            alert_id=alert_id,
            subject=subject,
            text_body=text,
            html_body=html,
            recipient=email
        )

        sent_results.append({
            "alert_id": alert_id,
            "recipient": email,
            "sent": sent
        })

    return {
        "status": "processed",
        "site_id": site_id,
        "risk_level": risk_level,
        "sent": sent_results
    }
