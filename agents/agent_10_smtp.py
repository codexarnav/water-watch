"""
Agent 11: SMTP Awareness Agent
-----------------------------
• Simple cron-based risk monitoring
• Sends email alerts when risk is detected
• Integrates with forecasting.py to check risks
"""

import os
import smtplib
import time
import schedule
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

from agent6_forecasting import forecast

# Load environment variables
load_dotenv()

# ============================
# CONFIGURATION
# ============================
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "")
SMTP_TO = os.getenv("SMTP_TO", "").split(",")  # Comma-separated recipients

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.45

# Monitoring configuration
MONITOR_SOURCES = os.getenv("MONITOR_SOURCES", "").split(",")  # Comma-separated source IDs
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "60"))  # Check every hour by default

# Track sent alerts to avoid spam
_sent_alerts: Dict[str, float] = {}  # {source_id: last_sent_timestamp}
ALERT_COOLDOWN_SEC = 3600  # Don't send same alert within 1 hour


# ============================
# EMAIL FUNCTIONS
# ============================
def draft_email(source_id: str, risk_data: Dict[str, Any]) -> MIMEMultipart:
    """
    Drafts an email based on risk forecast data.
    """
    risk_forecast = risk_data.get("risk_forecast", {})
    risk_score = risk_forecast.get("risk_score", 0.0)
    risk_level = risk_forecast.get("risk_level", "UNKNOWN")
    confidence = risk_forecast.get("confidence", 0.0)
    
    # Determine subject and urgency
    if risk_level == "HIGH":
        subject = f"🚨 HIGH RISK ALERT - {source_id}"
        urgency = "URGENT"
    elif risk_level == "MEDIUM":
        subject = f"⚠️ MEDIUM RISK WARNING - {source_id}"
        urgency = "MODERATE"
    else:
        subject = f"ℹ️ Risk Update - {source_id}"
        urgency = "LOW"
    
    # Build email body
    body = f"""
WATER QUALITY RISK ALERT
========================

Source ID: {source_id}
Risk Level: {risk_level}
Risk Score: {risk_score:.3f}
Confidence: {confidence:.3f}
Generated At: {risk_data.get('meta', {}).get('generated_at', 'N/A')}

"""
    
    # Add risk analysis if available
    if risk_data.get("why"):
        why = risk_data["why"]
        body += f"""
RISK ANALYSIS:
--------------
Local Features:
{', '.join([f"{k}: {v}" for k, v in why.get('local_features', {}).items()])}

Evidence Count: {why.get('retrieval_evidence_count', 0)}
"""
    
    # Add evidence if available
    if risk_data.get("evidence_pack"):
        body += f"""
EVIDENCE:
---------
Found {len(risk_data['evidence_pack'])} similar historical episodes:
"""
        for i, ev in enumerate(risk_data["evidence_pack"][:5], 1):
            body += f"""
{i}. Source: {ev.get('source_id', 'N/A')}
   Timestamp: {ev.get('timestamp', 'N/A')}
   Severity: {ev.get('severity', 0.0):.3f}
   Tags: {', '.join(ev.get('tags', []))}
"""
    
    # Add recommended actions
    if risk_data.get("next_actions"):
        body += f"""
RECOMMENDED ACTIONS:
-------------------
"""
        for action in risk_data["next_actions"]:
            body += f"• {action}\n"
    
    body += f"""
---
This is an automated alert from the Water Watch monitoring system.
Please review the risk assessment and take appropriate action.
"""
    
    # Create email
    msg = MIMEMultipart()
    msg["From"] = SMTP_FROM
    msg["To"] = ", ".join(SMTP_TO)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    
    return msg


def send_email(msg: MIMEMultipart) -> bool:
    """
    Sends email via SMTP.
    Returns True if successful, False otherwise.
    """
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        print("[AGENT11] ❌ SMTP credentials not configured")
        return False
    
    if not SMTP_TO or not SMTP_TO[0]:
        print("[AGENT11] ❌ No recipients configured")
        return False
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        text = msg.as_string()
        server.sendmail(SMTP_FROM, SMTP_TO, text)
        server.quit()
        
        print(f"[AGENT11] ✅ Email sent to {', '.join(SMTP_TO)}")
        return True
        
    except Exception as e:
        print(f"[AGENT11] ❌ Failed to send email: {e}")
        return False


# ============================
# RISK CHECKING
# ============================
def check_risk_for_source(source_id: str) -> Optional[Dict[str, Any]]:
    """
    Checks risk for a specific source and returns risk data if alert needed.
    """
    try:
        # Get risk forecast
        risk_data = forecast(source_id, window="24h", horizon="6h", mode="risk+evidence")
        
        risk_forecast = risk_data.get("risk_forecast", {})
        risk_score = risk_forecast.get("risk_score", 0.0)
        risk_level = risk_forecast.get("risk_level", "LOW")
        
        # Check if we need to alert
        should_alert = False
        if risk_score >= HIGH_RISK_THRESHOLD:
            should_alert = True
        elif risk_score >= MEDIUM_RISK_THRESHOLD:
            should_alert = True
        
        if should_alert:
            # Check cooldown
            last_sent = _sent_alerts.get(source_id, 0)
            now = time.time()
            
            if now - last_sent < ALERT_COOLDOWN_SEC:
                print(f"[AGENT11] ⏸️ Cooldown active for {source_id} (last sent {int((now - last_sent)/60)} min ago)")
                return None
            
            # Update last sent time
            _sent_alerts[source_id] = now
            return risk_data
        
        return None
        
    except Exception as e:
        print(f"[AGENT11] ❌ Error checking risk for {source_id}: {e}")
        return None


def check_all_sources():
    """
    Checks all monitored sources and sends alerts if needed.
    This is the function that will be called by the cron job.
    """
    print(f"\n[AGENT11] 🔍 Checking risks at {datetime.now(timezone.utc).isoformat()}")
    
    if not MONITOR_SOURCES or not MONITOR_SOURCES[0]:
        print("[AGENT11] ⚠️ No sources configured for monitoring")
        return
    
    for source_id in MONITOR_SOURCES:
        source_id = source_id.strip()
        if not source_id:
            continue
        
        print(f"[AGENT11] Checking {source_id}...")
        risk_data = check_risk_for_source(source_id)
        
        if risk_data:
            risk_level = risk_data.get("risk_forecast", {}).get("risk_level", "UNKNOWN")
            print(f"[AGENT11] ⚠️ Risk detected for {source_id}: {risk_level}")
            
            # Draft and send email
            email = draft_email(source_id, risk_data)
            send_email(email)
        else:
            print(f"[AGENT11] ✅ No alert needed for {source_id}")


# ============================
# CRON JOB SETUP
# ============================
def start_monitoring():
    """
    Starts the cron-based monitoring system.
    """
    print(f"\n[AGENT11] 🚀 Starting SMTP Monitoring Agent")
    print(f"[AGENT11] Configuration:")
    print(f"  - Check interval: {CHECK_INTERVAL_MINUTES} minutes")
    print(f"  - Monitoring sources: {', '.join(MONITOR_SOURCES) if MONITOR_SOURCES[0] else 'None configured'}")
    print(f"  - SMTP server: {SMTP_SERVER}:{SMTP_PORT}")
    print(f"  - Recipients: {', '.join(SMTP_TO) if SMTP_TO[0] else 'None configured'}")
    print(f"  - High risk threshold: {HIGH_RISK_THRESHOLD}")
    print(f"  - Medium risk threshold: {MEDIUM_RISK_THRESHOLD}")
    
    # Schedule the job
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(check_all_sources)
    
    # Run immediately on start
    check_all_sources()
    
    # Keep running
    print(f"\n[AGENT11] ✅ Monitoring started. Press Ctrl+C to stop.\n")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for scheduled jobs
    except KeyboardInterrupt:
        print("\n[AGENT11] ⏹️ Monitoring stopped by user")


# ============================
# MANUAL TRIGGER (for testing)
# ============================
def send_test_alert(source_id: str = "Well_Test_fairness"):
    """
    Manually trigger an alert for testing purposes.
    """
    print(f"[AGENT11] 🧪 Sending test alert for {source_id}")
    
    risk_data = forecast(source_id, window="24h", horizon="6h", mode="risk+evidence")
    email = draft_email(source_id, risk_data)
    send_email(email)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode: send one alert
        test_source = sys.argv[2] if len(sys.argv) > 2 else "Well_Test_fairness"
        send_test_alert(test_source)
    else:
        # Production mode: start monitoring
        start_monitoring()