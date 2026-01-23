"""
SMTP service for sending email alerts
"""
import logging
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Template
from typing import Dict, Any, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

logger = logging.getLogger(__name__)


class SMTPService:
    def __init__(self):
        self.host = settings.SMTP_HOST
        self.port = settings.SMTP_PORT
        self.user = settings.SMTP_USER
        self.password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_FROM
        self.to_email = settings.SMTP_TO
        
        # Email template
        self.alert_template = Template("""
        <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; }
                    .header { background-color: #ff4444; color: white; padding: 20px; }
                    .content { padding: 20px; }
                    .metric { margin: 10px 0; }
                    .recommendations { background-color: #f0f0f0; padding: 15px; margin-top: 20px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚨 Water Quality Alert - {{ site_id }}</h1>
                </div>
                <div class="content">
                    <p><strong>Risk Level:</strong> {{ risk_level | upper }}</p>
                    <p><strong>Risk Score:</strong> {{ risk_score }}</p>
                    <p><strong>Timestamp:</strong> {{ timestamp }}</p>
                    
                    <h3>Current Readings:</h3>
                    {% if ph %}
                    <div class="metric">📊 pH: {{ ph }}</div>
                    {% endif %}
                    {% if dissolved_oxygen %}
                    <div class="metric">💧 Dissolved Oxygen: {{ dissolved_oxygen }} mg/L</div>
                    {% endif %}
                    {% if salinity %}
                    <div class="metric">🧂 Salinity: {{ salinity }} ppt</div>
                    {% endif %}
                    {% if water_temp %}
                    <div class="metric">🌡️ Water Temperature: {{ water_temp }}°C</div>
                    {% endif %}
                    
                    {% if recommendations %}
                    <div class="recommendations">
                        <h3>Recommendations:</h3>
                        <ul>
                        {% for rec in recommendations %}
                            <li>{{ rec }}</li>
                        {% endfor %}
                        </ul>
                    </div>
                    {% endif %}
                </div>
            </body>
        </html>
        """)
    
    async def send_alert(
        self,
        site_id: str,
        risk_level: str,
        risk_score: float,
        sensor_data: Dict[str, Any],
        recommendations: list,
        recipient: Optional[str] = None
    ) -> bool:
        """Send alert email"""
        try:
            # Prepare email content
            html_content = self.alert_template.render(
                site_id=site_id,
                risk_level=risk_level,
                risk_score=risk_score,
                timestamp=sensor_data.get('timestamp', 'N/A'),
                ph=sensor_data.get('ph'),
                dissolved_oxygen=sensor_data.get('dissolved_oxygen'),
                salinity=sensor_data.get('salinity'),
                water_temp=sensor_data.get('water_temp'),
                recommendations=recommendations
            )
            
            # Create message
            message = MIMEMultipart('alternative')
            message['Subject'] = f"🚨 Water Quality Alert - {site_id} - {risk_level.upper()}"
            message['From'] = self.from_email
            message['To'] = recipient or self.to_email
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            message.attach(html_part)
            
            # Send email
            if not self.user or not self.password:
                logger.warning("SMTP credentials not configured, skipping email")
                return False
            
            await aiosmtplib.send(
                message,
                hostname=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                start_tls=True
            )
            
            logger.info(f"Alert email sent to {recipient or self.to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if SMTP is configured"""
        return bool(self.user and self.password)


# Global instance
smtp_service = SMTPService()
