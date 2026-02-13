import logging
import time
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from backend.config import settings

logger = logging.getLogger(__name__)


class SMTPService:
    def __init__(self):
        self.host = settings.SMTP_HOST
        self.port = settings.SMTP_PORT
        self.user = settings.SMTP_USER
        self.password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_FROM
        self._last_sent = {}

    def _should_send(self, alert_id: str) -> bool:
        now = time.time()

        if alert_id in self._last_sent:
            if now - self._last_sent[alert_id] < 900:
                logger.info(f"Throttled alert: {alert_id}")
                return False

        self._last_sent[alert_id] = now
        return True

    async def send_alert(
        self,
        alert_id: str,
        subject: str,
        text_body: str,
        html_body: str,
        recipient: str
    ) -> bool:

        if not self._should_send(alert_id):
            return False

        if not self.user or not self.password:
            logger.warning("SMTP not configured")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = recipient

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        await aiosmtplib.send(
            msg,
            hostname=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            start_tls=True,
            timeout=10
        )

        logger.info(f"Alert sent: {alert_id}")
        return True


_smtp_service = SMTPService()

def get_smtp_service() -> SMTPService:
    return _smtp_service
