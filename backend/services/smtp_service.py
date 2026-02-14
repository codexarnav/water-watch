import logging
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import settings

logger = logging.getLogger(__name__)


class SMTPService:
    """
    Infrastructure layer only.

    Responsibilities:
    - Connect to SMTP server
    - Send emails
    - No throttling
    - No business logic
    - No alert policy
    """

    def __init__(self):
        self.host = settings.SMTP_HOST
        self.port = settings.SMTP_PORT
        self.user = settings.SMTP_USER
        self.password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_FROM

    async def send_alert(
        self,
        subject: str,
        text_body: str,
        html_body: str,
        recipient: str
    ) -> bool:
        """
        Sends an email alert.

        Returns True if sent successfully.
        Returns False if SMTP is not configured.
        """

        if not self.user or not self.password:
            logger.warning("SMTP not configured. Email not sent.")
            return False

        try:
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
                timeout=10,
            )

            logger.info(f"Alert email sent to {recipient}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {e}")
            return False


# Singleton instance
_smtp_service = SMTPService()


def get_smtp_service() -> SMTPService:
    return _smtp_service