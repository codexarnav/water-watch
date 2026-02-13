import smtplib
from email.message import EmailMessage
from backend.config import settings

def test_smtp():
    msg = EmailMessage()
    msg["Subject"] = "WaterWatch SMTP Test"
    msg["From"] = settings.SMTP_FROM
    msg["To"] = settings.SMTP_TO
    msg.set_content("SMTP is working correctly.")

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.send_message(msg)

    print("✅ SMTP test email sent successfully")

if __name__ == "__main__":
    test_smtp()
