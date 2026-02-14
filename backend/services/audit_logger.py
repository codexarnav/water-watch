import logging
import json
from datetime import datetime

logger = logging.getLogger("alert_audit")
handler = logging.FileHandler("alert_audit.log")
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class AuditLogger:

    def log(self, data: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }
        logger.info(json.dumps(entry))


_audit = AuditLogger()


def get_audit_logger() -> AuditLogger:
    return _audit