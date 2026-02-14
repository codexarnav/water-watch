import redis
import logging
from config import settings

logger = logging.getLogger(__name__)


class RedisThrottleService:
    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )

    def allow(self, alert_id: str) -> bool:
        """
        Returns True if alert should be sent.
        Uses SETNX + TTL for deduplication.
        """
        key = f"alert:{alert_id}"

        was_set = self.client.set(
            key,
            "1",
            nx=True,
            ex=settings.ALERT_DEDUP_TTL_SECONDS
        )

        if not was_set:
            logger.info(f"Deduplicated alert {alert_id}")
            return False

        return True


_throttle = RedisThrottleService()


def get_redis_client() -> RedisThrottleService:
    return _throttle