"""
Kafka service for producing and consuming messages
"""
import logging
import json
from typing import Dict, Any, Optional, Callable
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

logger = logging.getLogger(__name__)


class KafkaService:
    def __init__(self):
        self.bootstrap_servers = settings.KAFKA_BOOTSTRAP_SERVERS
        self.producer: Optional[KafkaProducer] = None
        self.consumers: Dict[str, KafkaConsumer] = {}
        self._initialized = False
    
    def _init_producer(self):
        """Lazy initialize Kafka producer"""
        if self.producer or self._initialized:
            return
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8') if k else None,
                retries=5,
                acks='all'
            )
            self._initialized = True
            logger.info(f"Kafka producer initialized: {self.bootstrap_servers}")
        except Exception as e:
            logger.warning(f"Kafka not available yet: {e}")
            # Don't raise - allow app to start
    
    def produce(self, topic: str, key: str, value: Dict[str, Any]) -> bool:
        """Produce a message to Kafka topic"""
        self._init_producer()  # Ensure initialized
        
        if not self.producer:
            logger.error("Kafka producer not initialized")
            return False
        
        try:
            future = self.producer.send(topic, key=key, value=value)
            future.get(timeout=10)  # Wait for confirmation
            logger.info(f"Message sent to {topic}: {key}")
            return True
        except KafkaError as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            return False
    
    def create_consumer(
        self,
        topic: str,
        group_id: str,
        auto_offset_reset: str = 'earliest'
    ) -> Optional[KafkaConsumer]:
        """Create a Kafka consumer for a topic"""
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                auto_offset_reset=auto_offset_reset,
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                consumer_timeout_ms=1000
            )
            self.consumers[f"{topic}:{group_id}"] = consumer
            logger.info(f"Kafka consumer created for {topic} (group: {group_id})")
            return consumer
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer for {topic}: {e}")
            return None
    
    def consume_batch(
        self,
        topic: str,
        group_id: str,
        max_messages: int = 100
    ) -> list:
        """Consume a batch of messages from a topic"""
        consumer_key = f"{topic}:{group_id}"
        
        if consumer_key not in self.consumers:
            self.create_consumer(topic, group_id)
        
        consumer = self.consumers.get(consumer_key)
        if not consumer:
            return []
        
        messages = []
        try:
            for message in consumer:
                messages.append({
                    'key': message.key,
                    'value': message.value,
                    'partition': message.partition,
                    'offset': message.offset,
                    'timestamp': message.timestamp
                })
                
                if len(messages) >= max_messages:
                    break
        except Exception as e:
            logger.error(f"Error consuming from {topic}: {e}")
        
        return messages
    
    def get_lag(self, topic: str, group_id: str) -> int:
        """Get consumer lag for a topic"""
        try:
            consumer_key = f"{topic}:{group_id}"
            if consumer_key not in self.consumers:
                return 0
            
            consumer = self.consumers[consumer_key]
            
            # Get committed offsets
            committed = consumer.committed(consumer.assignment())
            
            # Get end offsets
            end_offsets = consumer.end_offsets(consumer.assignment())
            
            # Calculate lag
            lag = sum(
                end_offsets.get(tp, 0) - (committed.get(tp, 0) or 0)
                for tp in consumer.assignment()
            )
            
            return lag
        except Exception as e:
            logger.error(f"Error getting lag for {topic}: {e}")
            return 0
    
    def health_check(self) -> bool:
        """Check if Kafka is healthy"""
        try:
            self._init_producer()
            if not self.producer:
                return False
            # Try to get metadata
            return self.producer.bootstrap_connected()
        except:
            return False
    
    def close(self):
        """Close all Kafka connections"""
        if self.producer:
            self.producer.close()
        
        for consumer in self.consumers.values():
            consumer.close()
        
        logger.info("Kafka connections closed")


# Global instance
kafka_service = KafkaService()
