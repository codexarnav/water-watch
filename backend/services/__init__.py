"""Services package - lazy-loaded service instances"""

# Import classes only, don't instantiate
from .qdrant_service import QdrantService
from .embedding_service import EmbeddingService
from .kafka_service import KafkaService
from .smtp_service import SMTPService
from .risk_service import RiskForecastingService
from .rag_service import RAGChatbotService

# Create singleton instances (will be lazy-initialized)
qdrant_service = QdrantService()
embedding_service = EmbeddingService()
kafka_service = KafkaService()
smtp_service = SMTPService()
risk_forecasting_service = RiskForecastingService()
rag_chatbot_service = RAGChatbotService()

__all__ = [
    'qdrant_service',
    'embedding_service',
    'kafka_service',
    'smtp_service',
    'risk_forecasting_service',
    'rag_chatbot_service'
]
