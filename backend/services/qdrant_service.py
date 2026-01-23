"""
Qdrant service with binary quantization support
"""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SearchParams,
    QuantizationSearchParams,
    BinaryQuantization,
    BinaryQuantizationConfig,
    OptimizersConfigDiff
)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self):
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION
        self.vector_size = settings.QDRANT_VECTOR_SIZE
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization - only connect when first used"""
        if self._initialized:
            return
        
        try:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )
            self._ensure_collection()
            self._initialized = True
            logger.info("Qdrant service initialized successfully")
        except Exception as e:
            logger.warning(f"Qdrant not available yet: {e}")
            # Don't raise - allow app to start even if Qdrant is down
    
    def _ensure_collection(self):
        """Create collection with binary quantization if it doesn't exist"""
        if not self.client:
            return
        
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=10000
                    ),
                    quantization_config=BinaryQuantization(
                        binary=BinaryQuantizationConfig(
                            always_ram=True  # Keep quantized index in RAM
                        )
                    ),
                    on_disk_payload=True  # Store payloads on disk
                )
                logger.info(f"Collection created with binary quantization")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def store_vector(
        self,
        vector_id: str,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        """Store a vector with payload in Qdrant"""
        self._initialize()  # Ensure initialized
        
        if not self.client:
            logger.error("Qdrant client not available")
            return False
        
        try:
            point = PointStruct(
                id=vector_id,
                vector=vector,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.info(f"Stored vector: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing vector {vector_id}: {e}")
            return False
    
    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using binary quantization with rescoring
        """
        self._initialize()  # Ensure initialized
        
        if not self.client:
            logger.error("Qdrant client not available")
            return []
        
        try:
            search_params = SearchParams(
                quantization=QuantizationSearchParams(
                    rescore=True,  # Enable rescoring for accuracy
                    oversampling=2.0  # Retrieve 2x candidates for rescoring
                )
            )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                search_params=search_params,
                with_payload=True,
                with_vector=False,
                query_filter=filter_dict
            )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics"""
        self._initialize()  # Ensure initialized
        
        if not self.client:
            return {"error": "Qdrant not available"}
        
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[vector_id]
            )
            logger.info(f"Deleted vector: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector {vector_id}: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            self._initialize()  # Try to initialize
            if not self.client:
                return False
            self.client.get_collections()
            return True
        except:
            return False


# Global instance
qdrant_service = QdrantService()
