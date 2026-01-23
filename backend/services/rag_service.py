"""
RAG Chatbot service using Gemini and Qdrant
"""
import logging
from typing import List, Dict, Any
import google.generativeai as genai
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service
from config import settings
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class RAGChatbotService:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Chat history storage (in-memory for now)
        self.chat_history: List[Dict[str, Any]] = []
        
        logger.info("RAG Chatbot initialized with Gemini")
    
    def _retrieve_context(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context from Qdrant"""
        try:
            # Embed query
            query_embedding = embedding_service.embed_text(query)
            
            # Search Qdrant
            results = qdrant_service.search_similar(
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.5
            )
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def _build_prompt(
        self,
        query: str,
        context_items: List[Dict[str, Any]]
    ) -> str:
        """Build prompt with retrieved context"""
        # Format context
        context_text = ""
        for i, item in enumerate(context_items, 1):
            payload = item.get('payload', {})
            score = item.get('score', 0)
            
            context_text += f"\n[Context {i}] (Relevance: {score:.2f})\n"
            
            # Extract relevant fields
            if 'site_id' in payload:
                context_text += f"Site: {payload['site_id']}\n"
            if 'timestamp' in payload:
                context_text += f"Time: {payload['timestamp']}\n"
            if 'ph' in payload:
                context_text += f"pH: {payload['ph']}\n"
            if 'dissolved_oxygen' in payload:
                context_text += f"Dissolved Oxygen: {payload['dissolved_oxygen']} mg/L\n"
            if 'salinity' in payload:
                context_text += f"Salinity: {payload['salinity']} ppt\n"
            if 'water_temp' in payload:
                context_text += f"Water Temperature: {payload['water_temp']}°C\n"
            if 'risk_score' in payload:
                context_text += f"Risk Score: {payload['risk_score']}\n"
            if 'content' in payload:
                context_text += f"Content: {payload['content']}\n"
            
            context_text += "\n"
        
        # Build full prompt
        prompt = f"""You are a water quality monitoring assistant for the Water Watch system. 
Your role is to help users understand water quality data, trends, and risks.

Use the following context from the database to answer the user's question. 
If the context doesn't contain relevant information, say so clearly.

CONTEXT:
{context_text}

USER QUESTION:
{query}

INSTRUCTIONS:
- Provide a clear, concise answer based on the context
- If discussing water quality metrics, explain what they mean
- If there are risks, clearly state them
- Be helpful and informative
- If you don't have enough information, say so

ANSWER:"""
        
        return prompt
    
    async def query(
        self,
        user_query: str,
        context_limit: int = 5
    ) -> Dict[str, Any]:
        """
        Process user query with RAG
        """
        try:
            query_id = str(uuid.uuid4())[:8]
            
            # Retrieve context
            context_items = self._retrieve_context(user_query, context_limit)
            
            # Build prompt
            prompt = self._build_prompt(user_query, context_items)
            
            # Generate response with Gemini
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Format sources
            sources = [
                {
                    "timestamp": item['payload'].get('timestamp', 'Unknown'),
                    "content": self._format_source_content(item['payload']),
                    "score": round(item['score'], 3)
                }
                for item in context_items
            ]
            
            # Store in chat history
            chat_entry = {
                "query": user_query,
                "response": answer,
                "timestamp": datetime.utcnow()
            }
            self.chat_history.append(chat_entry)
            
            return {
                "response": answer,
                "sources": sources,
                "query_id": query_id
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "sources": [],
                "query_id": "error"
            }
    
    def _format_source_content(self, payload: Dict[str, Any]) -> str:
        """Format payload into readable source content"""
        parts = []
        
        if 'site_id' in payload:
            parts.append(f"Site {payload['site_id']}")
        
        metrics = []
        if 'ph' in payload:
            metrics.append(f"pH: {payload['ph']}")
        if 'dissolved_oxygen' in payload:
            metrics.append(f"DO: {payload['dissolved_oxygen']} mg/L")
        if 'salinity' in payload:
            metrics.append(f"Salinity: {payload['salinity']} ppt")
        
        if metrics:
            parts.append(", ".join(metrics))
        
        if 'content' in payload:
            parts.append(payload['content'])
        
        return " - ".join(parts) if parts else "No content"
    
    def get_chat_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent chat history"""
        return self.chat_history[-limit:]
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        logger.info("Chat history cleared")


# Global instance
rag_chatbot_service = RAGChatbotService()
