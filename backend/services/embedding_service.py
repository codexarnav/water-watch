"""
Embedding service for text, image, audio, and video
"""
import logging
import base64
import io
from typing import List, Optional
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import whisper
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None
        self.whisper_model = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization - load models only when first used"""
        if self._initialized:
            return
        
        try:
            logger.info("Loading embedding models...")
            # Text embeddings (384 dimensions)
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Image embeddings
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Audio transcription
            self.whisper_model = whisper.load_model("base")
            
            self._initialized = True
            logger.info("Embedding models loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load embedding models: {e}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate text embedding (384d)"""
        self._initialize()  # Ensure initialized
        
        if not self.text_model:
            logger.error("Text model not available")
            return []
        
        try:
            embedding = self.text_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return []
    
    def embed_image(self, image_base64: str) -> List[float]:
        """Generate image embedding via CLIP, then project to 384d"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Get CLIP embedding (512d)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Project to 384d (simple linear projection)
            clip_embedding = image_features.squeeze().numpy()
            projected = self._project_to_384d(clip_embedding)
            
            return projected.tolist()
        except Exception as e:
            logger.error(f"Error embedding image: {e}")
            return []
    
    def embed_audio(self, audio_base64: str) -> List[float]:
        """Transcribe audio with Whisper, then embed text"""
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(audio_base64)
            
            # Save temporarily for Whisper
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                audio_path = f.name
            
            # Transcribe
            result = self.whisper_model.transcribe(audio_path)
            text = result["text"]
            
            # Clean up
            import os
            os.unlink(audio_path)
            
            # Embed transcribed text
            return self.embed_text(text)
        except Exception as e:
            logger.error(f"Error embedding audio: {e}")
            return []
    
    def embed_video(self, video_base64: str, sample_frames: int = 5) -> List[float]:
        """
        Extract frames from video, embed each, then average
        For simplicity, we'll treat it as a single frame (first frame)
        """
        try:
            # For now, extract first frame and treat as image
            # In production, you'd use cv2 to extract multiple frames
            import tempfile
            import cv2
            
            # Decode base64 video
            video_data = base64.b64decode(video_base64)
            
            # Save temporarily
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_data)
                video_path = f.name
            
            # Extract first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Could not read video frame")
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Encode to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Clean up
            import os
            os.unlink(video_path)
            
            # Embed as image
            return self.embed_image(img_base64)
        except Exception as e:
            logger.error(f"Error embedding video: {e}")
            return []
    
    def _project_to_384d(self, vector: np.ndarray) -> np.ndarray:
        """Simple projection from any dimension to 384d"""
        if len(vector) == 384:
            return vector
        elif len(vector) > 384:
            # Truncate
            return vector[:384]
        else:
            # Pad with zeros
            return np.pad(vector, (0, 384 - len(vector)), mode='constant')
    
    def embed_sensor_data(self, sensor_reading: dict) -> List[float]:
        """Convert sensor reading to text and embed"""
        try:
            # Create semantic text from sensor data
            text_parts = [f"Site: {sensor_reading.get('site_id', 'Unknown')}"]
            
            if sensor_reading.get('ph'):
                text_parts.append(f"pH: {sensor_reading['ph']}")
            if sensor_reading.get('dissolved_oxygen'):
                text_parts.append(f"Dissolved Oxygen: {sensor_reading['dissolved_oxygen']} mg/L")
            if sensor_reading.get('salinity'):
                text_parts.append(f"Salinity: {sensor_reading['salinity']} ppt")
            if sensor_reading.get('water_temp'):
                text_parts.append(f"Water Temperature: {sensor_reading['water_temp']}°C")
            
            semantic_text = ". ".join(text_parts)
            return self.embed_text(semantic_text)
        except Exception as e:
            logger.error(f"Error embedding sensor data: {e}")
            return []


# Global instance
embedding_service = EmbeddingService()
