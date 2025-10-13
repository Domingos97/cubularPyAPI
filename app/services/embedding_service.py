"""
Embedding Service for generating query embeddings
Matches the functionality used by the TypeScript API
"""

import numpy as np
from typing import List, Optional
import openai
import asyncio
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service to generate embeddings for search queries
    Supports both OpenAI and local sentence transformers
    """
    
    def __init__(self):
        self.openai_client = None
        self.local_model = None
        
        # Initialize OpenAI client if API key is available
        if settings.openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("Initialized OpenAI embedding client")
        
        # Initialize local model as fallback
        try:
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized local SentenceTransformer model")
        except Exception as e:
            logger.warning(f"Failed to initialize local model: {e}")
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text
        """
        try:
            # Try OpenAI first if available
            if self.openai_client:
                try:
                    response = await self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    embedding = response.data[0].embedding
                    logger.debug(f"Generated OpenAI embedding for text: {text[:50]}...")
                    return embedding
                except Exception as e:
                    logger.warning(f"OpenAI embedding failed, falling back to local: {e}")
            
            # Fallback to local model
            if self.local_model:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None, 
                    lambda: self.local_model.encode([text])[0].tolist()
                )
                logger.debug(f"Generated local embedding for text: {text[:50]}...")
                return embedding
            
            logger.error("No embedding models available")
            return None
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        """
        try:
            # Try OpenAI first if available
            if self.openai_client:
                try:
                    # Process in batches to avoid rate limits
                    batch_size = 100
                    all_embeddings = []
                    
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        
                        response = await self.openai_client.embeddings.create(
                            model="text-embedding-3-small",
                            input=batch
                        )
                        
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                        
                        # Small delay to respect rate limits
                        if i + batch_size < len(texts):
                            await asyncio.sleep(0.1)
                    
                    logger.info(f"Generated {len(all_embeddings)} OpenAI embeddings")
                    return all_embeddings
                    
                except Exception as e:
                    logger.warning(f"OpenAI batch embeddings failed, falling back to local: {e}")
            
            # Fallback to local model
            if self.local_model:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, 
                    lambda: self.local_model.encode(texts).tolist()
                )
                logger.info(f"Generated {len(embeddings)} local embeddings")
                return embeddings
            
            logger.error("No embedding models available")
            return []
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service
        """
        if self.openai_client:
            return 1536  # text-embedding-3-small dimension
        elif self.local_model:
            return 384   # all-MiniLM-L6-v2 dimension
        else:
            return 384   # default


# Create service instance
embedding_service = EmbeddingService()