"""
Embedding Service for generating query embeddings
Uses module configuration to determine which embedding model and provider to use
"""

import numpy as np
from typing import List, Optional, Dict, Any
import openai
import asyncio
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service to generate embeddings for search queries
    Uses semantic_search_engine module configuration to determine model and provider
    """
    
    def __init__(self):
        self.openai_client = None
        self.local_model = None
        self.cached_config = None
        self.cached_client = None
        
        # Initialize fallback local model
        try:
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Initialized fallback local SentenceTransformer model")
        except Exception as e:
            logger.warning(f"Failed to initialize fallback local model: {e}")
    
    async def _get_semantic_search_config(self, db: AsyncSession = None) -> Optional[Dict[str, Any]]:
        """Get the active configuration for semantic_search_engine module"""
        try:
            if db is None:
                # Use lightweight DB service for configuration lookup
                from app.core.lightweight_dependencies import get_lightweight_db
                from app.services.lightweight_db_service import LightweightDBService
                
                # We can't use dependency injection here, so we'll create a direct instance
                db_service = LightweightDBService()
                
                # Get module configuration using raw query
                config_query = """
                SELECT mc.model, mc.temperature, mc.max_tokens, ls.provider, ls.encrypted_api_key
                FROM module_configurations mc
                LEFT JOIN llm_settings ls ON mc.llm_setting_id = ls.id
                WHERE mc.module_name = $1 AND mc.active = true
                ORDER BY mc.created_at DESC
                LIMIT 1
                """
                
                config_data = await db_service.execute_fetchrow(config_query, ["semantic_search_engine"])
                await db_service.close()
                
                if config_data:
                    return {
                        "model": config_data["model"],
                        "provider": config_data["provider"],
                        "encrypted_api_key": config_data["encrypted_api_key"]
                    }
            else:
                # Use full service when DB session is available
                from app.services.module_configuration_service import module_configuration_service
                config = await module_configuration_service.get_active_configuration_for_service(
                    db, "semantic_search_engine"
                )
                return config
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get semantic search configuration: {e}")
            return None
    
    async def _get_embedding_client(self, db: AsyncSession = None):
        """Get the appropriate embedding client based on module configuration"""
        try:
            config = await self._get_semantic_search_config(db)
            
            if not config:
                logger.info("No semantic_search_engine configuration found, using fallback local model")
                return self.local_model, "local"
            
            provider = config.get("provider", "").lower()
            model = config.get("model", "")
            
            if provider in ["openai", "openai-compatible"] and config.get("api_key"):
                # Decrypt API key
                from app.utils.encryption import encryption_service
                api_key = encryption_service.decrypt_api_key(config["encrypted_api_key"])
                
                client = openai.AsyncOpenAI(api_key=api_key)
                logger.info(f"Using OpenAI client with model: {model}")
                return client, "openai", model
            
            elif provider == "voyage-ai" and config.get("api_key"):
                # For Voyage AI, we'd need a different client
                # For now, fall back to local model
                logger.warning("Voyage AI not yet supported, using local model")
                return self.local_model, "local"
            
            elif provider == "cohere" and config.get("api_key"):
                # For Cohere, we'd need a different client
                # For now, fall back to local model
                logger.warning("Cohere not yet supported, using local model")
                return self.local_model, "local"
            
            else:
                logger.info(f"Provider {provider} not supported or no API key, using local model")
                return self.local_model, "local"
                
        except Exception as e:
            logger.error(f"Error getting embedding client: {e}")
            return self.local_model, "local"
    
    async def generate_embedding(self, text: str, db: AsyncSession = None) -> Optional[List[float]]:
        """
        Generate embedding for a single text using configured model (simplified - no caching)
        """
        try:
            # Direct generation without caching overhead
            return await self._generate_embedding_uncached(text, {})
            
        except Exception as e:
            logger.error(f"Error in generate_embedding: {e}")
            return None
    
    async def _generate_embedding_uncached(self, text: str, model_info: Dict[str, Any]) -> Optional[List[float]]:
        """
        Generate embedding without caching (used by cache service)
        """
        try:
            client_info = await self._get_embedding_client()
            
            if len(client_info) == 3:
                client, client_type, model = client_info
            else:
                client, client_type = client_info
                model = model_info.get('model', "text-embedding-3-small")
            
            if client_type == "openai" and client:
                try:
                    response = await client.embeddings.create(
                        model=model,
                        input=text
                    )
                    embedding = response.data[0].embedding
                    logger.debug(f"Generated {client_type} embedding using {model} for text: {text[:50]}...")
                    return embedding
                except Exception as e:
                    logger.warning(f"{client_type} embedding failed, falling back to local: {e}")
            
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
    
    async def generate_embeddings(self, texts: List[str], db: AsyncSession = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using configured model
        """
        try:
            client_info = await self._get_embedding_client(db)
            
            if len(client_info) == 3:
                client, client_type, model = client_info
            else:
                client, client_type = client_info
                model = "text-embedding-3-small"  # Default OpenAI model
            
            if client_type == "openai" and client:
                try:
                    # Process in batches to avoid rate limits
                    batch_size = 100
                    all_embeddings = []
                    
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        
                        response = await client.embeddings.create(
                            model=model,
                            input=batch
                        )
                        
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                        
                        # Small delay to respect rate limits
                        if i + batch_size < len(texts):
                            await asyncio.sleep(0.1)
                    
                    logger.info(f"Generated {len(all_embeddings)} {client_type} embeddings using {model}")
                    return all_embeddings
                    
                except Exception as e:
                    logger.warning(f"{client_type} batch embeddings failed, falling back to local: {e}")
            
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
    
    async def get_embedding_dimension(self, db: AsyncSession = None) -> int:
        """
        Get the dimension of embeddings produced by this service
        """
        try:
            client_info = await self._get_embedding_client(db)
            
            if len(client_info) == 3:
                client, client_type, model = client_info
            else:
                client, client_type = client_info
                model = "text-embedding-3-small"
            
            if client_type == "openai":
                # OpenAI model dimensions
                if "text-embedding-3-small" in model:
                    return 1536
                elif "text-embedding-3-large" in model:
                    return 3072
                elif "text-embedding-ada-002" in model:
                    return 1536
                else:
                    return 1536  # Default OpenAI dimension
            else:
                # Local model dimension
                return 384   # all-MiniLM-L6-v2 dimension
                
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {e}")
            return 384  # Default fallback


# Create service instance
embedding_service = EmbeddingService()