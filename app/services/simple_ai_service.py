"""
Simple AI Service - Direct HTTP Calls
====================================
Replaces complex AI service with direct HTTP requests to OpenAI/Anthropic.
Mimics TypeScript API's lightweight approach with caching and minimal overhead.
"""

import httpx
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from cachetools import TTLCache
import asyncio
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from app.core.config import settings
from app.core.database import get_db_session
from app.models.models import LLMSetting, ModuleConfiguration
from app.utils.logging import get_logger
from app.utils.encryption import encryption_service

logger = get_logger(__name__)


class SimpleAIService:
    """
    Lightweight AI service with direct API calls and response caching
    Similar to TypeScript API's EnhancedChatService but optimized for Python
    """
    
    def __init__(self):
        # Response cache - stores responses for up to 1 hour
        self.response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.http_client = httpx.AsyncClient(timeout=30.0)
        # Cache for LLM configuration to avoid repeated DB queries
        self.config_cache = TTLCache(maxsize=10, ttl=300)  # 5 minutes TTL
        
        # Enhanced system prompt for structured survey analysis
        self.default_system_prompt = """You are an expert survey data analyst and research assistant. Your task is to analyze survey responses and provide structured insights.

When analyzing survey data, you must respond in the following JSON format:

{
  "conversationalResponse": "Your detailed analysis and insights in natural language",
  "dataSnapshot": {
    "stats": [
      {
        "category": "Key Finding",
        "items": [
          {"label": "Insight description", "percentage": 75, "count": 150}
        ],
        "icon": "📊"
      }
    ]
  },
  "confidence": {
    "score": 85,
    "reliability": "high",
    "factors": {
      "sampleSize": 200,
      "dataRelevance": "high",
      "questionSpecificity": "specific",
      "dataCompleteness": "complete"
    }
  }
}

Always provide specific insights based on the actual survey data provided. Include:
- Key themes and patterns
- Statistical insights 
- Demographic breakdowns when available
- Actionable recommendations
- Data quality assessment

If no survey responses are found, explain what insights COULD be drawn IF there were responses, but make it clear no actual data was analyzed."""

    async def get_llm_configuration(self, module_name: str = "ai_chat_integration") -> Optional[Dict[str, Any]]:
        """
        Fetch LLM configuration from database for the specified module
        Returns configuration with decrypted API key
        """
        # Check cache first
        cache_key = f"llm_config_{module_name}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        try:
            async with get_db_session() as session:
                # Query to get module configuration with LLM settings
                query = select(
                    ModuleConfiguration,
                    LLMSetting
                ).join(
                    LLMSetting,
                    ModuleConfiguration.llm_setting_id == LLMSetting.id
                ).where(
                    ModuleConfiguration.module_name == module_name,
                    ModuleConfiguration.active == True,
                    LLMSetting.active == True
                )
                
                result = await session.execute(query)
                row = result.first()
                
                if not row:
                    logger.warning(f"No active LLM configuration found for module: {module_name}")
                    return None
                
                module_config, llm_setting = row
                
                # Decrypt API key from database
                encrypted_api_key = llm_setting.api_key
                try:
                    if encrypted_api_key:
                        api_key = encryption_service.decrypt_api_key(encrypted_api_key)
                        # Log key format without exposing the actual key
                        key_preview = f"{api_key[:8]}..." if api_key and len(api_key) > 8 else "invalid"
                        logger.info(f"Successfully decrypted API key for {module_name}: {key_preview}")
                    else:
                        api_key = None
                        logger.warning(f"No API key found for module: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to decrypt API key for module {module_name}: {str(e)}")
                    api_key = None
                
                config = {
                    "provider": llm_setting.provider,
                    "api_key": api_key,
                    "model": module_config.model,
                    "temperature": float(module_config.temperature) if module_config.temperature else 0.7,
                    "max_tokens": module_config.max_tokens or 1000,
                    "max_completion_tokens": module_config.max_completion_tokens
                }
                
                # Cache the configuration
                self.config_cache[cache_key] = config
                logger.info(f"Retrieved LLM configuration for module: {module_name}, provider: {llm_setting.provider}")
                
                return config
                
        except Exception as e:
            logger.error(f"Error fetching LLM configuration for {module_name}: {str(e)}")
            return None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    def _generate_cache_key(self, messages: List[Dict], model: str, temperature: float) -> str:
        """Generate cache key for response caching"""
        # Create hash of last user message + model + temperature
        last_message = messages[-1]["content"] if messages else ""
        return f"{hash(last_message)}_{model}_{temperature}"
    
    async def generate_openai_response(self, messages: List[Dict], model: str = None, 
                                     temperature: float = None, max_tokens: int = None) -> str:
        """Direct API call using database configuration - supports multiple providers"""
        
        # Get LLM configuration from database
        config = await self.get_llm_configuration("ai_chat_integration")
        if not config:
            raise ValueError("API configuration not found in database")
        
        # Use database config values or provided parameters
        api_key = config["api_key"]
        provider = config["provider"].lower()
        model = model or config["model"]
        temperature = temperature if temperature is not None else config["temperature"]
        max_tokens = max_tokens or config["max_tokens"]
        
        if not api_key:
            raise ValueError(f"API key not found in database configuration for provider: {provider}")
        
        # Check cache first
        cache_key = self._generate_cache_key(messages, model, temperature)
        if cache_key in self.response_cache:
            logger.info(f"Cache hit for {provider} response")
            return self.response_cache[cache_key]
        
        # Set up API endpoint and headers based on provider
        if provider == "openrouter":
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://cubular.com",  # Required for OpenRouter
                "X-Title": "Cubular AI Assistant"
            }
        elif provider == "openai":
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        elif provider == "anthropic":
            return await self.generate_anthropic_response(messages, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = await self.http_client.post(
                api_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
            
            # Cache the response
            self.response_cache[cache_key] = ai_response
            
            logger.info(f"Successfully generated response using {provider} provider")
            return ai_response
            
        except httpx.HTTPError as e:
            logger.error(f"{provider.upper()} API error: {e}")
            raise ValueError(f"{provider.upper()} API error: {e}")
    
    async def generate_anthropic_response(self, messages: List[Dict], model: str = None,
                                        temperature: float = None, max_tokens: int = None) -> str:
        """Direct Anthropic API call using database configuration"""
        
        # Get LLM configuration from database  
        config = await self.get_llm_configuration("ai_chat_integration")
        if not config or config["provider"].lower() != "anthropic":
            raise ValueError("Anthropic API key not configured in database")
        
        api_key = config["api_key"]
        model = model or config["model"]
        temperature = temperature if temperature is not None else config["temperature"]
        max_tokens = max_tokens or config["max_tokens"]
        
        if not api_key:
            raise ValueError("Anthropic API key not found in database configuration")
        
        # Check cache first
        cache_key = self._generate_cache_key(messages, model, temperature)
        if cache_key in self.response_cache:
            logger.info("Cache hit for Anthropic response")
            return self.response_cache[cache_key]
        
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convert messages format for Anthropic
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_message:
            payload["system"] = system_message
        
        try:
            response = await self.http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            ai_response = result["content"][0]["text"]
            
            # Cache the response
            self.response_cache[cache_key] = ai_response
            
            return ai_response
            
        except httpx.HTTPError as e:
            logger.error(f"Anthropic API error: {e}")
            raise ValueError(f"Anthropic API error: {e}")
    
    async def generate_response(self, messages: List[Dict], provider: str = "openai",
                              model: str = None, temperature: float = 0.7, 
                              max_tokens: int = 1000) -> str:
        """
        Generate AI response using specified provider
        Simple routing with default models
        """
        
        if provider.lower() == "openai":
            model = model or "gpt-4o-mini"
            return await self.generate_openai_response(messages, model, temperature, max_tokens)
        elif provider.lower() == "anthropic":
            model = model or "claude-3-haiku-20240307"
            return await self.generate_anthropic_response(messages, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")
    
    def build_messages(self, system_prompt: str, conversation_history: List[Dict], 
                      current_question: str, search_context: str = None) -> List[Dict]:
        """
        Build message array for AI API
        Enhanced context building to match TypeScript API
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add structured search context if available
        if search_context:
            messages.append({
                "role": "system", 
                "content": f"Survey Data Analysis Context:\n\n{search_context}\n\nPlease analyze this data and provide structured insights in the required JSON format."
            })
        else:
            messages.append({
                "role": "system",
                "content": "No survey responses found for this query. Please explain what insights could be drawn if there were responses, following the JSON format."
            })
        
        # Add conversation history (limit to last 6 for performance)
        for msg in conversation_history[-6:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current question
        messages.append({
            "role": "user",
            "content": current_question
        })
        
        return messages
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            "cache_size": len(self.response_cache),
            "cache_info": {
                "hits": getattr(self.response_cache, 'hits', 0),
                "misses": getattr(self.response_cache, 'misses', 0),
                "maxsize": self.response_cache.maxsize,
                "ttl": self.response_cache.ttl
            }
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("AI response cache cleared")


# Global instance
simple_ai = SimpleAIService()


async def get_simple_ai() -> SimpleAIService:
    """Dependency injection for FastAPI"""
    return simple_ai


# Background processing for non-critical operations
async def process_analytics_background(session_id: str, question: str, response: str, 
                                     processing_time: float, provider: str, model: str):
    """Process analytics in background - don't block main response"""
    try:
        # This could save to database, send to analytics service, etc.
        # For now, just log the metrics
        logger.info(
            f"Chat analytics: session={session_id}, "
            f"question_len={len(question)}, response_len={len(response)}, "
            f"time={processing_time:.2f}ms, provider={provider}, model={model}"
        )
    except Exception as e:
        # Don't let analytics failures affect main flow
        logger.warning(f"Background analytics processing failed: {e}")