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
from app.core.config import settings
from app.services.lightweight_db_service import lightweight_db
from app.utils.logging import get_logger
from app.utils.encryption import encryption_service

logger = get_logger(__name__)


class SimpleAIService:
    """
    Lightweight AI service with direct API calls and optimized HTTP client
    Enhanced with connection pooling, HTTP/2, and performance optimizations
    """
    
    def __init__(self):
        # Optimized HTTP client with connection pooling and shorter timeouts
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=5.0,     # 5s connection timeout (reduced from default 30s)
                read=15.0,       # 15s read timeout (reduced from default 30s)
                write=10.0,      # 10s write timeout
                pool=20.0        # 20s pool timeout
            ),
            limits=httpx.Limits(
                max_keepalive_connections=10,  # Keep 10 connections alive
                max_connections=20,            # Max 20 concurrent connections
                keepalive_expiry=30.0         # Keep connections alive for 30s
            ),
            http2=False,         # Disable HTTP/2 to avoid h2 dependency 
            follow_redirects=True
        )
        
        # Cache for LLM configuration to avoid repeated DB queries
        self.config_cache = TTLCache(maxsize=10, ttl=300)  # 5 minutes TTL
        
        # Provider health tracking
        self.provider_health = {
            "openai": {"last_success": None, "failure_count": 0},
            "anthropic": {"last_success": None, "failure_count": 0},
            "openrouter": {"last_success": None, "failure_count": 0}
        }
        
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

    def clear_config_cache(self, module_name: str = None):
        """Clear configuration cache for specific module or all modules"""
        if module_name:
            cache_key = f"llm_config_{module_name}"
            if cache_key in self.config_cache:
                del self.config_cache[cache_key]
                logger.info(f"Cleared config cache for module: {module_name}")
        else:
            self.config_cache.clear()
            logger.info("Cleared all config cache")

    def _update_provider_health(self, provider: str, success: bool):
        """Update provider health tracking"""
        if provider in self.provider_health:
            if success:
                self.provider_health[provider]["last_success"] = time.time()
                self.provider_health[provider]["failure_count"] = 0
            else:
                self.provider_health[provider]["failure_count"] += 1
                
    def _is_provider_healthy(self, provider: str) -> bool:
        """Check if provider is healthy based on recent failures"""
        if provider not in self.provider_health:
            return True
            
        health = self.provider_health[provider]
        failure_count = health["failure_count"]
        last_success = health["last_success"]
        
        # Consider unhealthy if more than 3 consecutive failures
        if failure_count >= 3:
            # But allow retry if it's been more than 5 minutes since last failure
            if last_success and (time.time() - last_success) > 300:
                return True
            return False
            
        return True

    def _normalize_model_for_provider(self, provider: str, model: str) -> str:
        """Normalize stored model strings for the specific provider API.

        - If model is stored as 'provider/name' and target provider expects plain name (e.g. openai, anthropic), strip prefix.
        - If target is openrouter and model is plain (no '/'), prefix with 'openai/' as a sensible default for common cases.
        Assumptions: when in doubt, prefer the simplest transformation that will work with the provider's API.
        """
        if not model:
            return model

        # Split into parts
        parts = model.split('/')

        lp = provider.lower() if provider else ''

        # For OpenAI and Anthropic, APIs expect bare model names (no provider/ prefix)
        if lp in ('openai', 'anthropic', 'google', 'cohere'):
            return parts[-1]

        # For OpenRouter, many models are in the form 'provider/name' (e.g. 'openai/gpt-4o-mini').
        # If stored model already contains a '/', keep it. Otherwise, default to prefixing with 'openai/'.
        if lp == 'openrouter':
            if len(parts) > 1:
                return model
            # default prefix for common OpenAI-origin models
            return f"openai/{model}"

        # Default: return model unchanged
        return model
        
    async def _make_http_request_with_retry(self, url: str, headers: dict, payload: dict, 
                                          provider: str, max_retries: int = 2) -> dict:
        """Make HTTP request with exponential backoff retry"""
        for attempt in range(max_retries + 1):
            try:
                # Add jitter to prevent thundering herd
                if attempt > 0:
                    wait_time = (2 ** attempt) + (time.time() % 1)  # Exponential backoff with jitter
                    await asyncio.sleep(wait_time)
                    logger.info(f"Retrying {provider} request, attempt {attempt + 1}")
                
                response = await self.http_client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                self._update_provider_health(provider, True)
                return response.json()
                
            except httpx.HTTPError as e:
                self._update_provider_health(provider, False)
                
                if attempt == max_retries:
                    logger.error(f"{provider.upper()} API error after {max_retries + 1} attempts: {e}")
                    raise ValueError(f"{provider.upper()} API error: {e}")
                else:
                    logger.warning(f"{provider.upper()} API error on attempt {attempt + 1}: {e}")
                    
        # This should never be reached, but added for safety
        raise ValueError(f"Maximum retries exceeded for {provider}")

    async def get_llm_configuration(self, module_name: str = "ai_chat_integration") -> Optional[Dict[str, Any]]:
        """
        Fetch LLM configuration from database for the specified module using lightweight DB service
        Returns configuration with decrypted API key
        """
        # Check cache first
        cache_key = f"llm_config_{module_name}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        try:
            # Query to get module configuration with LLM settings using lightweight DB service
            query = """
            SELECT mc.model, mc.temperature, mc.max_tokens, mc.max_completion_tokens, mc.ai_personality_id,
                   ls.provider, ls.api_key
            FROM module_configurations mc
            LEFT JOIN llm_settings ls ON mc.llm_setting_id = ls.id
            WHERE mc.module_name = $1 AND mc.active = true AND ls.active = true
            """
            
            result = await lightweight_db.execute_fetchrow(query, [module_name])
            
            if not result:
                logger.warning(f"No active LLM configuration found for module: {module_name}")
                return None
            
            # Decrypt API key from database
            encrypted_api_key = result.get("api_key")
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
                "provider": result.get("provider"),
                "api_key": api_key,
                "model": result.get("model"),
                "temperature": float(result.get("temperature")) if result.get("temperature") else 0.7,
                "max_tokens": result.get("max_tokens") or 1000,
                "max_completion_tokens": result.get("max_completion_tokens")
            }
            
            # Cache the configuration
            self.config_cache[cache_key] = config
            logger.info(f"Retrieved LLM configuration for module: {module_name}, provider: {result.get('provider')}")
            
            return config
                
        except Exception as e:
            logger.error(f"Error fetching LLM configuration for {module_name}: {str(e)}")
            return None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    async def generate_openai_response(self, messages: List[Dict], model: str = None, 
                                     temperature: float = None, max_tokens: int = None, 
                                     provider: str = None, api_key: str = None) -> str:
        """Direct API call using provided configuration or database fallback with health checking"""
        
        # Check provider health first
        effective_provider = provider.lower() if provider else "openai"
        if not self._is_provider_healthy(effective_provider):
            logger.warning(f"Provider {effective_provider} is unhealthy, will attempt anyway")
        
        # If explicit configuration provided, use it directly (faster path)
        if provider and api_key and model:
            config = {
                "provider": provider.lower(),
                "api_key": api_key,
                "model": model,
                "temperature": temperature if temperature is not None else 0.7,
                "max_tokens": max_tokens or 1000
            }
            logger.info(f"Using provided configuration: provider={provider}, model={model}")
        else:
            # Fallback to database configuration
            config = await self.get_llm_configuration("ai_chat_integration")
            if not config:
                raise ValueError("API configuration not found in database")
            
            # Use database config values or provided parameters
            config["model"] = model or config["model"]
            config["temperature"] = temperature if temperature is not None else config["temperature"]
            config["max_tokens"] = max_tokens or config["max_tokens"]
            logger.info(f"Using database configuration for ai_chat_integration")

            api_key = config["api_key"]
            provider = config["provider"].lower()
            # Normalize model string for target provider API
            model = self._normalize_model_for_provider(provider, config["model"])
            temperature = config["temperature"]
            max_tokens = config["max_tokens"]

            if not api_key:
                raise ValueError(f"API key not found for provider: {provider}")

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
                # Use optimized HTTP client with retry mechanism
                result = await self._make_http_request_with_retry(api_url, headers, payload, provider)
                ai_response = result["choices"][0]["message"]["content"]

                logger.info(f"Successfully generated response using {provider} provider")
                return ai_response

            except Exception as e:
                logger.error(f"{provider.upper()} API error: {e}")
                raise
    
    async def generate_anthropic_response(self, messages: List[Dict], model: str = None,
                                        temperature: float = None, max_tokens: int = None) -> str:
        """Direct Anthropic API call using database configuration with health checking"""
        
        # Check provider health first
        if not self._is_provider_healthy("anthropic"):
            logger.warning("Provider anthropic is unhealthy, will attempt anyway")
        
        # Get LLM configuration from database  
        config = await self.get_llm_configuration("ai_chat_integration")
        if not config or config["provider"].lower() != "anthropic":
            raise ValueError("Anthropic API key not configured in database")
        
        api_key = config["api_key"]
        model = model or self._normalize_model_for_provider("anthropic", config.get("model") or '')
        temperature = temperature if temperature is not None else config["temperature"]
        max_tokens = max_tokens or config["max_tokens"]
        
        if not api_key:
            raise ValueError("Anthropic API key not found in database configuration")

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
            # Use optimized HTTP client with retry mechanism
            result = await self._make_http_request_with_retry(
                "https://api.anthropic.com/v1/messages", 
                headers, 
                payload, 
                "anthropic"
            )
            ai_response = result["content"][0]["text"]
            
            logger.info("Successfully generated response using Anthropic provider")
            return ai_response
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def generate_openrouter_response(self, messages: List[Dict], model: str = None,
                                         temperature: float = None, max_tokens: int = None) -> str:
        """Generate response using OpenRouter API with health checking"""
        
        # Check provider health first
        if not self._is_provider_healthy("openrouter"):
            logger.warning("Provider openrouter is unhealthy, will attempt anyway")
        
        # Get API key for OpenRouter from database
        from app.utils.encryption import encryption_service
        
        # Get OpenRouter configuration
        config_query = """
        SELECT ls.api_key, ls.provider
        FROM llm_settings ls
        WHERE ls.provider = 'openrouter' AND ls.active = true
        LIMIT 1
        """
        
        config_data = await lightweight_db.execute_fetchrow(config_query)
        if not config_data:
            raise ValueError("OpenRouter configuration not found in database")
        
        # Decrypt API key
        encrypted_api_key = config_data["api_key"]
        api_key = encryption_service.decrypt_api_key(encrypted_api_key)

        if not api_key:
            raise ValueError("OpenRouter API key not found or could not be decrypted")

        # Set defaults (normalize for OpenRouter expectation)
        model = model or config.get("model") or "openai/gpt-4o-mini"
        model = self._normalize_model_for_provider("openrouter", model)
        temperature = float(temperature) if temperature is not None else 0.7
        max_tokens = max_tokens or 1000
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cubular.com",  # Required for OpenRouter
            "X-Title": "Cubular AI Assistant"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            # Use optimized HTTP client with retry mechanism
            result = await self._make_http_request_with_retry(
                "https://openrouter.ai/api/v1/chat/completions",
                headers,
                payload,
                "openrouter"
            )
            ai_response = result["choices"][0]["message"]["content"]
            
            logger.info("Successfully generated response using OpenRouter provider")
            return ai_response
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    async def generate_streaming_response(self, messages: List[Dict], provider: str = "openai",
                                        model: str = None, temperature: float = 0.7, 
                                        max_tokens: int = 1000, api_key: str = None):
        """
        Generate streaming AI response for improved perceived performance
        Yields chunks of the response as they arrive
        """
        
        # Check provider health first
        if not self._is_provider_healthy(provider.lower()):
            logger.warning(f"Provider {provider} is unhealthy, will attempt anyway")
        
        # Get configuration (using database if not provided)
        if not (api_key and model):
            config = await self.get_llm_configuration("ai_chat_integration")
            if not config:
                raise ValueError("API configuration not found in database")
            
            api_key = api_key or config["api_key"]
            provider = provider or config["provider"]
            model = model or self._normalize_model_for_provider(provider or config.get("provider", "openai"), config.get("model") or '')
            temperature = temperature if temperature is not None else config.get("temperature", 0.7)
            max_tokens = max_tokens or config.get("max_tokens", 1000)
        
        if not api_key:
            raise ValueError(f"API key not found for provider: {provider}")
        
        # Set up API endpoint and headers based on provider
        if provider.lower() == "openai":
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        elif provider.lower() == "openrouter":
            api_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://cubular.com",
                "X-Title": "Cubular AI Assistant"
            }
        elif provider.lower() == "anthropic":
            # Anthropic streaming will be handled differently
            async for chunk in self._generate_anthropic_streaming_response(
                messages, model, temperature, max_tokens, api_key
            ):
                yield chunk
            return
        else:
            raise ValueError(f"Streaming not supported for provider: {provider}")
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True  # Enable streaming
        }
        
        try:
            async with self.http_client.stream(
                "POST", api_url, headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                
                self._update_provider_health(provider.lower(), True)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        
                        if data == "[DONE]":
                            break
                            
                        try:
                            chunk = json.loads(data)
                            if chunk.get("choices") and chunk["choices"][0].get("delta"):
                                content = chunk["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue  # Skip malformed chunks
                            
        except Exception as e:
            self._update_provider_health(provider.lower(), False)
            logger.error(f"{provider.upper()} streaming error: {e}")
            raise
            
    async def _generate_anthropic_streaming_response(self, messages: List[Dict], 
                                                   model: str, temperature: float, 
                                                   max_tokens: int, api_key: str):
        """Handle Anthropic streaming response with different format"""
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
            "max_tokens": max_tokens,
            "stream": True
        }
        
        if system_message:
            payload["system"] = system_message
        
        try:
            async with self.http_client.stream(
                "POST", "https://api.anthropic.com/v1/messages", 
                headers=headers, json=payload
            ) as response:
                response.raise_for_status()
                
                self._update_provider_health("anthropic", True)
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        
                        try:
                            chunk = json.loads(data)
                            if chunk.get("type") == "content_block_delta":
                                content = chunk.get("delta", {}).get("text", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            self._update_provider_health("anthropic", False)
            logger.error(f"Anthropic streaming error: {e}")
            raise
            
    async def generate_response(self, messages: List[Dict], provider: str = "openai",
                              model: str = None, temperature: float = 0.7, 
                              max_tokens: int = 1000, api_key: str = None, 
                              stream: bool = False) -> str:
        """
        Generate AI response using specified provider
        Enhanced to accept explicit configuration to bypass database lookups
        Supports both streaming and non-streaming modes
        """
        
        if stream:
            # For streaming, collect all chunks and return complete response
            full_response = ""
            async for chunk in self.generate_streaming_response(
                messages, provider, model, temperature, max_tokens, api_key
            ):
                full_response += chunk
            return full_response
        
        # Non-streaming mode (existing behavior)
        if provider.lower() == "openai":
            model = model or "gpt-4o-mini"
            return await self.generate_openai_response(
                messages, model, temperature, max_tokens, provider, api_key
            )
        elif provider.lower() == "anthropic":
            model = model or "claude-3-haiku-20240307"
            return await self.generate_anthropic_response(messages, model, temperature, max_tokens)
        elif provider.lower() == "openrouter":
            model = model or "openai/gpt-4o-mini"
            return await self.generate_openrouter_response(messages, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")
            
    async def generate_batch_responses(self, batch_requests: List[Dict[str, Any]], 
                                     max_concurrent: int = 3) -> List[str]:
        """
        Generate multiple AI responses concurrently with controlled concurrency
        Useful for processing multiple questions or follow-ups simultaneously
        
        Args:
            batch_requests: List of request dicts with keys: messages, provider, model, etc.
            max_concurrent: Maximum number of concurrent requests (default 3)
            
        Returns:
            List of response strings in the same order as input requests
        """
        
        if not batch_requests:
            return []
            
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _process_single_request(request: Dict[str, Any], index: int) -> tuple[int, str]:
            """Process a single request with semaphore limiting"""
            async with semaphore:
                try:
                    response = await self.generate_response(
                        messages=request.get("messages", []),
                        provider=request.get("provider", "openai"),
                        model=request.get("model"),
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens", 1000),
                        api_key=request.get("api_key"),
                        stream=request.get("stream", False)
                    )
                    return (index, response)
                except Exception as e:
                    logger.error(f"Batch request {index} failed: {e}")
                    return (index, f"Error: {str(e)}")
        
        # Execute all requests concurrently
        tasks = [
            _process_single_request(request, i) 
            for i, request in enumerate(batch_requests)
        ]
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort results by original index and extract responses
        sorted_results = sorted(results, key=lambda x: x[0] if isinstance(x, tuple) else 0)
        responses = [
            result[1] if isinstance(result, tuple) else f"Error: {str(result)}" 
            for result in sorted_results
        ]
        
        logger.info(f"Completed batch processing of {len(batch_requests)} requests")
        return responses
    
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
    
    def clear_cache(self):
        """Clear configuration cache and reset provider health"""
        self.config_cache.clear()
        # Reset provider health tracking
        for provider in self.provider_health:
            self.provider_health[provider] = {"last_success": None, "failure_count": 0}
        logger.info("AI configuration cache and provider health cleared")
        
    def get_provider_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current provider health status for monitoring"""
        return {
            provider: {
                "healthy": self._is_provider_healthy(provider),
                "failure_count": health["failure_count"],
                "last_success": health["last_success"]
            }
            for provider, health in self.provider_health.items()
        }


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