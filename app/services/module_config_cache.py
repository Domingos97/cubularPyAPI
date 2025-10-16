"""
Module Configuration Cache Service - ENHANCED
=============================================
High-performance configuration caching with intelligent preloading, bulk operations,
and optimized database access patterns for Phase 2C optimization.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Set
from collections import OrderedDict
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModuleConfigCache:
    """
    High-performance module configuration cache with intelligent features:
    - Bulk loading and batch operations
    - Automatic preloading of frequently used configs
    - LRU eviction with configurable TTL
    - Connection-level caching to reduce DB queries
    - Performance monitoring and optimization
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 1800):  # 30 minutes TTL
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        
        # Bulk operation tracking
        self._pending_loads: Dict[str, asyncio.Future] = {}
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.bulk_loads = 0
        self.preloads = 0
        
        # Frequently accessed modules for preloading
        self.frequent_modules = {
            "ai_chat_integration", "survey_analysis", "user_management", 
            "notification_service", "file_management"
        }
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired."""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used item when cache is full."""
        if len(self._cache) >= self.max_size:
            key, _ = self._cache.popitem(last=False)  # Remove oldest
            self._timestamps.pop(key, None)
            self._access_counts.pop(key, None)
            self.evictions += 1
    
    def _update_access(self, key: str):
        """Update access timestamp and move to end (most recent)."""
        self._cache.move_to_end(key)
        self._timestamps[key] = time.time()
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
    
    async def get_config(self, db, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Get module configuration from cache or database with enhanced performance features
        """
        async with self._lock:
            # Check for pending load
            if module_name in self._pending_loads:
                try:
                    return await self._pending_loads[module_name]
                except Exception:
                    pass
            
            # Check cache with TTL validation
            if module_name in self._cache and not self._is_expired(module_name):
                self._update_access(module_name)
                self.hits += 1
                return self._cache[module_name].copy()
            
            # Cache miss
            self.misses += 1
        
        # Create future for this load to prevent duplicate queries
        future = asyncio.Future()
        async with self._lock:
            if module_name in self._pending_loads:
                return await self._pending_loads[module_name]
            self._pending_loads[module_name] = future
        
        try:
            config_data = await self._fetch_config_from_db(db, module_name)
            
            # Update cache
            async with self._lock:
                if config_data:
                    self._evict_lru()
                    self._cache[module_name] = config_data
                    self._update_access(module_name)
                
                # Clean up pending loads
                self._pending_loads.pop(module_name, None)
            
            future.set_result(config_data)
            return config_data
            
        except Exception as e:
            async with self._lock:
                self._pending_loads.pop(module_name, None)
            future.set_exception(e)
            raise
    
    async def get_multiple_configs(self, db, module_names: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Efficiently get multiple configurations with bulk loading."""
        results = {}
        missing_modules = []
        
        # Check cache first
        async with self._lock:
            for module_name in module_names:
                if module_name in self._cache and not self._is_expired(module_name):
                    self._update_access(module_name)
                    results[module_name] = self._cache[module_name].copy()
                    self.hits += 1
                else:
                    missing_modules.append(module_name)
                    self.misses += 1
        
        # Bulk load missing configs
        if missing_modules:
            await self._bulk_load_configs(db, missing_modules)
            
            # Get from cache after bulk load
            async with self._lock:
                for module_name in missing_modules:
                    if module_name in self._cache:
                        results[module_name] = self._cache[module_name].copy()
                    else:
                        results[module_name] = None
        
        return results
    
    async def _bulk_load_configs(self, db, module_names: List[str]):
        """Load multiple configurations in optimized batch operation."""
        self.bulk_loads += 1
        
        from app.services.query_optimizer import query_optimizer
        
        # Use bulk query for better performance
        configs = await query_optimizer.get_multiple_module_configs(db, module_names)
        
        # Update cache with results
        async with self._lock:
            for module_name, config_data in configs.items():
                if config_data:
                    self._evict_lru()
                    self._cache[module_name] = config_data
                    self._update_access(module_name)
    
    async def preload_frequent_configs(self, db):
        """Preload frequently used module configurations."""
        self.preloads += 1
        missing_frequent = []
        
        async with self._lock:
            for module_name in self.frequent_modules:
                if module_name not in self._cache or self._is_expired(module_name):
                    missing_frequent.append(module_name)
        
        if missing_frequent:
            await self._bulk_load_configs(db, missing_frequent)
    
    async def _fetch_config_from_db(self, db, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Fetch module configuration from database with optimized query
        """
        from app.services.query_optimizer import query_optimizer
        
        result = await query_optimizer.get_module_config_with_personality(db, module_name)
        
        if not result:
            return None
        
        # Decrypt API key
        api_key = None
        if result.get("api_key"):
            try:
                from app.utils.encryption import encryption_service
                api_key = encryption_service.decrypt_api_key(result["api_key"])
            except Exception as e:
                logger.error(f"Failed to decrypt API key for {module_name}: {str(e)}")
                api_key = None
        
        return {
            "provider": result["provider"],
            "model": result["model"],
            "temperature": float(result.get("temperature", 0.7)) if result.get("temperature") is not None else 0.7,
            "max_tokens": result.get("max_tokens", 2000),
            "api_key": api_key,
            "ai_personality_id": result.get("ai_personality_id"),
            "personality_name": result.get("personality_name"),
            "detailed_analysis_prompt": result.get("detailed_analysis_prompt")
        }
    
    async def invalidate(self, module_name: str = None):
        """
        Invalidate cache for specific module or all modules with enhanced cleanup
        """
        async with self._lock:
            if module_name:
                self._cache.pop(module_name, None)
                self._timestamps.pop(module_name, None)
                self._access_counts.pop(module_name, None)
            else:
                self._cache.clear()
                self._timestamps.clear()
                self._access_counts.clear()
                self.hits = 0
                self.misses = 0
                self.evictions = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics for monitoring and optimization
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cached_modules": list(self._cache.keys()),
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": f"{hit_rate:.1f}%",
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "bulk_loads": self.bulk_loads,
            "preloads": self.preloads,
            "most_accessed": sorted(
                self._access_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5] if self._access_counts else [],
            "ttl_seconds": self.ttl_seconds
        }


# Global cache instance
module_config_cache = ModuleConfigCache()