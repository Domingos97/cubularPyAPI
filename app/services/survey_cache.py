"""
Survey Data Cache Service
========================
High-performance caching for survey data with LRU eviction and preloading.
Eliminates repeated pickle file loading from disk.
"""

import asyncio
import pickle
import numpy as np
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import OrderedDict
from threading import RLock
from app.utils.logging import get_performance_logger

logger = get_performance_logger(__name__)


class LRUSurveyCache:
    """
    Thread-safe LRU cache for survey data with intelligent preloading
    """
    
    def __init__(self, max_size: int = 50, preload_size: int = 10):
        self.max_size = max_size
        self.preload_size = preload_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._access_counts: Dict[str, int] = {}
        self._load_times: Dict[str, float] = {}
        self._lock = RLock()
        self.base_paths = [Path("survey_data")]
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
    
    def _move_to_end(self, key: str):
        """Move key to end (most recently used)"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
    
    def _evict_lru(self):
        """Evict least recently used item"""
        with self._lock:
            if len(self._cache) >= self.max_size:
                lru_key, _ = self._cache.popitem(last=False)
                self._access_counts.pop(lru_key, None)
                self._load_times.pop(lru_key, None)
                self.evictions += 1
                logger.debug(f"Evicted survey {lru_key} from cache")
    
    async def get(self, survey_id: str) -> Optional[Dict[str, Any]]:
        """
        Get survey data from cache or load from disk
        """
        # Check cache first
        with self._lock:
            if survey_id in self._cache:
                self._move_to_end(survey_id)
                self.hits += 1
                return self._cache[survey_id]
        
        # Cache miss - load from disk
        self.misses += 1
        survey_data = await self._load_survey_from_disk(survey_id)
        
        if survey_data:
            with self._lock:
                # Evict LRU if necessary
                self._evict_lru()
                
                # Add to cache
                self._cache[survey_id] = survey_data
                self._access_counts[survey_id] = 1
                self._load_times[survey_id] = time.time()
                self.loads += 1
        
        return survey_data
    
    async def _load_survey_from_disk(self, survey_id: str) -> Optional[Dict[str, Any]]:
        """
        Load survey data from disk with optimized I/O
        """
        start_time = time.time()
        
        try:
            # Try each possible path
            for base_path in self.base_paths:
                pickle_path = base_path / survey_id / "survey_data.pkl"
                
                if pickle_path.exists():
                    # Use thread pool for I/O to avoid blocking
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(None, self._load_pickle_file, pickle_path)
                    
                    if data:
                        # Extract and process texts
                        texts = []
                        if 'semantic_dict' in data and isinstance(data['semantic_dict'], list):
                            texts = [entry.get('cleanedText', '') for entry in data['semantic_dict'] if entry.get('cleanedText')]
                        elif 'texts' in data:
                            texts = data['texts']
                        
                        embeddings = np.array(data.get('embeddings', []))
                        
                        result = {
                            'embeddings': embeddings,
                            'texts': texts,
                            'semantic_dict': data.get('semantic_dict', []),
                            'metadata': data.get('metadata', {}),
                            'load_time': time.time() - start_time
                        }
                        
                        return result
                
                # Check subdirectories for multi-file surveys
                survey_dir = base_path / survey_id
                if survey_dir.exists() and survey_dir.is_dir():
                    for item in survey_dir.iterdir():
                        if item.is_dir():
                            sub_pickle = item / "survey_data.pkl"
                            if sub_pickle.exists():
                                loop = asyncio.get_event_loop()
                                data = await loop.run_in_executor(None, self._load_pickle_file, sub_pickle)
                                
                                if data:
                                    texts = []
                                    if 'semantic_dict' in data and isinstance(data['semantic_dict'], list):
                                        texts = [entry.get('cleanedText', '') for entry in data['semantic_dict'] if entry.get('cleanedText')]
                                    elif 'texts' in data:
                                        texts = data['texts']
                                    
                                    embeddings = np.array(data.get('embeddings', []))
                                    
                                    result = {
                                        'embeddings': embeddings,
                                        'texts': texts,
                                        'semantic_dict': data.get('semantic_dict', []),
                                        'metadata': data.get('metadata', {}),
                                        'load_time': time.time() - start_time
                                    }
                                    
                                    return result
            
            load_time = time.time() - start_time
            logger.warning(f"No survey data found for {survey_id} after {load_time:.3f}s search")
            return None
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Error loading survey {survey_id} after {load_time:.3f}s: {e}")
            return None
    
    def _load_pickle_file(self, pickle_path: Path) -> Optional[Dict[str, Any]]:
        """
        Synchronous pickle file loading for thread pool execution
        """
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load pickle file {pickle_path}: {e}")
            return None
    
    async def preload_frequent_surveys(self, survey_ids: List[str] = None):
        """
        Preload frequently accessed surveys into cache
        """
        if not survey_ids:
            # Auto-discover surveys if none provided
            survey_ids = await self._discover_available_surveys()
        
        # Limit to preload_size most recent/frequent surveys
        surveys_to_load = survey_ids[:self.preload_size]
        
        logger.info(f"Preloading {len(surveys_to_load)} surveys into cache...")
        start_time = time.time()
        
        # Load surveys in parallel
        tasks = [self.get(survey_id) for survey_id in surveys_to_load]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_loads = sum(1 for result in results if result and not isinstance(result, Exception))
        load_time = time.time() - start_time
        
        logger.info(f"Preloaded {successful_loads}/{len(surveys_to_load)} surveys in {load_time:.3f}s")
    
    async def _discover_available_surveys(self) -> List[str]:
        """
        Discover available surveys from the file system
        """
        survey_ids = []
        
        for base_path in self.base_paths:
            if base_path.exists():
                for item in base_path.iterdir():
                    if item.is_dir():
                        survey_pickle = item / "survey_data.pkl"
                        if survey_pickle.exists():
                            survey_ids.append(item.name)
                        else:
                            # Check subdirectories
                            for sub_item in item.iterdir():
                                if sub_item.is_dir() and (sub_item / "survey_data.pkl").exists():
                                    survey_ids.append(item.name)
                                    break
        
        return survey_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_ratio = self.hits / total_requests if total_requests > 0 else 0
            
            # Get most accessed surveys
            sorted_access = sorted(self._access_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": hit_ratio,
                "loads": self.loads,
                "evictions": self.evictions,
                "cached_surveys": list(self._cache.keys()),
                "most_accessed": sorted_access[:5],
                "total_texts": sum(len(data.get('texts', [])) for data in self._cache.values()),
                "total_embeddings": sum(len(data.get('embeddings', [])) for data in self._cache.values())
            }
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()
            self._load_times.clear()
            self.hits = self.misses = self.loads = self.evictions = 0


# Global cache instance
survey_cache = LRUSurveyCache(max_size=50, preload_size=10)