"""
Compressed Survey Data Cache Service
===================================
Enhanced high-performance caching with LZ4 compression for survey data.
Reduces I/O overhead and memory usage through intelligent compression.
"""

import asyncio
import pickle
import lz4.frame
import numpy as np
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import OrderedDict
from threading import RLock
from app.utils.logging import get_performance_logger

logger = get_performance_logger(__name__)


class CompressedSurveyCache:
    """
    Thread-safe LRU cache with LZ4 compression for survey data
    Provides intelligent compression, batch loading, and memory optimization
    """
    
    def __init__(self, max_size: int = 50, preload_size: int = 10, enable_compression: bool = True):
        self.max_size = max_size
        self.preload_size = preload_size
        self.enable_compression = enable_compression
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._compressed_cache: OrderedDict[str, bytes] = OrderedDict()  # LZ4 compressed data
        self._access_counts: Dict[str, int] = {}
        self._load_times: Dict[str, float] = {}
        self._compression_ratios: Dict[str, float] = {}
        self._lock = RLock()
        self.base_paths = [Path("survey_data")]
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.evictions = 0
        self.compression_time = 0
        self.decompression_time = 0
        self.bytes_saved = 0
        
        # Metadata cache for faster file system operations
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
    
    def _move_to_end(self, key: str):
        """Move key to end (most recently used)"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
            elif key in self._compressed_cache:
                self._compressed_cache.move_to_end(key)
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
    
    def _evict_lru(self):
        """Evict least recently used item"""
        with self._lock:
            # Prefer evicting from uncompressed cache first
            if len(self._cache) >= self.max_size // 2:
                if self._cache:
                    lru_key, lru_data = self._cache.popitem(last=False)
                    
                    # Compress before evicting if compression is enabled
                    if self.enable_compression and len(self._compressed_cache) < self.max_size:
                        compressed_data = self._compress_survey_data(lru_data)
                        if compressed_data:
                            self._compressed_cache[lru_key] = compressed_data
                    
                    self.evictions += 1
                    logger.debug(f"Evicted uncompressed survey {lru_key} from cache")
            
            # Evict from compressed cache if needed
            if len(self._compressed_cache) >= self.max_size:
                if self._compressed_cache:
                    lru_key, _ = self._compressed_cache.popitem(last=False)
                    self._access_counts.pop(lru_key, None)
                    self._load_times.pop(lru_key, None)
                    self._compression_ratios.pop(lru_key, None)
                    self.evictions += 1
                    logger.debug(f"Evicted compressed survey {lru_key} from cache")
    
    def _compress_survey_data(self, data: Dict[str, Any]) -> Optional[bytes]:
        """Compress survey data using LZ4"""
        if not self.enable_compression:
            return None
            
        try:
            start_time = time.time()
            
            # Serialize data with pickle first
            pickled_data = pickle.dumps(data)
            original_size = len(pickled_data)
            
            # Compress with LZ4 (high compression mode)
            compressed_data = lz4.frame.compress(
                pickled_data, 
                compression_level=9,  # High compression
                auto_flush=True
            )
            
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            self.compression_time += (time.time() - start_time) * 1000
            self.bytes_saved += (original_size - compressed_size)
            
            logger.debug(f"Compressed survey data: {original_size} -> {compressed_size} bytes (ratio: {compression_ratio:.2f}x)")
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Failed to compress survey data: {e}")
            return None
    
    def _decompress_survey_data(self, compressed_data: bytes) -> Optional[Dict[str, Any]]:
        """Decompress survey data from LZ4"""
        try:
            start_time = time.time()
            
            # Decompress with LZ4
            pickled_data = lz4.frame.decompress(compressed_data)
            
            # Deserialize with pickle
            data = pickle.loads(pickled_data)
            
            self.decompression_time += (time.time() - start_time) * 1000
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to decompress survey data: {e}")
            return None
    
    async def get(self, survey_id: str) -> Optional[Dict[str, Any]]:
        """
        Get survey data from cache or load from disk with compression support
        """
        # Check uncompressed cache first (fastest access)
        with self._lock:
            if survey_id in self._cache:
                self._move_to_end(survey_id)
                self.hits += 1
                return self._cache[survey_id]
        
        # Check compressed cache
        with self._lock:
            if survey_id in self._compressed_cache:
                compressed_data = self._compressed_cache[survey_id]
                self._move_to_end(survey_id)
                
                # Decompress data
                decompressed_data = self._decompress_survey_data(compressed_data)
                if decompressed_data:
                    # Move to uncompressed cache for faster future access
                    self._evict_lru()
                    self._cache[survey_id] = decompressed_data
                    self.hits += 1
                    return decompressed_data
        
        # Cache miss - load from disk
        self.misses += 1
        survey_data = await self._load_survey_from_disk(survey_id)
        
        if survey_data:
            with self._lock:
                # Evict LRU if necessary
                self._evict_lru()
                
                # Add to uncompressed cache
                self._cache[survey_id] = survey_data
                self._access_counts[survey_id] = 1
                self._load_times[survey_id] = time.time()
                self.loads += 1
        
        return survey_data
    
    async def get_batch(self, survey_ids: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Efficiently load multiple surveys concurrently with optimized I/O
        """
        results = {}
        uncached_ids = []
        
        # Check cache for all surveys first
        with self._lock:
            for survey_id in survey_ids:
                if survey_id in self._cache:
                    self._move_to_end(survey_id)
                    results[survey_id] = self._cache[survey_id]
                    self.hits += 1
                elif survey_id in self._compressed_cache:
                    compressed_data = self._compressed_cache[survey_id]
                    decompressed_data = self._decompress_survey_data(compressed_data)
                    if decompressed_data:
                        self._move_to_end(survey_id)
                        results[survey_id] = decompressed_data
                        self.hits += 1
                    else:
                        uncached_ids.append(survey_id)
                else:
                    uncached_ids.append(survey_id)
        
        # Load uncached surveys concurrently
        if uncached_ids:
            self.misses += len(uncached_ids)
            
            # Create concurrent loading tasks
            tasks = [self._load_survey_from_disk(survey_id) for survey_id in uncached_ids]
            loaded_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process loaded data
            for survey_id, data in zip(uncached_ids, loaded_data):
                if data and not isinstance(data, Exception):
                    with self._lock:
                        self._evict_lru()
                        self._cache[survey_id] = data
                        self._access_counts[survey_id] = 1
                        self._load_times[survey_id] = time.time()
                        self.loads += 1
                    results[survey_id] = data
                else:
                    results[survey_id] = None
        
        return results
    
    async def _load_survey_from_disk(self, survey_id: str) -> Optional[Dict[str, Any]]:
        """
        Load survey data from disk with optimized I/O and metadata caching
        """
        start_time = time.time()
        
        # Check metadata cache first
        if survey_id in self._metadata_cache:
            metadata = self._metadata_cache[survey_id]
            pickle_path = Path(metadata['path'])
            
            # Check if file still exists and hasn't been modified
            if pickle_path.exists():
                current_mtime = pickle_path.stat().st_mtime
                if current_mtime == metadata['mtime']:
                    # Use cached path directly
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(None, self._load_pickle_file, pickle_path)
                    
                    if data:
                        return self._process_survey_data(data, start_time)
        
        try:
            # Search for survey data files
            for base_path in self.base_paths:
                pickle_path = base_path / survey_id / "survey_data.pkl"
                
                if pickle_path.exists():
                    # Cache metadata for future use
                    self._metadata_cache[survey_id] = {
                        'path': str(pickle_path),
                        'mtime': pickle_path.stat().st_mtime,
                        'size': pickle_path.stat().st_size
                    }
                    
                    # Use thread pool for I/O to avoid blocking
                    loop = asyncio.get_event_loop()
                    data = await loop.run_in_executor(None, self._load_pickle_file, pickle_path)
                    
                    if data:
                        return self._process_survey_data(data, start_time)
                
                # Check subdirectories for multi-file surveys
                survey_dir = base_path / survey_id
                if survey_dir.exists() and survey_dir.is_dir():
                    for item in survey_dir.iterdir():
                        if item.is_dir():
                            sub_pickle = item / "survey_data.pkl"
                            if sub_pickle.exists():
                                # Cache metadata
                                self._metadata_cache[survey_id] = {
                                    'path': str(sub_pickle),
                                    'mtime': sub_pickle.stat().st_mtime,
                                    'size': sub_pickle.stat().st_size
                                }
                                
                                loop = asyncio.get_event_loop()
                                data = await loop.run_in_executor(None, self._load_pickle_file, sub_pickle)
                                
                                if data:
                                    return self._process_survey_data(data, start_time)
            
            load_time = time.time() - start_time
            logger.warning(f"No survey data found for {survey_id} after {load_time:.3f}s search")
            return None
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Error loading survey {survey_id} after {load_time:.3f}s: {e}")
            return None
    
    def _process_survey_data(self, data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Process loaded survey data into standardized format"""
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
        Preload frequently accessed surveys into cache with compression
        """
        if not survey_ids:
            # Auto-discover surveys if none provided
            survey_ids = await self._discover_available_surveys()
        
        # Limit to preload_size most recent/frequent surveys
        surveys_to_load = survey_ids[:self.preload_size]
        
        logger.info(f"Preloading {len(surveys_to_load)} surveys into compressed cache...")
        start_time = time.time()
        
        # Use batch loading for efficiency
        results = await self.get_batch(surveys_to_load)
        
        successful_loads = sum(1 for result in results.values() if result is not None)
        load_time = time.time() - start_time
        
        logger.info(f"Preloaded {successful_loads}/{len(surveys_to_load)} surveys in {load_time:.3f}s")
    
    async def _discover_available_surveys(self) -> List[str]:
        """
        Discover available surveys from the file system with caching
        """
        survey_ids = []
        
        for base_path in self.base_paths:
            if not base_path.exists():
                continue
                
            try:
                # Use os.scandir for faster directory scanning
                with os.scandir(base_path) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            survey_dir = Path(entry.path)
                            if (survey_dir / "survey_data.pkl").exists():
                                survey_ids.append(entry.name)
                            else:
                                # Check subdirectories
                                try:
                                    with os.scandir(survey_dir) as sub_entries:
                                        for sub_entry in sub_entries:
                                            if sub_entry.is_dir():
                                                sub_dir = Path(sub_entry.path)
                                                if (sub_dir / "survey_data.pkl").exists():
                                                    survey_ids.append(entry.name)
                                                    break
                                except (PermissionError, OSError):
                                    continue
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot scan directory {base_path}: {e}")
        
        return survey_ids
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_size': len(self._cache),
                'compressed_cache_size': len(self._compressed_cache),
                'total_cached_items': len(self._cache) + len(self._compressed_cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate_percent': hit_rate,
                'loads': self.loads,
                'evictions': self.evictions,
                'compression_time_ms': self.compression_time,
                'decompression_time_ms': self.decompression_time,
                'bytes_saved': self.bytes_saved,
                'metadata_cache_size': len(self._metadata_cache),
                'average_compression_ratio': sum(self._compression_ratios.values()) / len(self._compression_ratios) if self._compression_ratios else 0
            }
    
    def clear_cache(self):
        """Clear all caches"""
        with self._lock:
            self._cache.clear()
            self._compressed_cache.clear()
            self._access_counts.clear()
            self._load_times.clear()
            self._compression_ratios.clear()
            self._metadata_cache.clear()
            
            # Reset metrics
            self.hits = 0
            self.misses = 0
            self.loads = 0
            self.evictions = 0
            self.compression_time = 0
            self.decompression_time = 0
            self.bytes_saved = 0


# Global instance with compression enabled
compressed_survey_cache = CompressedSurveyCache(max_size=50, preload_size=10, enable_compression=True)


async def get_compressed_survey_cache() -> CompressedSurveyCache:
    """Dependency injection for FastAPI"""
    return compressed_survey_cache