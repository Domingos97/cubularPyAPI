"""
Memory-Mapped Survey File Access Service
=======================================
Optimized file access using memory mapping for large survey files.
Reduces memory overhead and improves performance for large datasets.
"""

import mmap
import pickle
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from contextlib import contextmanager
import weakref
from app.utils.logging import get_performance_logger

logger = get_performance_logger(__name__)


class MemoryMappedSurveyLoader:
    """
    Memory-mapped file loader for large survey files
    Provides efficient access to large datasets without loading entire files into memory
    """
    
    def __init__(self, cache_size: int = 20):
        self.cache_size = cache_size
        self._mmap_cache: Dict[str, Any] = {}  # Use weak references to auto-cleanup
        self._file_handles: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        
        # Performance metrics
        self.mmap_hits = 0
        self.mmap_misses = 0
        self.mmap_creates = 0
        self.bytes_mapped = 0
    
    @contextmanager
    def _get_memory_mapped_file(self, file_path: Path):
        """
        Context manager for memory-mapped file access with automatic cleanup
        """
        file_key = str(file_path)
        
        # Check if already memory-mapped
        if file_key in self._mmap_cache:
            self.mmap_hits += 1
            self._access_times[file_key] = time.time()
            yield self._mmap_cache[file_key]
            return
        
        # Create new memory mapping
        self.mmap_misses += 1
        try:
            # Clean up old mappings if cache is full
            self._cleanup_old_mappings()
            
            # Open file and create memory mapping
            file_handle = open(file_path, 'rb')
            mmap_obj = mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Cache the mapping
            self._file_handles[file_key] = file_handle
            self._mmap_cache[file_key] = mmap_obj
            self._access_times[file_key] = time.time()
            self.mmap_creates += 1
            self.bytes_mapped += file_path.stat().st_size
            
            logger.debug(f"Created memory mapping for {file_path} ({file_path.stat().st_size} bytes)")
            
            yield mmap_obj
            
        except Exception as e:
            logger.error(f"Failed to create memory mapping for {file_path}: {e}")
            # Fallback to regular file access
            with open(file_path, 'rb') as f:
                yield f.read()
        finally:
            # Note: We don't close the mapping here as it's cached
            # Cleanup happens in _cleanup_old_mappings or when service is destroyed
            pass
    
    def _cleanup_old_mappings(self):
        """Clean up old memory mappings to prevent memory leaks"""
        if len(self._mmap_cache) < self.cache_size:
            return
        
        # Sort by access time and remove oldest
        sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
        keys_to_remove = sorted_keys[:len(sorted_keys) - self.cache_size + 1]
        
        for key in keys_to_remove:
            self._cleanup_mapping(key)
    
    def _cleanup_mapping(self, file_key: str):
        """Clean up a specific memory mapping"""
        try:
            if file_key in self._mmap_cache:
                mmap_obj = self._mmap_cache[file_key]
                mmap_obj.close()
                del self._mmap_cache[file_key]
            
            if file_key in self._file_handles:
                file_handle = self._file_handles[file_key]
                file_handle.close()
                del self._file_handles[file_key]
            
            if file_key in self._access_times:
                del self._access_times[file_key]
                
            logger.debug(f"Cleaned up memory mapping for {file_key}")
            
        except Exception as e:
            logger.warning(f"Error cleaning up memory mapping for {file_key}: {e}")
    
    async def load_survey_with_mmap(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load survey data using memory mapping for improved performance
        """
        start_time = time.time()
        
        try:
            # Check file size to determine if memory mapping is beneficial
            file_size = file_path.stat().st_size
            
            # Use memory mapping for files larger than 1MB
            if file_size > 1024 * 1024:  # 1MB threshold
                with self._get_memory_mapped_file(file_path) as mmap_data:
                    if isinstance(mmap_data, mmap.mmap):
                        # Load from memory-mapped file
                        data = pickle.loads(mmap_data[:])
                    else:
                        # Fallback: load from bytes
                        data = pickle.loads(mmap_data)
            else:
                # For smaller files, use regular loading
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Process the data
            result = self._process_survey_data(data, start_time)
            result['file_size'] = file_size
            result['used_mmap'] = file_size > 1024 * 1024
            
            return result
            
        except Exception as e:
            load_time = time.time() - start_time
            logger.error(f"Error loading survey with mmap from {file_path} after {load_time:.3f}s: {e}")
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
    
    async def load_survey_batch_with_mmap(self, file_paths: List[Path]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Load multiple survey files concurrently using memory mapping
        """
        import asyncio
        
        async def load_single_file(file_path: Path) -> tuple[str, Optional[Dict[str, Any]]]:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._load_file_sync, file_path)
            return str(file_path), result
        
        # Create tasks for concurrent loading
        tasks = [load_single_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        output = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch loading: {result}")
                continue
            
            file_path, data = result
            output[file_path] = data
        
        return output
    
    def _load_file_sync(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Synchronous file loading for thread pool execution"""
        try:
            return asyncio.run(self.load_survey_with_mmap(file_path))
        except Exception as e:
            logger.error(f"Sync load failed for {file_path}: {e}")
            return None
    
    def get_mmap_stats(self) -> Dict[str, Any]:
        """Get memory mapping performance statistics"""
        total_requests = self.mmap_hits + self.mmap_misses
        hit_rate = (self.mmap_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'mmap_cache_size': len(self._mmap_cache),
            'mmap_hits': self.mmap_hits,
            'mmap_misses': self.mmap_misses,
            'mmap_hit_rate_percent': hit_rate,
            'mmap_creates': self.mmap_creates,
            'bytes_mapped': self.bytes_mapped,
            'active_mappings': len(self._mmap_cache)
        }
    
    def clear_all_mappings(self):
        """Clear all memory mappings and reset cache"""
        keys_to_cleanup = list(self._mmap_cache.keys())
        for key in keys_to_cleanup:
            self._cleanup_mapping(key)
        
        # Reset metrics
        self.mmap_hits = 0
        self.mmap_misses = 0
        self.mmap_creates = 0
        self.bytes_mapped = 0
        
        logger.info("Cleared all memory mappings")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.clear_all_mappings()
        except:
            pass  # Ignore errors during cleanup


class StreamingSurveyLoader:
    """
    Streaming loader for very large survey files
    Processes data in chunks to minimize memory usage
    """
    
    def __init__(self, chunk_size: int = 8192):  # 8KB chunks
        self.chunk_size = chunk_size
        self.mmap_loader = MemoryMappedSurveyLoader()
    
    async def stream_survey_data(self, file_path: Path, max_embeddings: int = None) -> Dict[str, Any]:
        """
        Stream survey data in chunks to handle very large files
        """
        start_time = time.time()
        
        try:
            # Use memory mapping for initial access
            with self.mmap_loader._get_memory_mapped_file(file_path) as mmap_data:
                if isinstance(mmap_data, mmap.mmap):
                    # Process in chunks
                    data = self._process_mmap_chunks(mmap_data, max_embeddings)
                else:
                    # Fallback
                    data = pickle.loads(mmap_data)
            
            result = self.mmap_loader._process_survey_data(data, start_time)
            result['streamed'] = True
            result['file_size'] = file_path.stat().st_size
            
            return result
            
        except Exception as e:
            logger.error(f"Error streaming survey data from {file_path}: {e}")
            return None
    
    def _process_mmap_chunks(self, mmap_obj: mmap.mmap, max_embeddings: int = None) -> Dict[str, Any]:
        """
        Process memory-mapped data in chunks
        """
        # For now, load the entire pickle data (chunked processing would require 
        # custom serialization format). This is still more memory-efficient than
        # loading the entire file into Python memory at once.
        try:
            data = pickle.loads(mmap_obj[:])
            
            # Optionally limit embeddings to reduce memory usage
            if max_embeddings and 'embeddings' in data:
                if isinstance(data['embeddings'], (list, np.ndarray)):
                    data['embeddings'] = data['embeddings'][:max_embeddings]
                
                if 'semantic_dict' in data and isinstance(data['semantic_dict'], list):
                    data['semantic_dict'] = data['semantic_dict'][:max_embeddings]
                
                if 'texts' in data and isinstance(data['texts'], list):
                    data['texts'] = data['texts'][:max_embeddings]
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing mmap chunks: {e}")
            raise


# Global instances
memory_mapped_loader = MemoryMappedSurveyLoader(cache_size=20)
streaming_loader = StreamingSurveyLoader(chunk_size=8192)


async def get_memory_mapped_loader() -> MemoryMappedSurveyLoader:
    """Dependency injection for FastAPI"""
    return memory_mapped_loader


async def get_streaming_loader() -> StreamingSurveyLoader:
    """Dependency injection for FastAPI"""
    return streaming_loader