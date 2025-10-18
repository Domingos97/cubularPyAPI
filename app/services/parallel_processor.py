"""
Parallel Survey Processing Service - ENHANCED
=============================================
Processes multiple surveys concurrently with optimized loading mechanisms.
Uses compressed cache, memory mapping, and batch loading for maximum performance.
"""

import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from app.services.compressed_survey_cache import compressed_survey_cache
from app.services.memory_mapped_survey_loader import memory_mapped_loader
from app.utils.logging import get_performance_logger

logger = get_performance_logger(__name__)


class ParallelProcessor:
    """
    Enhanced parallel processor with compressed caching and memory mapping
    Provides significant performance improvements for survey data loading
    """
    
    def __init__(self, max_concurrent: int = 8):  # Increased from 5 for better throughput
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.use_compression = True
        self.use_memory_mapping = True
        self.batch_size = 10  # Optimal batch size for concurrent loading
    
    async def search_surveys_parallel(
        self,
        query_embedding: List[float],
        survey_ids: List[str],
        threshold: float = 0.55,
        max_results_per_survey: int = 10,
        file_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search multiple surveys in parallel with optimized loading and optional file filtering
        """
        if not query_embedding or not survey_ids:
            return []
        
        start_time = time.time()
        
        # Use batch loading for better I/O efficiency
        if len(survey_ids) > self.batch_size:
            return await self._search_surveys_batched(
                query_embedding, survey_ids, threshold, max_results_per_survey, start_time, file_ids
            )
        
        # For smaller sets, use direct parallel processing
        return await self._search_surveys_direct(
            query_embedding, survey_ids, threshold, max_results_per_survey, start_time, file_ids
        )
    
    async def _search_surveys_batched(
        self,
        query_embedding: List[float],
        survey_ids: List[str],
        threshold: float,
        max_results_per_survey: int,
        start_time: float,
        file_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search surveys using batch loading for optimal I/O performance with optional file filtering
        """
        all_results = []
        successful_surveys = 0
        total_load_time = 0
        total_search_time = 0
        
        # Process surveys in batches
        for i in range(0, len(survey_ids), self.batch_size):
            batch_ids = survey_ids[i:i + self.batch_size]
            
            # Load batch of surveys using compressed cache
            batch_start = time.time()
            survey_data_batch = await compressed_survey_cache.get_batch(batch_ids)
            batch_load_time = (time.time() - batch_start) * 1000
            total_load_time += batch_load_time
            
            # Process each survey in the batch
            search_start = time.time()
            batch_tasks = [
                self._process_single_survey_data(
                    query_embedding=query_embedding,
                    survey_id=survey_id,
                    survey_data=survey_data_batch.get(survey_id),
                    threshold=threshold,
                    max_results=max_results_per_survey,
                    file_ids=file_ids
                )
                for survey_id in batch_ids
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            batch_search_time = (time.time() - search_start) * 1000
            total_search_time += batch_search_time
            
            # Process batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing survey {batch_ids[j]}: {str(result)}")
                elif result:
                    all_results.extend(result)
                    successful_surveys += 1
            
            logger.debug(f"Batch {i//self.batch_size + 1}: loaded in {batch_load_time:.1f}ms, searched in {batch_search_time:.1f}ms")
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('value', 0), reverse=True)
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batched processing: {successful_surveys}/{len(survey_ids)} surveys in {total_time:.1f}ms "
                   f"(load: {total_load_time:.1f}ms, search: {total_search_time:.1f}ms)")
        
        return all_results
    
    async def _search_surveys_direct(
        self,
        query_embedding: List[float],
        survey_ids: List[str],
        threshold: float,
        max_results_per_survey: int,
        start_time: float,
        file_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Direct parallel search for smaller survey sets with optional file filtering
        """
        # Create tasks for parallel processing
        tasks = [
            self._process_single_survey(
                query_embedding=query_embedding,
                survey_id=survey_id,
                threshold=threshold,
                max_results=max_results_per_survey,
                file_ids=file_ids
            )
            for survey_id in survey_ids
        ]
        
        # Execute in parallel with concurrency limit
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results and filter out exceptions
        all_results = []
        successful_surveys = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing survey {survey_ids[i]}: {str(result)}")
            elif result:
                all_results.extend(result)
                successful_surveys += 1
        
        # Sort by relevance score
        all_results.sort(key=lambda x: x.get('value', 0), reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Direct processing: {successful_surveys}/{len(survey_ids)} surveys in {processing_time:.1f}ms")
        
        return all_results
    
    async def _process_single_survey(
        self,
        query_embedding: List[float],
        survey_id: str,
        threshold: float,
        max_results: int,
        file_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a single survey with concurrency control and optimized loading
        """
        async with self.semaphore:
            try:
                # Load survey data from compressed cache
                survey_data = await compressed_survey_cache.get(survey_id)
                
                return await self._process_single_survey_data(
                    query_embedding, survey_id, survey_data, threshold, max_results, file_ids
                )
                
            except Exception as e:
                logger.error(f"Error in _process_single_survey for {survey_id}: {str(e)}")
                return []
    
    async def _process_single_survey_data(
        self,
        query_embedding: List[float],
        survey_id: str,
        survey_data: Optional[Dict[str, Any]],
        threshold: float,
        max_results: int,
        file_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process survey data for similarity search with optional file filtering
        """
        try:
            if not survey_data:
                return []
            
            embeddings = survey_data.get('embeddings')
            texts = survey_data.get('texts', [])
            
            if embeddings is None or len(embeddings) == 0 or len(texts) == 0:
                return []
            
            # If file filtering is requested, filter the data first
            if file_ids:
                # Get file_ids from survey data if available
                survey_file_ids = survey_data.get('file_ids', [])
                survey_files = survey_data.get('files', {})
                
                if survey_file_ids or survey_files:
                    # Filter by file_ids
                    filtered_indices = []
                    for i, text in enumerate(texts):
                        # Check if this text belongs to any of the allowed files
                        text_file_id = None
                        
                        # Try to find file_id from metadata if available
                        if i < len(survey_file_ids):
                            text_file_id = survey_file_ids[i]
                        elif 'metadata' in survey_data and i < len(survey_data['metadata']):
                            text_file_id = survey_data['metadata'][i].get('file_id')
                        
                        if text_file_id and text_file_id in file_ids:
                            filtered_indices.append(i)
                    
                    if not filtered_indices:
                        logger.debug(f"No matching files found in survey {survey_id} for file_ids: {file_ids}")
                        return []
                    
                    # Filter embeddings and texts to only include allowed files
                    embeddings = embeddings[filtered_indices] if isinstance(embeddings, np.ndarray) else [embeddings[i] for i in filtered_indices]
                    texts = [texts[i] for i in filtered_indices]
                    
                    logger.debug(f"File filtering reduced {len(survey_data.get('texts', []))} texts to {len(texts)} for survey {survey_id}")
            
            # Ensure embeddings is numpy array
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Handle edge cases
            if embeddings.size == 0:
                return []
            
            # Reshape if necessary
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            query_array = np.array(query_embedding).reshape(1, -1)
            
            # Compute cosine similarity
            similarities = cosine_similarity(query_array, embeddings)[0]
            
            # Find matches above threshold
            matching_indices = np.where(similarities >= threshold)[0]
            
            if len(matching_indices) == 0:
                return []
            
            # Create results with scores
            results = []
            for idx in matching_indices:
                if idx < len(texts):  # Safety check
                    result = {
                        'text': texts[idx],
                        'value': float(similarities[idx]),
                        'survey_id': survey_id,
                        'index': int(idx)
                    }
                    
                    # Add file_id if available and file filtering was used
                    if file_ids and 'file_ids' in survey_data and idx < len(survey_data['file_ids']):
                        result['file_id'] = survey_data['file_ids'][idx]
                    
                    results.append(result)
            
            # Sort by similarity score and limit results
            results.sort(key=lambda x: x['value'], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error processing survey data for {survey_id}: {str(e)}")
            return []
        
    '''
    async def get_survey_stats_parallel(self, survey_ids: List[str]) -> Dict[str, Any]:
        """
        Get statistics for multiple surveys in parallel using optimized loading
        """
        start_time = time.time()
        
        # Use batch loading for efficiency
        survey_data_batch = await compressed_survey_cache.get_batch(survey_ids)
        
        # Aggregate stats
        total_texts = 0
        total_embeddings = 0
        successful_surveys = 0
        
        for survey_id, survey_data in survey_data_batch.items():
            if survey_data:
                total_texts += len(survey_data.get('texts', []))
                total_embeddings += len(survey_data.get('embeddings', []))
                successful_surveys += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "surveys_processed": successful_surveys,
            "surveys_requested": len(survey_ids),
            "total_texts": total_texts,
            "total_embeddings": total_embeddings,
            "processing_time_ms": processing_time,
            "cache_stats": compressed_survey_cache.get_performance_stats()
        }
    
    async def preload_surveys(self, survey_ids: List[str]) -> Dict[str, Any]:
        """
        Preload multiple surveys into compressed cache in parallel
        """
        start_time = time.time()
        
        # Use optimized batch loading
        survey_data_batch = await compressed_survey_cache.get_batch(survey_ids)
        
        successful_loads = sum(1 for data in survey_data_batch.values() if data is not None)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "surveys_loaded": successful_loads,
            "surveys_requested": len(survey_ids),
            "processing_time_ms": processing_time,
            "cache_stats": compressed_survey_cache.get_performance_stats(),
            "memory_stats": memory_mapped_loader.get_mmap_stats()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for the optimized processor
        """
        cache_stats = compressed_survey_cache.get_performance_stats()
        mmap_stats = memory_mapped_loader.get_mmap_stats()
        
        return {
            "processor_config": {
                "max_concurrent": self.max_concurrent,
                "use_compression": self.use_compression,
                "use_memory_mapping": self.use_memory_mapping,
                "batch_size": self.batch_size
            },
            "cache_performance": cache_stats,
            "memory_mapping_performance": mmap_stats,
            "optimization_features": {
                "lz4_compression": cache_stats.get("bytes_saved", 0) > 0,
                "batch_loading": True,
                "concurrent_processing": True,
                "memory_mapped_io": mmap_stats.get("mmap_creates", 0) > 0
            }
        }

    '''
    

# Global optimized processor instance
parallel_processor = ParallelProcessor(max_concurrent=8)


async def get_parallel_processor() -> ParallelProcessor:
    """Dependency injection for FastAPI"""
    return parallel_processor