"""
Fast Search Service - Direct Pickle File Search
===============================================
Mimics the original TypeScript->PythonAPI flow for ultra-fast responses.
No database queries, just direct file access and vector operations.
"""

import os
import pickle
import time
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import asyncio
from cachetools import LRUCache, TTLCache

from app.utils.logging import get_logger

logger = get_logger(__name__)


class FastSearchService:
    """
    Ultra-fast search service that directly accesses pickle files
    Similar to the original Python search API (port 8001)
    Enhanced with LRU cache for better memory management
    """
    
    def __init__(self):
        # PERFORMANCE OPTIMIZATION: Use LRU cache with size limit and TTL cache for embeddings
        self.survey_cache = LRUCache(maxsize=100)  # Limit cache size to prevent memory bloat
        self.embedding_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache embeddings for 1 hour
        self.base_paths = [
            Path("survey_data")                 # PyAPI processed survey data directory
        ]
        logger.info("FastSearchService initialized for PyAPI survey_data directory access")
    
    def load_survey_data_sync(self, survey_id: str) -> Optional[Dict[str, Any]]:
        """
        Synchronously load survey data from pickle files (fast)
        """
        # Check cache first
        if survey_id in self.survey_cache:
            logger.debug(f"Cache hit for survey {survey_id}")
            return self.survey_cache[survey_id]
        
        logger.info(f"Loading survey data for: {survey_id}")
        
        try:
            # Try each possible path
            for i, base_path in enumerate(self.base_paths):
                logger.debug(f"Trying path {i+1}/{len(self.base_paths)}: {base_path}")
                
                pickle_path = base_path / survey_id / "survey_data.pkl"
                logger.debug(f"Looking for: {pickle_path.absolute()}")
                
                if pickle_path.exists():
                    logger.info(f"Found survey data at: {pickle_path.absolute()}")
                    with open(pickle_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Extract texts from semantic_dict
                    texts = []
                    if 'semantic_dict' in data and isinstance(data['semantic_dict'], list):
                        texts = [entry.get('cleanedText', '') for entry in data['semantic_dict'] if entry.get('cleanedText')]
                        logger.info(f"Extracted {len(texts)} texts from semantic_dict")
                    elif 'texts' in data:
                        texts = data['texts']
                        logger.info(f"Found {len(texts)} texts in data['texts']")
                    else:
                        logger.warning(f"No texts found in survey data for {survey_id}")
                    
                    embeddings_count = len(data.get('embeddings', []))
                    logger.info(f"Found {embeddings_count} embeddings for survey {survey_id}")
                    
                    result = {
                        'embeddings': np.array(data['embeddings']),
                        'texts': texts,
                        'semantic_dict': data.get('semantic_dict', []),
                        'metadata': data.get('metadata', {})
                    }
                    
                    # Cache it
                    self.survey_cache[survey_id] = result
                    logger.info(f"Successfully loaded and cached survey {survey_id}: {len(texts)} responses, {embeddings_count} embeddings")
                    return result
                
                # Check subdirectories for multi-file surveys
                survey_dir = base_path / survey_id
                if survey_dir.exists() and survey_dir.is_dir():
                    logger.debug(f"Checking subdirectories in: {survey_dir}")
                    for item in survey_dir.iterdir():
                        if item.is_dir():
                            sub_pickle = item / "survey_data.pkl"
                            logger.debug(f"Checking subdir pickle: {sub_pickle}")
                            if sub_pickle.exists():
                                logger.info(f"Found survey data in subdir: {sub_pickle}")
                                with open(sub_pickle, 'rb') as f:
                                    data = pickle.load(f)
                                
                                # Process the same way
                                texts = []
                                if 'semantic_dict' in data and isinstance(data['semantic_dict'], list):
                                    texts = [entry.get('cleanedText', '') for entry in data['semantic_dict'] if entry.get('cleanedText')]
                                elif 'texts' in data:
                                    texts = data['texts']
                                
                                result = {
                                    'embeddings': np.array(data['embeddings']),
                                    'texts': texts,
                                    'semantic_dict': data.get('semantic_dict', []),
                                    'metadata': data.get('metadata', {})
                                }
                                
                                self.survey_cache[survey_id] = result
                                logger.info(f"Successfully loaded survey {survey_id} from subdir {item.name}: {len(texts)} responses")
                                return result
            
            logger.warning(f"No survey data found for {survey_id} in any of the {len(self.base_paths)} search paths")
            logger.debug(f"Search paths were: {[str(p.absolute()) for p in self.base_paths]}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading survey {survey_id}: {e}", exc_info=True)
            return None
    
    async def search_surveys(
        self,
        question: str,
        survey_ids: List[str],
        user_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search surveys using semantic search with question text
        This method bridges the gap between the semantic chat endpoint and fast_search
        """
        try:
            # Import embedding service to generate embeddings from question
            from app.services.embedding_service import embedding_service
            
            # Generate embedding for the question
            query_embedding = await embedding_service.generate_embedding(question)
            if not query_embedding:
                logger.warning(f"Failed to generate embedding for question: {question}")
                return {"results": [], "total": 0, "processing_time": 0}
            
            # Call the existing fast_search method
            search_result = await self.fast_search(
                query_embedding=query_embedding,
                survey_ids=survey_ids,
                threshold=0.25,  # Default threshold
                max_results=limit,
                query_text=question
            )
            
            # Return in the format expected by semantic_chat
            return {
                "results": search_result.get("results", []),
                "total": search_result.get("total_results", 0),
                "processing_time": search_result.get("processing_time", 0),
                "user_id": user_id,
                "query": question
            }
            
        except Exception as e:
            logger.error(f"Error in search_surveys: {str(e)}")
            return {"results": [], "total": 0, "processing_time": 0}
    
    async def fast_search(
        self,
        query_embedding: List[float],
        survey_ids: List[str],
        threshold: float = 0.25,
        max_results: int = 1000,
        query_text: str = ""
    ) -> Dict[str, Any]:
        """
        Ultra-fast search operation - no database, direct file access
        """
        start_time = time.time()
        
        logger.info(f"Starting fast_search for {len(survey_ids)} surveys with threshold {threshold}")
        logger.debug(f"Survey IDs: {survey_ids}")
        logger.debug(f"Query: {query_text[:100]}...")
        
        try:
            if not query_embedding or not survey_ids:
                logger.warning(f"Missing required parameters: embedding={bool(query_embedding)}, survey_ids={bool(survey_ids)}")
                return self._empty_response(start_time)
            
            all_results = []
            surveys_processed = 0
            surveys_found = 0
            
            # Process each survey synchronously for speed
            for survey_id in survey_ids:
                logger.debug(f"Processing survey: {survey_id}")
                survey_data = self.load_survey_data_sync(survey_id)
                if not survey_data:
                    logger.warning(f"No data loaded for survey: {survey_id}")
                    continue
                
                surveys_found += 1
                embeddings = survey_data['embeddings']
                texts = survey_data['texts']
                
                if len(embeddings) == 0 or len(texts) == 0:
                    logger.warning(f"Survey {survey_id} has no embeddings ({len(embeddings)}) or texts ({len(texts)})")
                    continue
                
                logger.debug(f"Survey {survey_id}: {len(embeddings)} embeddings, {len(texts)} texts")
                
                # ULTRA FAST: Direct numpy operations
                similarities = cosine_similarity([query_embedding], embeddings)[0]
                
                # Filter by threshold
                valid_mask = similarities >= threshold
                valid_indices = np.where(valid_mask)[0]
                
                matches_found = len(valid_indices)
                logger.debug(f"Survey {survey_id}: {matches_found} matches above threshold {threshold}")
                
                if len(valid_indices) == 0:
                    continue
                
                # Extract results
                valid_similarities = similarities[valid_indices]
                valid_texts = [texts[i] for i in valid_indices]
                
                # Create result objects
                for i, similarity in enumerate(valid_similarities):
                    all_results.append({
                        'text': valid_texts[i],
                        'value': float(similarity),
                        'survey_id': survey_id,
                        'index': int(valid_indices[i])
                    })
                
                surveys_processed += 1
                logger.debug(f"Survey {survey_id}: Added {len(valid_similarities)} results")
            
            # Sort and limit
            all_results.sort(key=lambda x: x['value'], reverse=True)
            if max_results > 0:
                all_results = all_results[:max_results]
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Fast search completed in {processing_time:.1f}ms: {len(all_results)} results from {surveys_processed}/{surveys_found}/{len(survey_ids)} surveys")
            
            # Generate simple data snapshot for AI
            data_snapshot = self._generate_simple_snapshot(all_results, query_text)
            
            # Simple demographics (like original)
            demographics = {
                "age": [["25-34", 0.3], ["35-44", 0.25], ["18-24", 0.2]],
                "gender": [["Female", 0.52], ["Male", 0.46], ["Other", 0.02]],
                "location": [["Urban", 0.6], ["Suburban", 0.3], ["Rural", 0.1]]
            }
            
            psychology = {
                "op": 0.65, "co": 0.58, "ex": 0.62, "ag": 0.71, "ne": 0.45
            }
            
            return {
                "responses": all_results,
                "demographics": demographics,
                "psychology": psychology,
                "phrases": {"societal": [], "disposition": [], "media": []},
                "dataSnapshot": data_snapshot,
                "metadata": {
                    "total_matches": len(all_results),
                    "search_strategy": "fast_direct_search",
                    "processing_time": processing_time,
                    "threshold": threshold,
                    "surveys_processed": surveys_processed,
                    "surveys_found": surveys_found,
                    "surveys_requested": len(survey_ids),
                    "cache_hits": sum(1 for sid in survey_ids if sid in self.survey_cache)
                }
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Fast search error after {processing_time:.1f}ms: {e}")
            return self._empty_response(start_time, error=str(e))
    
    def _generate_simple_snapshot(self, results: List[Dict], query_text: str) -> Dict[str, Any]:
        """Generate simple data snapshot for AI analysis"""
        if not results:
            return {"stats": [], "rawData": {"totalResults": 0, "texts": [], "query": query_text}}
        
        texts = [result['text'] for result in results if result.get('text')]
        high_relevance = len([r for r in results if r.get('value', 0) > 0.8])
        
        return {
            "totalResults": len(results),
            "highRelevanceCount": high_relevance,
            "averageScore": sum(r.get('value', 0) for r in results) / len(results) if results else 0,
            "texts": texts,
            "query": query_text
        }
    
    def _empty_response(self, start_time: float, error: str = "") -> Dict[str, Any]:
        """Generate empty response with timing"""
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "responses": [],
            "demographics": {"age": [], "gender": [], "location": []},
            "psychology": {"op": 0, "co": 0, "ex": 0, "ag": 0, "ne": 0},
            "phrases": {"societal": [], "disposition": [], "media": []},
            "dataSnapshot": {"stats": [], "rawData": {"totalResults": 0, "texts": [], "query": ""}},
            "metadata": {
                "total_matches": 0,
                "search_strategy": "fast_direct_search",
                "processing_time": processing_time,
                "surveys_processed": 0,
                "surveys_found": 0,
                "surveys_requested": 0,
                "cache_hits": 0,
                "error": error or "No survey data found"
            }
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get enhanced cache information with performance metrics"""
        return {
            "survey_cache": {
                "cached_surveys": list(self.survey_cache.keys()),
                "cache_size": len(self.survey_cache),
                "max_size": self.survey_cache.maxsize,
                "current_size": self.survey_cache.currsize,
                "total_responses": sum(len(data.get('texts', [])) for data in self.survey_cache.values())
            },
            "embedding_cache": {
                "cache_size": len(self.embedding_cache),
                "max_size": self.embedding_cache.maxsize,
                "ttl": self.embedding_cache.ttl
            }
        }
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear all caches and return number of items cleared"""
        survey_size = len(self.survey_cache)
        embedding_size = len(self.embedding_cache)
        
        self.survey_cache.clear()
        self.embedding_cache.clear()
        
        logger.info(f"Cleared fast search caches: {survey_size} surveys, {embedding_size} embeddings")
        return {
            "surveys_cleared": survey_size,
            "embeddings_cleared": embedding_size
        }


# Singleton instance
fast_search_service = FastSearchService()