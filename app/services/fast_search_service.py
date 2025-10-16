"""
Fast Search Service - OPTIMIZED for Performance
==============================================
Uses optimized parallel processing, compressed caching, and memory mapping for maximum performance.
Incorporates Phase 2B data loading optimizations.
"""

import time
from typing import Dict, List, Any

from app.utils.logging import get_performance_logger
from app.services.compressed_survey_cache import compressed_survey_cache
from app.services.parallel_processor import parallel_processor

logger = get_performance_logger(__name__)


class FastSearchService:
    """
    OPTIMIZED search service using parallel processing and persistent caching
    """
    
    def __init__(self):
        # Using optimized survey_cache service
        pass
    
    async def search_surveys(
        self,
        question: str,
        survey_ids: List[str],
        user_id: str,
        limit: int = 10,
        file_ids: List[str] = None
    ) -> Dict[str, Any]:
        """
        SIMPLIFIED search - direct single query approach with optional file filtering
        """
        try:
            start_time = time.time()
            
            # Always use direct single query search - simpler and faster
            search_result = await self._single_query_search(question, survey_ids, limit, file_ids)
            
            # Format result
            result = {
                "results": search_result.get("responses", []),
                "total": len(search_result.get("responses", [])),
                "processing_time": search_result.get("metadata", {}).get("processing_time", 0),
                "user_id": user_id,
                "query": question,
                "search_strategy": search_result.get("metadata", {}).get("search_strategy", "unknown"),
                "file_filtered": bool(file_ids)
            }
            
            # REMOVED CACHING FOR PERFORMANCE
            
            processing_time = (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            logger.error(f"Error in search_surveys: {str(e)}")
            return {"results": [], "total": 0, "processing_time": 0, "user_id": user_id, "query": question}
    
    async def _single_query_search(self, query: str, survey_ids: List[str], limit: int, file_ids: List[str] = None) -> Dict[str, Any]:
        """Perform single query search using existing fast_search method"""
        from app.services.embedding_service import embedding_service
        
        query_embedding = await embedding_service.generate_embedding(query)
        if not query_embedding:
            logger.warning(f"Failed to generate embedding for query: {query}")
            return {"responses": [], "metadata": {"processing_time": 0}}
        
        return await self.fast_search(
            query_embedding=query_embedding,
            survey_ids=survey_ids,
            threshold=0.55,  # Match legacy API threshold
            max_results=limit,
            query_text=query,
            file_ids=file_ids
        )
    
    async def fast_search(
        self,
        query_embedding: List[float],
        survey_ids: List[str],
        threshold: float = 0.55,  # Match legacy API threshold
        max_results: int = 1000,
        query_text: str = "",
        file_ids: List[str] = None
    ) -> Dict[str, Any]:
        """OPTIMIZED search using parallel processing and caching with optional file filtering"""
        start_time = time.time()
        
        try:
            if not query_embedding or not survey_ids:
                return self._empty_response(start_time)
            
            # Use parallel processor for multiple surveys
            all_results = await parallel_processor.search_surveys_parallel(
                query_embedding=query_embedding,
                survey_ids=survey_ids,
                threshold=threshold,
                max_results_per_survey=max_results // len(survey_ids) + 10 if len(survey_ids) > 1 else max_results,
                file_ids=file_ids
            )
            
            # If file_ids specified, filter results to only include those files
            if file_ids:
                logger.info(f"Filtering search results to {len(file_ids)} specific files")
                filtered_results = []
                for result in all_results:
                    # Check if result has file_id and it's in our allowed list
                    result_file_id = result.get("file_id") or result.get("fileId")
                    if result_file_id and result_file_id in file_ids:
                        filtered_results.append(result)
                all_results = filtered_results
                logger.info(f"File filtering reduced results from {len(all_results)} to {len(filtered_results)}")
            
            # Limit total results
            if max_results > 0:
                all_results = all_results[:max_results]
            
            processing_time = (time.time() - start_time) * 1000
            
            # Generate simple data snapshot for AI
            data_snapshot = self._generate_simple_snapshot(all_results, query_text)
            
            return {
                "responses": all_results,
                "dataSnapshot": data_snapshot,
                "metadata": {
                    "total_matches": len(all_results),
                    "processing_time": processing_time,
                    "file_filtered": bool(file_ids),
                    "search_strategy": "file_filtered" if file_ids else "survey_wide"
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
            "dataSnapshot": {"totalResults": 0, "texts": [], "query": ""},
            "metadata": {
                "total_matches": 0,
                "processing_time": processing_time,
                "error": error or "No survey data found"
            }
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get optimized cache information from compressed survey cache service"""
        return {
            "survey_cache": compressed_survey_cache.get_performance_stats()
        }
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear optimized cache and return statistics"""
        old_stats = compressed_survey_cache.get_performance_stats()
        compressed_survey_cache.clear_cache()
        
        return {
            "surveys_cleared": old_stats.get("cache_size", 0)
        }


# Singleton instance
fast_search_service = FastSearchService()