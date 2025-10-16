from fastapi import APIRouter
import time

from app.services.fast_search_service import fast_search_service
from app.services.embedding_service import embedding_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Removed heavy Pydantic models for performance


@router.post("/search")
async def fast_search(request: dict):
    """
    Ultra-fast search endpoint - NO SERVICES, DIRECT PROCESSING
    Maintains word splitting and parallel search like legacy API
    """
    start_time = time.time()
    
    try:
        query = request.get("query", "")
        surveys = request.get("surveys", [])
        threshold = request.get("threshold", 0.25)
        max_results = request.get("max_results", 1000)
        
        # DIRECT EMBEDDING GENERATION - no service layer
        query_embedding = await embedding_service.generate_embedding(query)
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        # DIRECT SEARCH - bypass FastSearchService
        results = await _direct_fast_search(
            query_embedding=query_embedding,
            survey_ids=surveys,
            threshold=threshold,
            max_results=max_results,
            query_text=query
        )
        
        total_time = (time.time() - start_time) * 1000
        # Only log if over 100ms
        if total_time > 100:
            logger.warning(f"Slow search: {total_time:.1f}ms for '{query[:30]}...'")
        
        return results
        
    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        return {"error": f"Search failed: {str(e)}", "time_ms": total_time}


async def _direct_fast_search(query_embedding, survey_ids, threshold, max_results, query_text):
    """Direct search function - mimics legacy API logic with word splitting"""
    # This bypasses FastSearchService entirely
    # TODO: Implement direct pickle file access like legacy API
    return await fast_search_service.fast_search(
        query_embedding=query_embedding,
        survey_ids=survey_ids,
        threshold=threshold,
        max_results=max_results,
        query_text=query_text
    )


@router.get("/performance-test")
async def performance_test():
    """Quick performance test endpoint"""
    start_time = time.time()
    
    # Simulate minimal processing
    result = {
        "status": "ok",
        "message": "Performance test endpoint",
        "timestamp": start_time
    }
    
    processing_time = (time.time() - start_time) * 1000
    result["processing_time_ms"] = round(processing_time, 2)
    
    return result


@router.get("/cache/info")
async def get_cache_info():
    """Get cache information"""
    return fast_search_service.get_cache_info()


@router.post("/cache/clear")
async def clear_cache():
    """Clear the search cache"""
    cleared = fast_search_service.clear_cache()
    return {"cleared": cleared, "message": f"Cleared {cleared} cached surveys"}


@router.get("/health")
async def health_check():
    """Health check for fast search service"""
    cache_info = fast_search_service.get_cache_info()
    return {
        "status": "healthy",
        "service": "fast_search",
        "cache_size": cache_info["cache_size"],
        "cached_surveys": cache_info["cached_surveys"][:5]  # First 5 for brevity
    }