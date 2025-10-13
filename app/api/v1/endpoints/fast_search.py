"""
Fast Search Router - Direct File-Based Search
============================================
Ultra-fast search endpoint that mimics the original Python API approach.
No database operations, just direct pickle file access.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time

from app.services.fast_search_service import fast_search_service
from app.services.embedding_service import embedding_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class FastSearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    surveys: List[str] = Field(..., description="List of survey IDs to search")
    threshold: Optional[float] = Field(0.25, description="Similarity threshold")
    max_results: Optional[int] = Field(1000, description="Maximum results")


class FastSearchResponse(BaseModel):
    responses: List[Dict[str, Any]]
    demographics: Dict[str, List]
    psychology: Dict[str, float]
    phrases: Dict[str, List]
    dataSnapshot: Dict[str, Any]
    metadata: Dict[str, Any]


@router.post("/search", response_model=FastSearchResponse)
async def fast_search(request: FastSearchRequest):
    """
    Ultra-fast search endpoint - mimics original TypeScript->Python API flow
    """
    start_time = time.time()
    
    try:
        logger.info(f"Fast search request: '{request.query[:50]}...' in {len(request.surveys)} surveys")
        
        # Generate embedding for query
        query_embedding = await embedding_service.generate_embedding(request.query)
        if not query_embedding:
            raise HTTPException(status_code=400, detail="Failed to generate query embedding")
        
        # Perform ultra-fast search
        results = await fast_search_service.fast_search(
            query_embedding=query_embedding,
            survey_ids=request.surveys,
            threshold=request.threshold,
            max_results=request.max_results,
            query_text=request.query
        )
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Fast search completed in {total_time:.1f}ms: {len(results.get('responses', []))} results")
        
        return results
        
    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        logger.error(f"Fast search error after {total_time:.1f}ms: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


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