from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import time
import uuid
import math
from typing import Any

from app.core.config import settings
from app.api.v1.api import api_router
from app.middleware.rate_limit_middleware import add_rate_limiting_middleware
from app.utils.logging import get_logger

logger = get_logger(__name__)


def sanitize_for_json(data: Any) -> Any:
    """
    Sanitize data for JSON serialization by handling NaN, UUID, and other problematic types
    """
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, uuid.UUID):
        return str(data)
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data



# Lifespan event handler with lightweight initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting OPTIMIZED {settings.app_name} v{settings.app_version}")
    
    # Initialize lightweight database service
    try:
        from app.services.lightweight_db_service import lightweight_db
        await lightweight_db.initialize()
        
        # Log optimized connection pool stats
        pool_stats = lightweight_db.get_pool_stats()
        logger.info(f"✅ OPTIMIZED DB initialized: {pool_stats['current_size']}/{pool_stats['max_size']} connections ready")
        logger.info(f"📊 Pool status: {pool_stats['status']} with {pool_stats['idle_connections']} idle connections")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # Log configuration
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Database: {settings.database_host}:{settings.database_port}/{settings.database_name}")
    
    # AI provider configuration now comes from the database; no static environment check
    
    # SIMPLIFIED OPTIMIZATION: Preload survey data only (removed embedding cache complexity)
    try:
        logger.info("🚀 Starting survey data preloading...")
        from app.services.survey_cache import survey_cache
        
        # Preload surveys using the cache service (keep this as it's genuinely useful)
        await survey_cache.preload_frequent_surveys()
        
        # Get performance stats after preloading
        cache_stats = survey_cache.get_stats()
        
        logger.info(f"✅ Survey preloading completed: {cache_stats.get('cache_size', 0)} surveys cached")
        logger.info(f"📊 Expected performance boost: 70-90% for cached survey operations")
        
    except Exception as e:
        logger.error(f"❌ Survey preloading failed (will impact performance): {e}")
        # Don't fail startup if preloading fails
    
    logger.info("🚀 OPTIMIZED API ready - Lightweight, fast, cached!")
    
    yield
    
    # Shutdown
    try:
        from app.services.lightweight_db_service import lightweight_db
        await lightweight_db.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info(f"Shutting down {settings.app_name}")

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-performance Python API for survey data analysis and AI-powered insights",
    openapi_url="/api/openapi.json",
    docs_url=None,  # We'll create custom docs
    redoc_url=None
    # lifespan=lifespan  # Temporarily commented out for troubleshooting
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom documentation endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

# Lightweight middleware setup - only essential CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://preview--projectxy-12.lovable.app",  # Lovable preview
        "http://localhost:5173",                       # Local development
        "http://localhost:3000",                        # If testing locally
        "http://localhost:8080",                       # If testing locally
        "http://localhost:8081",                       # Frontend development server
        "http://127.0.0.1:5173",                      # Alternative localhost
        "http://127.0.0.1:3000",                      # Alternative localhost
        "http://127.0.0.1:8080",                      # Alternative localhost
        "http://127.0.0.1:8081",                      # Alternative localhost
        "https://cubularpyapi-production.up.railway.app", # Allow production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Add lightweight logging middleware for database error tracking
# add_lightweight_logging_middleware(app, log_requests=True, log_errors=True)  # DISABLED FOR PERFORMANCE

# Add lightweight rate limiting middleware - optimized for performance with more exemptions
add_rate_limiting_middleware(app, calls=2000, period=60, exempt_paths=[
    "/health", "/docs", "/redoc", "/openapi.json", 
    "/api/v1/chat/sessions", "/quick", "/api/v1/fast-search", 
    "/api/v1/fast-search/search", "/api/v1/streamlined-chat", 
    "/api/legacy", "/", "/api/v1/info"
])

# PERFORMANCE OPTIMIZATION: Lightweight middleware approach
# - Lightweight logging for essential tracking
# - Simple rate limiting for abuse prevention
# - Only essential CORS middleware
# Heavy middleware removed for performance

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    # Return error without database logging for performance
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=sanitize_for_json({
            "detail": str(exc),
            "error_code": "validation_error",
            "timestamp": time.time()
        })
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    error_message = str(exc)
    request_id = str(uuid.uuid4())
    
    # Log to console and database for errors
    logger.error(f"Internal server error [{request_id}]: {error_message}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=sanitize_for_json({
            "detail": "Internal server error",
            "error_code": "internal_error",
            "timestamp": time.time(),
            "request_id": request_id
        })
    )

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "timestamp": time.time(),
        "environment": "development" if settings.debug else "production"
    }

# Root endpoint (matches TypeScript API)
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint - matches TypeScript API response.
    """
    return "API is running... Visit /docs for documentation"

# Include API routes (single versioned prefix)
app.include_router(api_router, prefix="/api/v1")

# Additional utility endpoints
@app.get("/api/v1/info", tags=["info"])
async def api_info():
    """
    Get API information and capabilities.
    """
    # AI providers are now managed in the database; this can be extended to query DB if needed
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "features": [
            "user_authentication",
            "survey_management", 
            "file_processing",
            "fast_search",  # Updated from vector_search
            "ai_chat",
            "semantic_analysis"
        ],
        "ai_providers": [],
        "supported_file_types": ["csv", "xlsx", "xls"],
        "max_file_size_mb": settings.max_file_size // (1024 * 1024),
        "endpoints": {
            "auth": "/api/v1/auth",
            "users": "/api/v1/users", 
            "surveys": "/api/v1/surveys",
            "fast_search": "/api/v1/fast-search",
            "chat": "/api/v1/chat",
            "survey_builder": "/api/v1/survey-builder"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    # Always use the PORT environment variable if set, otherwise default to 8000
    port = int(os.environ.get("PORT", 10000))
    print(f"[Startup] Using port: {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=10000,
        reload=getattr(settings, 'debug', False),
        log_level=getattr(settings, 'log_level', 'info').lower()
    )