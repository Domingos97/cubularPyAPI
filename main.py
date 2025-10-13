from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import time
import uuid

from app.core.config import settings
from app.api.v1.api import api_router
# LEGACY MIDDLEWARE REMOVED FOR PERFORMANCE
# from app.middleware.loggingMiddleware import LoggingMiddleware  # DISABLED - Heavy overhead
# from app.middleware.rateLimitMiddleware import RateLimitMiddleware  # DISABLED - Heavy overhead
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Lifespan event handler with lightweight initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting OPTIMIZED {settings.app_name} v{settings.app_version}")
    
    # Initialize lightweight database service
    try:
        from app.services.lightweight_db_service import lightweight_db
        await lightweight_db.initialize()
        logger.info("Lightweight DB service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # Log configuration
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Database: {settings.database_host}:{settings.database_port}/{settings.database_name}")
    
    # Check AI provider configuration
    providers = []
    if settings.openai_api_key:
        providers.append("OpenAI")
    if settings.anthropic_api_key:
        providers.append("Anthropic")
    
    if providers:
        logger.info(f"AI providers configured: {', '.join(providers)}")
    else:
        logger.warning("No AI providers configured - chat functionality will be limited")
    
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
    redoc_url=None,
    lifespan=lifespan
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
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# PERFORMANCE OPTIMIZATION: Removed heavy middleware
# - LoggingMiddleware (UUID generation, extensive logging)
# - RateLimitMiddleware (complex in-memory tracking) 
# - TrustedHostMiddleware (host validation overhead)
# Only essential CORS middleware remains for production compatibility

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "error_code": "validation_error",
            "timestamp": time.time()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_code": "internal_error",
            "timestamp": time.time(),
            "request_id": str(uuid.uuid4())
        }
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

# Include API routes
app.include_router(api_router, prefix="/api")

# Additional utility endpoints
@app.get("/api/info", tags=["info"])
async def api_info():
    """
    Get API information and capabilities.
    """
    providers = []
    if settings.openai_api_key:
        providers.append("openai")
    if settings.anthropic_api_key:
        providers.append("anthropic")
    
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
        "ai_providers": providers,
        "supported_file_types": ["csv", "xlsx", "xls"],
        "max_file_size_mb": settings.max_file_size // (1024 * 1024),
        "endpoints": {
            "auth": "/api/auth",
            "users": "/api/users", 
            "surveys": "/api/surveys",
            "fast_search": "/api/fast-search",  # Updated from vector_search
            "chat": "/api/chat"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )