"""
Lightweight Logging Middleware
=============================
Minimal overhead middleware for logging API requests and responses to database.
Replaces the heavy middleware that was disabled for performance.
"""

import asyncio
import time
import traceback
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.lightweight_db_service import lightweight_db
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LightweightLoggingMiddleware(BaseHTTPMiddleware):
    """
    Lightweight middleware for logging requests and responses
    Only logs essential information with minimal performance impact
    """
    
    def __init__(self, app, log_requests: bool = True, log_errors: bool = True):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_errors = log_errors
    
    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()
        
        # Get user ID if available (for authenticated requests)
        user_id = None
        try:
            # Try to extract user ID from request state (set by auth middleware)
            user_id = getattr(request.state, 'user_id', None)
        except AttributeError:
            pass
        
        # Store request body for error logging if needed
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Store body in request state for error logging
                    request.state.body = body
                    # For logging, we'll just note that there was a body (not log actual content for performance)
                    import json
                    request_body = json.dumps({"has_body": True, "size": len(body)})
            except Exception:
                pass
        
        # Process request
        try:
            response = await call_next(request)
            response_time = int((time.time() - start_time) * 1000)  # milliseconds
            
            # Log successful requests (async, non-blocking)
            if self.log_requests:
                asyncio.create_task(
                    self._log_request(
                        method=request.method,
                        endpoint=str(request.url.path),
                        status_code=response.status_code,
                        response_time=response_time,
                        user_id=user_id,
                        request_body=request_body
                    )
                )
            
            return response
            
        except Exception as exc:
            response_time = int((time.time() - start_time) * 1000)
            
            # Log errors (async, non-blocking)
            if self.log_errors:
                asyncio.create_task(
                    self._log_error(
                        method=request.method,
                        endpoint=str(request.url.path),
                        error_message=str(exc),
                        response_time=response_time,
                        user_id=user_id,
                        request_body=request_body,
                        stack_trace=traceback.format_exc()
                    )
                )
            
            # Re-raise the exception to be handled by exception handlers
            raise exc
    
    async def _log_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        response_time: int,
        user_id: Optional[str] = None,
        request_body: Optional[dict] = None
    ):
        """Log successful request to database (async, non-blocking)"""
        try:
            # Only log errors and warnings, not all requests (for performance)
            if status_code >= 400:
                await lightweight_db.log_api_error(
                    error_message=f"HTTP {status_code} response",
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    user_id=user_id,
                    request_body=request_body
                )
        except Exception as e:
            # Don't log database errors to avoid infinite loops
            logger.warning(f"Failed to log request to database: {e}")
    
    async def _log_error(
        self,
        method: str,
        endpoint: str,
        error_message: str,
        response_time: int,
        user_id: Optional[str] = None,
        request_body: Optional[dict] = None,
        stack_trace: Optional[str] = None
    ):
        """Log error to database (async, non-blocking)"""
        try:
            await lightweight_db.log_api_error(
                error_message=error_message,
                endpoint=endpoint,
                method=method,
                status_code=500,  # Default for unhandled exceptions
                user_id=user_id,
                request_body=request_body,
                stack_trace=stack_trace
            )
        except Exception as e:
            # Don't log database errors to avoid infinite loops
            logger.error(f"Failed to log error to database: {e}")
            logger.error(f"Original error: {error_message}")


# Helper function to add middleware to app
def add_lightweight_logging_middleware(app, log_requests: bool = True, log_errors: bool = True):
    """
    Add lightweight logging middleware to FastAPI app
    
    Args:
        app: FastAPI application instance
        log_requests: Whether to log HTTP requests (only errors and warnings)
        log_errors: Whether to log application errors
    """
    app.add_middleware(
        LightweightLoggingMiddleware,
        log_requests=log_requests,
        log_errors=log_errors
    )