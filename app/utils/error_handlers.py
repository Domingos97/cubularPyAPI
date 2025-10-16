"""
Standard Error Handlers
======================
Centralized error handling utilities for consistent API responses
"""

from typing import Dict, Any, Optional, List
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import traceback
import uuid

from app.models.schemas import ErrorResponse, ValidationErrorResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _safe_body_to_string(body: Any) -> Optional[str]:
    """Safely convert request body to string for logging"""
    if body is None:
        return None
    if isinstance(body, str):
        return body
    if isinstance(body, bytes):
        try:
            return body.decode('utf-8')
        except UnicodeDecodeError:
            return f"<binary data, {len(body)} bytes>"
    if isinstance(body, dict):
        import json
        try:
            return json.dumps(body)
        except (TypeError, ValueError):
            return str(body)
    return str(body)


class StandardErrorHandler:
    """Standard error handler for consistent API responses"""
    
    @staticmethod
    def create_error_response(
        status_code: int,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Create a standardized error response"""
        content = {
            "detail": message,
            "error_code": error_code or f"error_{status_code}",
            "timestamp": time.time(),
            "error_id": error_id or str(uuid.uuid4())
        }
        
        if details:
            content["details"] = details
            
        return JSONResponse(
            status_code=status_code,
            content=content
        )
    
    @staticmethod
    def create_validation_error_response(
        errors: List[Dict[str, Any]],
        error_id: Optional[str] = None
    ) -> JSONResponse:
        """Create a standardized validation error response"""
        content = {
            "detail": errors,
            "error_code": "validation_error",
            "timestamp": time.time(),
            "error_id": error_id or str(uuid.uuid4())
        }
        
        return JSONResponse(
            status_code=422,
            content=content
        )
    
    @staticmethod
    def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTPException with standard format"""
        return StandardErrorHandler.create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_code=f"http_{exc.status_code}"
        )
    
    @staticmethod
    def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle validation errors with standard format"""
        error_details = []
        for error in exc.errors():
            error_details.append({
                "loc": list(error["loc"]),
                "msg": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        return StandardErrorHandler.create_validation_error_response(error_details)
    
    @staticmethod
    async def log_error_to_database(
        request: Request,
        error_message: str,
        status_code: int,
        stack_trace: Optional[str] = None,
        error_id: Optional[str] = None
    ) -> str:
        """Log error to database and return error ID"""
        try:
            from app.services.lightweight_db_service import lightweight_db
            
            error_log_id = await lightweight_db.log_api_error(
                error_message=error_message,
                endpoint=str(request.url.path),
                method=request.method,
                status_code=status_code,
                user_id=getattr(request.state, 'user_id', None),
                request_body=_safe_body_to_string(getattr(request.state, 'body', None)),
                stack_trace=stack_trace
            )
            
            return error_log_id
        except Exception as db_error:
            logger.error(f"Failed to log error to database: {db_error}")
            return error_id or "db_logging_failed"


# Common HTTP status codes and their standard messages
HTTP_STATUS_MESSAGES = {
    400: "Bad Request",
    401: "Unauthorized", 
    403: "Forbidden",
    404: "Not Found",
    409: "Conflict",
    422: "Validation Error",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable"
}


def get_standard_error_message(status_code: int) -> str:
    """Get standard error message for HTTP status code"""
    return HTTP_STATUS_MESSAGES.get(status_code, "Unknown Error")


# Commonly used error responses
def unauthorized_error(message: str = "Authentication required") -> HTTPException:
    """Standard 401 Unauthorized error"""
    return HTTPException(status_code=401, detail=message)


def forbidden_error(message: str = "Access forbidden") -> HTTPException:
    """Standard 403 Forbidden error"""
    return HTTPException(status_code=403, detail=message)


def not_found_error(resource: str = "Resource") -> HTTPException:
    """Standard 404 Not Found error"""
    return HTTPException(status_code=404, detail=f"{resource} not found")


def conflict_error(message: str = "Resource conflict") -> HTTPException:
    """Standard 409 Conflict error"""
    return HTTPException(status_code=409, detail=message)


def validation_error(message: str = "Validation failed") -> HTTPException:
    """Standard 422 Validation error"""
    return HTTPException(status_code=422, detail=message)


def rate_limit_error(message: str = "Rate limit exceeded") -> HTTPException:
    """Standard 429 Rate Limit error"""
    return HTTPException(status_code=429, detail=message)


def internal_server_error(message: str = "Internal server error") -> HTTPException:
    """Standard 500 Internal Server error"""
    return HTTPException(status_code=500, detail=message)