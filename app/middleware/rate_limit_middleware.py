"""
Lightweight Rate Limiting Middleware
===================================
Simple in-memory rate limiting to prevent API abuse
"""

import time
from collections import defaultdict, deque
from typing import Dict, Tuple
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.utils.logging import get_logger

logger = get_logger(__name__)


class LightweightRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Lightweight rate limiting middleware using sliding window approach
    Stores rate limit data in memory (not persistent across restarts)
    """
    
    def __init__(
        self, 
        app,
        calls: int = 100,  # Number of calls allowed
        period: int = 60,  # Time period in seconds
        exempt_paths: list = None,  # Paths to exempt from rate limiting
        get_client_ip=None  # Function to extract client IP
    ):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
        self.get_client_ip = get_client_ip or self._default_get_client_ip
        
        # In-memory storage: {client_ip: deque of request timestamps}
        self.clients: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Cleanup old entries every 5 minutes
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes
    
    def _default_get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check X-Forwarded-For header (common with reverse proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from rate limiting"""
        return any(exempt in path for exempt in self.exempt_paths)
    
    def _cleanup_old_entries(self):
        """Remove old timestamps beyond the rate limit period"""
        current_time = time.time()
        
        # Only cleanup every 5 minutes to avoid performance impact
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - self.period
        
        # Clean up old timestamps for all clients
        for client_ip in list(self.clients.keys()):
            timestamps = self.clients[client_ip]
            
            # Remove timestamps older than the period
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()
            
            # Remove clients with no recent requests
            if not timestamps:
                del self.clients[client_ip]
        
        self.last_cleanup = current_time
    
    def _is_rate_limited(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if client is rate limited
        Returns: (is_limited, requests_remaining)
        """
        current_time = time.time()
        cutoff_time = current_time - self.period
        
        # Get client's request timestamps
        timestamps = self.clients[client_ip]
        
        # Remove old timestamps
        while timestamps and timestamps[0] < cutoff_time:
            timestamps.popleft()
        
        # Count current requests in the time window
        current_requests = len(timestamps)
        
        if current_requests >= self.calls:
            return True, 0
        
        # Add current request timestamp
        timestamps.append(current_time)
        
        requests_remaining = self.calls - (current_requests + 1)
        return False, requests_remaining
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for exempt paths
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Periodic cleanup
        self._cleanup_old_entries()
        
        # Get client IP
        client_ip = self.get_client_ip(request)
        
        # Check rate limit
        is_limited, requests_remaining = self._is_rate_limited(client_ip)
        
        if is_limited:
            # Log rate limit violation
            logger.warning(f"Rate limit exceeded for IP {client_ip} on {request.url.path}")
            
            # Return rate limit error
            return Response(
                status_code=429,
                content='{"detail": "Rate limit exceeded", "error_code": "rate_limit_exceeded"}',
                headers={
                    "Content-Type": "application/json",
                    "X-RateLimit-Limit": str(self.calls),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + self.period)),
                    "Retry-After": str(self.period)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(requests_remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + self.period))
        
        return response


def add_rate_limiting_middleware(
    app, 
    calls: int = 100, 
    period: int = 60,
    exempt_paths: list = None
):
    """
    Add rate limiting middleware to FastAPI app
    
    Args:
        app: FastAPI application instance
        calls: Number of calls allowed per period
        period: Time period in seconds
        exempt_paths: List of paths to exempt from rate limiting
    """
    app.add_middleware(
        LightweightRateLimitMiddleware, 
        calls=calls, 
        period=period, 
        exempt_paths=exempt_paths
    )
    
    logger.info(f"Rate limiting enabled: {calls} calls per {period} seconds")