from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select

from app.core.config import settings
from app.core.security import verify_token
from app.models.models import User, Role
from app.core.database import get_db

# Security
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
    except Exception:
        raise credentials_exception
    
    # Get user from database with role information
    try:
        query = (
            select(User)
            .options(selectinload(User.role))
            .where(User.id == user_id)
        )
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if user is None:
            raise credentials_exception
            
        return user
        
    except Exception:
        raise credentials_exception
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    # Add any additional checks for user status if needed
    return current_user


async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current admin user"""
    if current_user.role.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin access required."
        )
    return current_user


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    if not credentials:
        return None
    
    try:
        payload = verify_token(credentials.credentials)
        if payload is None:
            return None
        
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        # This would need to be async, but we can't make this dependency async
        # For now, return None and handle authentication in the endpoint
        return None
        
    except Exception:
        return None


class CommonQueryParams:
    """Common query parameters for list endpoints"""
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ):
        self.skip = skip
        self.limit = min(limit, 1000)  # Max 1000 items per request
        self.search = search
        self.sort_by = sort_by
        self.sort_order = sort_order.lower() if sort_order.lower() in ["asc", "desc"] else "asc"


def common_parameters(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "asc"
) -> CommonQueryParams:
    """Dependency for common query parameters"""
    return CommonQueryParams(skip, limit, search, sort_by, sort_order)


# Rate limiting (placeholder for future implementation)
class RateLimiter:
    """Rate limiting dependency"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def __call__(self, request):
        # Placeholder for rate limiting logic
        # In a production environment, you would implement Redis-based rate limiting
        pass


def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Rate limiting dependency factory"""
    return RateLimiter(max_requests, window_seconds)


# Validation dependencies
def validate_uuid(uuid_str: str) -> str:
    """Validate UUID format"""
    import uuid
    try:
        uuid.UUID(uuid_str)
        return uuid_str
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format"
        )


def validate_file_size(max_size: int = None):
    """Validate file size dependency factory"""
    if max_size is None:
        max_size = settings.max_file_size
    
    def validator(file):
        if hasattr(file, 'size') and file.size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {max_size} bytes"
            )
        return file
    
    return validator


def validate_content_type(allowed_types: list):
    """Validate content type dependency factory"""
    def validator(file):
        if hasattr(file, 'content_type') and file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
            )
        return file
    
    return validator


# Logging dependencies
async def log_request_info(request) -> dict:
    """Extract request information for logging"""
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent")
    }