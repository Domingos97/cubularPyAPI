"""
Lightweight Dependencies - No SQLAlchemy Overhead
===============================================
Simple dependency injection without ORM complexity.
Direct database queries for authentication.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.security import verify_token
from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Security
security = HTTPBearer()


class SimpleUser:
    """Simple user model without SQLAlchemy overhead"""
    
    def __init__(self, user_data: dict):
        self.id = user_data["id"]
        self.email = user_data["email"]
        self.role = user_data.get("role", "user")
        self.language = user_data.get("language", "en")
        self.is_active = user_data.get("is_active", True)


async def get_current_user_lightweight(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: LightweightDBService = Depends(get_lightweight_db)
) -> SimpleUser:
    """
    Get current authenticated user - lightweight version
    Direct database query, no ORM overhead
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify JWT token
        payload = verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Get user from database - direct SQL query
        user_data = await db.get_user_by_id(user_id)
        
        if user_data is None:
            raise credentials_exception
        
        if not user_data.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user"
            )
        
        return SimpleUser(user_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception


# Alias for compatibility with existing code
get_current_user = get_current_user_lightweight


async def get_current_admin_user(
    current_user: SimpleUser = Depends(get_current_user_lightweight)
) -> SimpleUser:
    """Get current user and verify admin role"""
    if current_user.role not in ["admin", "super_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_current_active_user(
    current_user: SimpleUser = Depends(get_current_user_lightweight)
) -> SimpleUser:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


# Optional authentication for public endpoints
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: LightweightDBService = Depends(get_lightweight_db)
) -> Optional[SimpleUser]:
    """
    Get current user optionally - for endpoints that work with or without auth
    """
    if credentials is None:
        return None
    
    try:
        payload = verify_token(credentials.credentials)
        if payload is None:
            return None
        
        user_id = payload.get("sub")
        if user_id is None:
            return None
        
        user_data = await db.get_user_by_id(user_id)
        
        if user_data is None or not user_data.get("is_active", True):
            return None
        
        return SimpleUser(user_data)
        
    except Exception as e:
        logger.warning(f"Optional auth failed: {e}")
        return None