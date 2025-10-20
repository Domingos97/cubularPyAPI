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
        # Convert UUID objects to strings to avoid asyncpg UUID object issues
        self.id = str(user_data["id"]) if user_data["id"] else None
        self.email = user_data["email"]
        self.username = user_data.get("username")
        self.role = user_data.get("role_name", user_data.get("role", "user"))
        self.language = user_data.get("language", "en")
        self.is_active = True  # Default to active since column doesn't exist in database
        self.email_confirmed = user_data.get("email_confirmed", False)
        self.welcome_popup_dismissed = user_data.get("welcome_popup_dismissed", False)
        self.last_login = user_data.get("last_login")
        self.created_at = user_data.get("created_at")
        self.updated_at = user_data.get("updated_at")
        self.preferred_personality = str(user_data.get("preferred_personality")) if user_data.get("preferred_personality") else None
        # New flag for AI personalities access
        self.has_ai_personalities_access = bool(user_data.get("has_ai_personalities_access", False))


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: LightweightDBService = Depends(get_lightweight_db),
    required_role: Optional[str] = None,
    require_active: bool = True
) -> SimpleUser:
    """
    SINGLE authentication function - handles all authentication and authorization
    
    Args:
        credentials: JWT token from request
        db: Database service
        required_role: If specified, user must have this role ('admin' or 'user')
        require_active: Whether user must be active
    
    Returns:
        Authenticated and authorized user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify JWT token
        logger.info(f"Verifying token: {credentials.credentials[:20]}...")
        payload = verify_token(credentials.credentials)
        if payload is None:
            logger.error("Token verification failed")
            raise credentials_exception
        
        user_id = payload.get("sub")
        if user_id is None:
            logger.error("No user_id (sub) in token payload")
            raise credentials_exception
        
        logger.info(f"Token verified for user_id: {user_id}")
        
        # Get user from database - direct SQL query
        user_data = await db.get_user_by_id(user_id)
        
        if user_data is None:
            logger.error(f"User not found in database: {user_id}")
            raise credentials_exception

        user = SimpleUser(user_data)
        logger.info(f"User loaded: {user.email}, role: {user.role}, is_active: {user.is_active}")
        
        # Check if user must be active
        if require_active and not user.is_active:
            logger.error(f"User {user.email} is inactive, denying access")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )
        
        # Check role if specified
        if required_role and user.role != required_role:
            logger.error(f"User {user.email} has role '{user.role}', required '{required_role}'")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {required_role}"
            )
        
        logger.info(f"Authentication successful for user: {user.email}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise credentials_exception


# ==========================================
# ROLE-SPECIFIC AUTHENTICATION FUNCTIONS
# ==========================================

async def get_current_admin_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: LightweightDBService = Depends(get_lightweight_db)
) -> SimpleUser:
    """Get current user - requires admin role"""
    return await get_current_user(credentials, db, required_role="admin")


async def get_current_regular_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: LightweightDBService = Depends(get_lightweight_db)
) -> SimpleUser:
    """Get current user - any authenticated user"""
    return await get_current_user(credentials, db, required_role=None)


# ==========================================
# COMMON QUERY PARAMETERS
# ==========================================

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