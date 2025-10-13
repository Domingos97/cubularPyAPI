from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any
import uuid

from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_admin_user
from app.models.schemas import (
    UserCreate, 
    UserResponse, 
    Token, 
    LoginRequest,
    RefreshTokenRequest,
    ResendConfirmationRequest,
    SuccessResponse,
    ErrorResponse
)
from app.models.models import User
from app.services.auth_service import auth_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Add explicit OPTIONS handlers to match TypeScript API
@router.options("/login")
@router.options("/register") 
@router.options("/refresh")
async def options_handler():
    """Handle OPTIONS requests for CORS preflight"""
    return JSONResponse(
        status_code=200,
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user
    
    - **email**: User's email address (must be unique)
    - **username**: User's display name
    - **password**: Password (minimum 6 characters)
    - **language**: User's preferred language (default: en)
    """
    try:
        user = await auth_service.create_user(db, user_data)
        
        logger.info(f"New user registered: {user.email}")
        
        return UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            language=user.language,
            email_confirmed=user.email_confirmed,
            welcome_popup_dismissed=user.welcome_popup_dismissed,
            last_login=user.last_login,
            role=user.role.role,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
        
    except ValueError as e:
        logger.warning(f"Registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login")
async def login(
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Login user and return access tokens
    
    - **email**: User's email address
    - **password**: User's password
    
    Returns access token and refresh token in camelCase format for frontend compatibility
    """
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(
            db, 
            login_data.email, 
            login_data.password
        )
        
        if not user:
            logger.warning(f"Failed login attempt for: {login_data.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        tokens = await auth_service.create_tokens(db, user)
        
        logger.info(f"User logged in: {user.email}")
        
        # Return in the exact format the frontend expects (camelCase)
        return JSONResponse(
            status_code=200,
            content=tokens  # tokens already in camelCase format from service
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh")
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token
    
    - **refresh_token**: Valid refresh token
    
    Returns new access token in camelCase format for frontend compatibility
    """
    try:
        tokens = await auth_service.refresh_access_token(
            db, 
            refresh_data.refresh_token
        )
        
        if not tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Return in the exact format the frontend expects (camelCase)
        return JSONResponse(
            status_code=200,
            content=tokens  # tokens already in camelCase format from service
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout", response_model=SuccessResponse)
async def logout(
    refresh_data: RefreshTokenRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout user by revoking refresh token
    
    - **refresh_token**: Refresh token to revoke
    """
    try:
        success = await auth_service.revoke_refresh_token(
            db, 
            refresh_data.refresh_token
        )
        
        if success:
            logger.info(f"User logged out: {current_user.email}")
            return SuccessResponse(message="Successfully logged out")
        else:
            return SuccessResponse(message="Token already revoked")
            
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/logout-all", response_model=SuccessResponse)
async def logout_all(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout from all devices by revoking all refresh tokens
    """
    try:
        count = await auth_service.revoke_all_user_tokens(db, current_user.id)
        
        logger.info(f"User logged out from all devices: {current_user.email}, tokens revoked: {count}")
        
        return SuccessResponse(
            message=f"Successfully logged out from all devices. {count} tokens revoked."
        )
        
    except Exception as e:
        logger.error(f"Logout all error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user information
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        language=current_user.language,
        email_confirmed=current_user.email_confirmed,
        welcome_popup_dismissed=current_user.welcome_popup_dismissed,
        last_login=current_user.last_login,
        role=current_user.role.role,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )


@router.get("/confirm-email")
async def confirm_email(
    token: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Confirm user email with token
    
    - **token**: Email confirmation token
    """
    try:
        user = await auth_service.confirm_email(db, token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired confirmation token"
            )
        
        logger.info(f"Email confirmed for user: {user.email}")
        
        return SuccessResponse(message="Email confirmed successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email confirmation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email confirmation failed"
        )


@router.post("/resend-confirmation", response_model=SuccessResponse)
async def resend_confirmation_email(
    request_data: ResendConfirmationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Resend confirmation email
    
    - **email**: Email address to resend confirmation to
    """
    try:
        user = await auth_service.resend_confirmation_email(
            db, 
            request_data.email
        )
        
        if not user:
            # For security, don't reveal if email exists or not
            return SuccessResponse(
                message="If the email exists and is not confirmed, a new confirmation email has been sent"
            )
        
        if user.email_confirmed:
            return SuccessResponse(message="Email is already confirmed")
        
        logger.info(f"Confirmation email resent to: {request_data.email}")
        
        return SuccessResponse(
            message="Confirmation email has been resent"
        )
        
    except Exception as e:
        logger.error(f"Resend confirmation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resend confirmation email"
        )


@router.get("/is-admin/{user_id}")
async def check_admin_status(
    user_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Check if specified user is admin (requires authentication)
    
    - **user_id**: User ID to check admin status for
    """
    # For security, only allow admins to check other users' admin status
    # or users to check their own status
    if current_user.id != user_id and current_user.role.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to check this user's admin status"
        )
    
    # Get the target user
    if current_user.id == user_id:
        target_user = current_user
    else:
        target_user = await auth_service.get_user_by_id(db, user_id)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    
    return {
        "is_admin": target_user.role.role == "admin",
        "role": target_user.role.role,
        "user_id": str(target_user.id)
    }


@router.post("/upgrade-to-admin", response_model=SuccessResponse)
async def upgrade_to_admin(
    email: str,
    admin_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upgrade a user to admin role (admin only)
    
    - **email**: Email of user to upgrade
    """
    try:
        from sqlalchemy import select
        from app.models.models import Role
        
        # Get the user to upgrade
        user = await auth_service.get_user_by_email(db, email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get admin role
        admin_role_query = select(Role).where(Role.role == "admin")
        admin_role_result = await db.execute(admin_role_query)
        admin_role = admin_role_result.scalar_one_or_none()
        
        if not admin_role:
            # Create admin role if it doesn't exist
            admin_role = Role(role="admin")
            db.add(admin_role)
            await db.commit()
            await db.refresh(admin_role)
        
        # Update user role
        user.roleid = admin_role.id
        await db.commit()
        
        logger.info(f"User upgraded to admin: {user.email} by {admin_user.email}")
        
        return SuccessResponse(message=f"User {email} upgraded to admin successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin upgrade error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin upgrade failed"
        )


@router.post("/change-password", response_model=SuccessResponse)
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change user password
    
    - **current_password**: Current password
    - **new_password**: New password (minimum 6 characters)
    """
    try:
        from app.core.security import verify_password
        
        # Verify current password
        if not verify_password(current_password, current_user.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        if len(new_password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 6 characters long"
            )
        
        # Update password
        user = await auth_service.update_user_password(
            db, 
            current_user.id, 
            new_password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Password changed for user: {current_user.email}")
        
        return SuccessResponse(message="Password changed successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )