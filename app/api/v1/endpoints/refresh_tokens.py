from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user, get_current_admin_user, get_db
from app.models.models import User
from app.models.schemas import (
    RefreshTokenRequest,
    Token,
    SuccessResponse
)
from app.services.refresh_token_service import RefreshTokenService

router = APIRouter()

@router.post("/refresh", response_model=Token, tags=["auth"])
async def refresh_access_token(
    token_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token
    """
    try:
        result = await RefreshTokenService.refresh_access_token(
            db, token_data.refresh_token
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        return Token(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh token: {str(e)}"
        )

@router.post("/revoke", response_model=SuccessResponse, tags=["auth"])
async def revoke_refresh_token(
    token_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Revoke a specific refresh token
    """
    try:
        success = await RefreshTokenService.revoke_refresh_token(
            db, token_data.refresh_token
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Refresh token not found"
            )
        
        return SuccessResponse(message="Refresh token revoked successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke token: {str(e)}"
        )

@router.post("/revoke-all", response_model=SuccessResponse, tags=["auth"])
async def revoke_all_user_tokens(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Revoke all refresh tokens for the current user (logout all devices)
    """
    try:
        success = await RefreshTokenService.revoke_all_user_tokens(
            db, str(current_user.id)
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to revoke tokens"
            )
        
        return SuccessResponse(message="All refresh tokens revoked successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke all tokens: {str(e)}"
        )

@router.post("/cleanup", response_model=Dict[str, Any], tags=["auth"])
async def cleanup_expired_tokens(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Clean up expired refresh tokens (admin only)
    """
    try:
        deleted_count = await RefreshTokenService.cleanup_expired_tokens(db)
        
        return {
            "message": f"Cleaned up {deleted_count} expired refresh tokens",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup tokens: {str(e)}"
        )

@router.get("/user/{user_id}/count", response_model=Dict[str, Any], tags=["auth"])
async def get_user_token_count(
    user_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get count of active refresh tokens for a user (admin only)
    """
    try:
        count = await RefreshTokenService.get_user_refresh_tokens_count(db, user_id)
        
        return {
            "user_id": user_id,
            "active_tokens": count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get token count: {str(e)}"
        )

@router.get("/user/{user_id}/tokens", response_model=List[Dict[str, Any]], tags=["auth"])
async def get_user_token_info(
    user_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get refresh token information for a user (admin only)
    """
    try:
        token_info = await RefreshTokenService.get_refresh_token_info(db, user_id)
        return token_info
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get token info: {str(e)}"
        )

@router.get("/my-tokens/count", response_model=Dict[str, Any], tags=["auth"])
async def get_my_token_count(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get count of active refresh tokens for the current user
    """
    try:
        count = await RefreshTokenService.get_user_refresh_tokens_count(
            db, str(current_user.id)
        )
        
        return {
            "user_id": str(current_user.id),
            "active_tokens": count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get token count: {str(e)}"
        )