from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid

from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_admin_user
from app.models.schemas import (
    LLMSettingCreate,
    LLMSettingUpdate,
    LLMSettingResponse,
    SuccessResponse,
    ErrorResponse
)
from app.models.models import User
from app.services.llm_settings_service import llm_settings_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[LLMSettingResponse])
async def get_llm_settings(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get all LLM settings (Admin only)
    
    - **provider**: Optional filter by provider (openai, anthropic, google, azure, etc.)
    - **skip**: Number of settings to skip (for pagination)
    - **limit**: Maximum number of settings to return
    
    Returns list of LLM settings (API keys are masked for security)
    """
    try:
        if provider:
            settings = await llm_settings_service.get_settings_by_provider(db, provider)
            return [settings] if settings else []
        else:
            settings = await llm_settings_service.get_all_settings(
                db, skip=skip, limit=limit
            )
            return settings
        
    except Exception as e:
        logger.error(f"Error retrieving LLM settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve LLM settings"
        )


@router.get("/{setting_id}", response_model=LLMSettingResponse)
async def get_llm_setting_by_id(
    setting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get specific LLM setting by ID (Admin only)
    
    - **setting_id**: UUID of the LLM setting to retrieve
    
    Returns the LLM setting details (API key is masked)
    """
    try:
        setting = await llm_settings_service.get_setting_by_id(db, setting_id)
        
        if not setting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LLM setting not found"
            )
        
        logger.info(f"Retrieved LLM setting {setting_id} for admin {current_user.id}")
        return setting
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving LLM setting {setting_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve LLM setting"
        )


@router.post("/", response_model=LLMSettingResponse, status_code=status.HTTP_201_CREATED)
async def create_llm_setting(
    setting_data: LLMSettingCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Create a new LLM setting (Admin only)
    
    - **provider**: LLM provider name (openai, anthropic, google, azure, etc.)
    - **api_key**: API key for the provider (will be encrypted)
    
    Returns the created LLM setting
    """
    try:
        setting = await llm_settings_service.create_setting(
            db, setting_data, created_by=current_user.id
        )
        
        logger.info(f"Created LLM setting {setting.id} for provider {setting.provider} by admin {current_user.id}")
        return setting
        
    except ValueError as e:
        logger.warning(f"Invalid data for LLM setting creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating LLM setting: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create LLM setting"
        )


@router.put("/{setting_id}", response_model=LLMSettingResponse)
async def update_llm_setting(
    setting_id: uuid.UUID,
    setting_data: LLMSettingUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Update an existing LLM setting (Admin only)
    
    - **setting_id**: UUID of the LLM setting to update
    - **api_key**: Optional new API key (will be encrypted)
    - **active**: Optional active status
    
    Returns the updated LLM setting
    """
    try:
        setting = await llm_settings_service.update_setting(
            db, setting_id, setting_data
        )
        
        if not setting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LLM setting not found"
            )
        
        logger.info(f"Updated LLM setting {setting_id} by admin {current_user.id}")
        return setting
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid data for LLM setting update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating LLM setting {setting_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update LLM setting"
        )


@router.post("/upsert", response_model=LLMSettingResponse)
async def upsert_llm_setting(
    setting_data: LLMSettingCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Create or update LLM setting based on provider (Admin only)
    
    - **provider**: LLM provider name
    - **api_key**: API key for the provider
    
    If a setting for this provider exists, it will be updated.
    Otherwise, a new setting will be created.
    """
    try:
        setting = await llm_settings_service.upsert_setting(
            db, setting_data, created_by=current_user.id
        )
        
        logger.info(f"Upserted LLM setting for provider {setting.provider} by admin {current_user.id}")
        return setting
        
    except ValueError as e:
        logger.warning(f"Invalid data for LLM setting upsert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error upserting LLM setting: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create/update LLM setting"
        )


@router.delete("/{setting_id}", response_model=SuccessResponse)
async def delete_llm_setting(
    setting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Delete an LLM setting (Admin only)
    
    - **setting_id**: UUID of the LLM setting to delete
    
    Returns success confirmation
    """
    try:
        success = await llm_settings_service.delete_setting(db, setting_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LLM setting not found"
            )
        
        logger.info(f"Deleted LLM setting {setting_id} by admin {current_user.id}")
        return SuccessResponse(
            message=f"LLM setting {setting_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting LLM setting {setting_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete LLM setting"
        )


@router.get("/{setting_id}/decrypted-api-key")
async def get_decrypted_api_key(
    setting_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get decrypted API key for a specific setting (Admin only)
    
    - **setting_id**: UUID of the LLM setting
    
    Returns the decrypted API key (use with caution)
    """
    try:
        api_key = await llm_settings_service.get_decrypted_api_key_by_id(
            db, setting_id
        )
        
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LLM setting not found or no API key configured"
            )
        
        logger.warning(f"Decrypted API key accessed for setting {setting_id} by admin {current_user.id}")
        return {"api_key": api_key}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decrypted API key for setting {setting_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API key"
        )


@router.get("/provider/{provider}/decrypted-api-key")
async def get_decrypted_api_key_by_provider(
    provider: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get decrypted API key for a specific provider (Admin only)
    
    - **provider**: Provider name (openai, anthropic, etc.)
    
    Returns the decrypted API key for the active setting of this provider
    """
    try:
        api_key = await llm_settings_service.get_decrypted_api_key_by_provider(
            db, provider
        )
        
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active LLM setting found for provider {provider} or no API key configured"
            )
        
        logger.warning(f"Decrypted API key accessed for provider {provider} by admin {current_user.id}")
        return {"api_key": api_key}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting decrypted API key for provider {provider}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API key"
        )


@router.get("/active/list", response_model=List[LLMSettingResponse])
async def get_active_llm_settings(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get only active LLM settings (for dropdowns, etc.)
    
    Returns list of active LLM settings with masked API keys
    """
    try:
        settings = await llm_settings_service.get_active_settings(db)
        
        logger.info(f"Retrieved {len(settings)} active LLM settings for user {current_user.id}")
        return settings
        
    except Exception as e:
        logger.error(f"Error retrieving active LLM settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active LLM settings"
        )