from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
import uuid

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import  get_current_user, get_current_admin_user, SimpleUser
from app.models.schemas import (
    LLMSettingCreate,
    LLMSettingUpdate,
    LLMSettingResponse,
    SuccessResponse
)
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[LLMSettingResponse])
async def get_llm_settings(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    skip: int = 0,
    limit: int = 100,
    db: LightweightDBService = Depends(get_lightweight_db)
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
            # Get all settings and filter by provider (simplified approach)
            all_settings = await db.get_all_llm_settings(skip=0, limit=1000)
            settings_data = [s for s in all_settings if s.get("provider") == provider]
        else:
            settings_data = await db.get_all_llm_settings(skip=skip, limit=limit)
        
        # Convert to response format
        settings = [
            LLMSettingResponse(
                id=str(s["id"]),
                provider=s["provider"],
                active=s.get("active", True),
                api_key_configured=bool(s.get("api_key")),
                created_by=str(s["created_by"]) if s.get("created_by") else None,
                created_at=s["created_at"] if s.get("created_at") else None,
                updated_at=s["updated_at"] if s.get("updated_at") else None
            )
            for s in settings_data
        ]
        
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get specific LLM setting by ID (Admin only)
    
    - **setting_id**: UUID of the LLM setting to retrieve
    
    Returns the LLM setting details (API key is masked)
    """
    try:
        setting_data = await db.get_llm_setting_by_id(str(setting_id))
        
        if not setting_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LLM setting not found"
            )
        
        setting = LLMSettingResponse(
            id=str(setting_data["id"]),
            provider=setting_data["provider"],
            active=setting_data.get("active", True),
            api_key_configured=bool(setting_data.get("api_key")),
            created_by=str(setting_data["created_by"]) if setting_data.get("created_by") else None,
            created_at=setting_data["created_at"] if setting_data.get("created_at") else None,
            updated_at=setting_data["updated_at"] if setting_data.get("updated_at") else None
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Create a new LLM setting (Admin only)
    
    - **provider**: LLM provider name (openai, anthropic, google, azure, etc.)
    - **api_key**: API key for the provider (will be encrypted)
    
    Returns the created LLM setting
    """
    try:
        setting_dict = setting_data.dict()
        setting_dict['created_by'] = current_user.id
        
        setting_result = await db.create_llm_setting(setting_dict)
        
        setting = LLMSettingResponse(
            id=str(setting_result["id"]),
            provider=setting_result["provider"],
            active=setting_result.get("active", True),
            api_key_configured=bool(setting_result.get("api_key")),
            created_by=str(setting_result["created_by"]) if setting_result.get("created_by") else None,
            created_at=setting_result["created_at"] if setting_result.get("created_at") else None,
            updated_at=setting_result["updated_at"] if setting_result.get("updated_at") else None
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Update an existing LLM setting (Admin only)
    
    - **setting_id**: UUID of the LLM setting to update
    - **api_key**: Optional new API key (will be encrypted)
    - **active**: Optional active status
    
    Returns the updated LLM setting
    """
    try:
        setting_dict = setting_data.dict(exclude_unset=True)
        setting_result = await db.update_llm_setting(str(setting_id), setting_dict)
        
        if not setting_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="LLM setting not found"
            )
        
        setting = LLMSettingResponse(
            id=str(setting_result["id"]),
            provider=setting_result["provider"],
            active=setting_result.get("active", True),
            api_key_configured=bool(setting_result.get("api_key")),
            created_by=str(setting_result["created_by"]) if setting_result.get("created_by") else None,
            created_at=setting_result["created_at"] if setting_result.get("created_at") else None,
            updated_at=setting_result["updated_at"] if setting_result.get("updated_at") else None
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Create or update LLM setting based on provider (Admin only)
    
    - **provider**: LLM provider name
    - **api_key**: API key for the provider
    
    If a setting for this provider exists, it will be updated.
    Otherwise, a new setting will be created.
    """
    try:
        setting_dict = setting_data.dict()
        setting_dict['created_by'] = current_user.id
        
        setting_result = await db.upsert_llm_setting(setting_dict)
        
        setting = LLMSettingResponse(
            id=str(setting_result["id"]),
            provider=setting_result["provider"],
            active=setting_result.get("active", True),
            api_key_configured=bool(setting_result.get("api_key")),
            created_by=str(setting_result["created_by"]) if setting_result.get("created_by") else None,
            created_at=setting_result["created_at"] if setting_result.get("created_at") else None,
            updated_at=setting_result["updated_at"] if setting_result.get("updated_at") else None
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Delete an LLM setting (Admin only)
    
    - **setting_id**: UUID of the LLM setting to delete
    
    Returns success confirmation
    """
    try:
        success = await db.delete_llm_setting(str(setting_id))
        
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get decrypted API key for a specific setting (Admin only)
    
    - **setting_id**: UUID of the LLM setting
    
    Returns the decrypted API key (use with caution)
    """
    try:
        # TODO: Implement decryption in lightweight service
        # This requires access to encryption service
        # For now, returning an error message
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="API key decryption not yet implemented in lightweight service"
        )
        
        # api_key = await llm_settings_service.get_decrypted_api_key_by_id(
        #     db, setting_id
        # )
        
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get decrypted API key for a specific provider (Admin only)
    
    - **provider**: Provider name (openai, anthropic, etc.)
    
    Returns the decrypted API key for the active setting of this provider
    """
    try:
        # TODO: Implement decryption in lightweight service
        # This requires access to encryption service
        # For now, returning an error message
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="API key decryption not yet implemented in lightweight service"
        )
        
        # api_key = await llm_settings_service.get_decrypted_api_key_by_provider(
        #     db, provider
        # )
        
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get only active LLM settings (for dropdowns, etc.)
    
    Returns list of active LLM settings with masked API keys
    """
    try:
        settings_data = await db.get_active_llm_settings()
        
        settings = [
            LLMSettingResponse(
                id=str(setting["id"]),
                provider=setting["provider"],
                active=setting.get("active", True),
                api_key_configured=bool(setting.get("api_key")),
                created_by=str(setting["created_by"]) if setting.get("created_by") else None,
                created_at=setting["created_at"] if setting.get("created_at") else None,
                updated_at=setting["updated_at"] if setting.get("updated_at") else None
            )
            for setting in settings_data
        ]
        
        logger.info(f"Retrieved {len(settings)} active LLM settings for user {current_user.id}")
        return settings
        
    except Exception as e:
        logger.error(f"Error retrieving active LLM settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active LLM settings"
        )
