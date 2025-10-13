from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import uuid

from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_admin_user
from app.models.schemas import (
    ModuleConfigurationCreate,
    ModuleConfigurationUpdate,
    ModuleConfigurationResponse,
    LLMSettingResponse,
    SuccessResponse,
    ErrorResponse
)
from app.models.models import User
from app.services.module_configuration_service import module_configuration_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[ModuleConfigurationResponse])
async def get_module_configurations(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get all module configurations (Admin only)
    
    - **skip**: Number of configurations to skip (for pagination)
    - **limit**: Maximum number of configurations to return
    
    Returns list of all module configurations with joined LLM settings and AI personality data
    """
    try:
        configurations = await module_configuration_service.get_all_configurations(
            db, skip=skip, limit=limit
        )
        
        logger.info(f"Retrieved {len(configurations)} module configurations for admin {current_user.id}")
        return configurations
        
    except Exception as e:
        logger.error(f"Error retrieving module configurations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve module configurations"
        )


@router.get("/{config_id}", response_model=ModuleConfigurationResponse)
async def get_module_configuration_by_id(
    config_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get specific module configuration by ID (Admin only)
    
    - **config_id**: UUID of the module configuration to retrieve
    
    Returns the module configuration details with joined data
    """
    try:
        configuration = await module_configuration_service.get_configuration_by_id(
            db, config_id
        )
        
        if not configuration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module configuration not found"
            )
        
        logger.info(f"Retrieved module configuration {config_id} for admin {current_user.id}")
        return configuration
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving module configuration {config_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve module configuration"
        )


@router.post("/", response_model=ModuleConfigurationResponse, status_code=status.HTTP_201_CREATED)
async def create_module_configuration(
    config_data: ModuleConfigurationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Create or update module configuration (upsert) (Admin only)
    
    - **module_name**: Module name (data_processing_engine, semantic_search_engine, ai_chat_integration, survey_suggestions_generation)
    - **llm_setting_id**: UUID of the LLM setting to use
    - **model**: AI model name for this module
    - **temperature**: Optional temperature setting (0.0-2.0)
    - **max_tokens**: Optional maximum tokens
    - **max_completion_tokens**: Optional maximum completion tokens
    - **ai_personality_id**: Optional AI personality to use
    
    If configuration exists for this module, it will be updated. Otherwise, a new one will be created.
    """
    try:
        configuration = await module_configuration_service.upsert_configuration(
            db, config_data, created_by=current_user.id
        )
        
        logger.info(f"Created/updated module configuration for {config_data.module_name} by admin {current_user.id}")
        return configuration
        
    except ValueError as e:
        logger.warning(f"Invalid data for module configuration creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating module configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create module configuration"
        )


@router.put("/{config_id}", response_model=ModuleConfigurationResponse)
async def update_module_configuration(
    config_id: uuid.UUID,
    config_data: ModuleConfigurationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Update an existing module configuration (Admin only)
    
    - **config_id**: UUID of the module configuration to update
    - **model**: Optional new model name
    - **temperature**: Optional new temperature setting
    - **max_tokens**: Optional new maximum tokens
    - **max_completion_tokens**: Optional new maximum completion tokens
    - **active**: Optional active status
    - **ai_personality_id**: Optional new AI personality
    
    Returns the updated module configuration
    """
    try:
        configuration = await module_configuration_service.update_configuration(
            db, config_id, config_data
        )
        
        if not configuration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module configuration not found"
            )
        
        logger.info(f"Updated module configuration {config_id} by admin {current_user.id}")
        return configuration
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid data for module configuration update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating module configuration {config_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update module configuration"
        )


@router.delete("/{config_id}", response_model=SuccessResponse)
async def delete_module_configuration(
    config_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Delete a module configuration (Admin only)
    
    - **config_id**: UUID of the module configuration to delete
    
    Returns success confirmation
    """
    try:
        success = await module_configuration_service.delete_configuration(
            db, config_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module configuration not found"
            )
        
        logger.info(f"Deleted module configuration {config_id} by admin {current_user.id}")
        return SuccessResponse(
            message=f"Module configuration {config_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting module configuration {config_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete module configuration"
        )


@router.get("/module/{module_name}/active", response_model=ModuleConfigurationResponse)
async def get_active_module_configuration(
    module_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get active configuration for a specific module
    
    - **module_name**: Name of the module to get configuration for
    
    Returns the active configuration for the specified module
    """
    try:
        configuration = await module_configuration_service.get_active_configuration_for_module(
            db, module_name
        )
        
        if not configuration:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active configuration found for module {module_name}"
            )
        
        logger.info(f"Retrieved active configuration for module {module_name} for user {current_user.id}")
        return configuration
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving active configuration for module {module_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active module configuration"
        )


@router.get("/llm-settings/available", response_model=List[LLMSettingResponse])
async def get_available_llm_settings(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get available LLM settings for dropdown/selection (Admin only)
    
    Returns list of active LLM settings that can be used in module configurations
    """
    try:
        llm_settings = await module_configuration_service.get_available_llm_settings(db)
        
        logger.info(f"Retrieved {len(llm_settings)} available LLM settings for admin {current_user.id}")
        return llm_settings
        
    except Exception as e:
        logger.error(f"Error retrieving available LLM settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve available LLM settings"
        )


@router.get("/modules/supported")
async def get_supported_modules(
    current_user: User = Depends(get_current_user)
):
    """
    Get list of supported module names
    
    Returns list of valid module names that can be configured
    """
    try:
        modules = module_configuration_service.get_supported_modules()
        
        logger.info(f"Retrieved supported modules for user {current_user.id}")
        return {
            "modules": modules,
            "total": len(modules)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving supported modules: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supported modules"
        )


@router.get("/validate/module-name/{module_name}")
async def validate_module_name(
    module_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Validate if a module name is supported
    
    - **module_name**: Module name to validate
    
    Returns validation result
    """
    try:
        is_valid = module_configuration_service.validate_module_name(module_name)
        
        return {
            "module_name": module_name,
            "is_valid": is_valid,
            "supported_modules": module_configuration_service.get_supported_modules()
        }
        
    except Exception as e:
        logger.error(f"Error validating module name {module_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate module name"
        )