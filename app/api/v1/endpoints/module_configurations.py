from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import uuid

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import get_current_regular_user, get_current_admin_user, SimpleUser
from app.models.schemas import (
    ModuleConfigurationCreate,
    ModuleConfigurationUpdate,
    ModuleConfigurationResponse,
    LLMSettingResponse,
    SuccessResponse
)
from app.services.survey_builder_service import survey_builder_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[ModuleConfigurationResponse])
async def get_module_configurations(
    skip: int = 0,
    limit: int = 100,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get all module configurations (Admin only)
    
    - **skip**: Number of configurations to skip (for pagination)
    - **limit**: Maximum number of configurations to return
    
    Returns list of all module configurations with joined LLM settings and AI personality data
    """
    try:
        configurations_data = await db.get_all_module_configurations(skip=skip, limit=limit)
        
        # Convert to response format
        configurations = [
            ModuleConfigurationResponse(
                id=str(c["id"]),
                module_name=c["module_name"],
                llm_setting_id=str(c["llm_setting_id"]) if c.get("llm_setting_id") else None,
                temperature=float(c["temperature"]) if c.get("temperature") else 0.7,
                max_tokens=c.get("max_tokens", 1000),
                max_completion_tokens=c.get("max_completion_tokens", 1000),
                active=c.get("active", True),
                created_at=c["created_at"].isoformat() if c.get("created_at") else "",
                updated_at=c["updated_at"].isoformat() if c.get("updated_at") else "",
                created_by=str(c["created_by"]) if c.get("created_by") else None,
                ai_personality_id=str(c["ai_personality_id"]) if c.get("ai_personality_id") else None,
                model=c.get("model", "")
            )
            for c in configurations_data
        ]
        
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get specific module configuration by ID (Admin only)
    
    - **config_id**: UUID of the module configuration to retrieve
    
    Returns the module configuration details with joined data
    """
    try:
        configuration_data = await db.get_module_configuration_by_id(str(config_id))
        
        if not configuration_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module configuration not found"
            )
        
        configuration = ModuleConfigurationResponse(
            id=str(configuration_data["id"]),
            module_name=configuration_data["module_name"],
            llm_setting_id=str(configuration_data["llm_setting_id"]) if configuration_data.get("llm_setting_id") else None,
            temperature=float(configuration_data["temperature"]) if configuration_data.get("temperature") else 0.7,
            max_tokens=configuration_data.get("max_tokens", 1000),
            max_completion_tokens=configuration_data.get("max_completion_tokens", 1000),
            active=configuration_data.get("active", True),
            created_at=configuration_data["created_at"].isoformat() if configuration_data.get("created_at") else "",
            updated_at=configuration_data["updated_at"].isoformat() if configuration_data.get("updated_at") else "",
            created_by=str(configuration_data["created_by"]) if configuration_data.get("created_by") else None,
            ai_personality_id=str(configuration_data["ai_personality_id"]) if configuration_data.get("ai_personality_id") else None,
            model=configuration_data.get("model", "")
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Create or update module configuration (upsert) (Admin only)
    
    - **module_name**: Module name (semantic_search_engine, ai_chat_integration, survey_suggestions_generation, survey_builder)
    - **llm_setting_id**: UUID of the LLM setting to use
    - **model**: AI model name for this module
    - **temperature**: Optional temperature setting (0.0-2.0)
    - **max_tokens**: Optional maximum tokens
    - **max_completion_tokens**: Optional maximum completion tokens
    - **ai_personality_id**: Optional AI personality to use
    
    If configuration exists for this module, it will be updated. Otherwise, a new one will be created.
    """
    try:
        # Use the LightweightDBService directly instead of the old service
        configuration_data = await db.upsert_module_configuration(
            module_name=config_data.module_name,
            llm_setting_id=str(config_data.llm_setting_id),
            model=config_data.model,
            temperature=config_data.temperature or 0.7,
            max_tokens=config_data.max_tokens or 1000,
            max_completion_tokens=config_data.max_completion_tokens or 1000,
            active=True,  # Default to active
            ai_personality_id=str(config_data.ai_personality_id) if config_data.ai_personality_id else None,
            created_by=current_user.id
        )
        
        # Convert to response format
        configuration = ModuleConfigurationResponse(
            id=str(configuration_data["id"]),
            module_name=configuration_data["module_name"],
            llm_setting_id=str(configuration_data["llm_setting_id"]) if configuration_data.get("llm_setting_id") else None,
            model=configuration_data["model"],
            temperature=float(configuration_data["temperature"]) if configuration_data.get("temperature") else 0.7,
            max_tokens=configuration_data.get("max_tokens", 1000),
            max_completion_tokens=configuration_data.get("max_completion_tokens", 1000),
            active=configuration_data.get("active", True),
            created_at=configuration_data["created_at"].isoformat() if configuration_data.get("created_at") else None,
            updated_at=configuration_data["updated_at"].isoformat() if configuration_data.get("updated_at") else None,
            created_by=str(configuration_data["created_by"]) if configuration_data.get("created_by") else None,
            ai_personality_id=str(configuration_data["ai_personality_id"]) if configuration_data.get("ai_personality_id") else None
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
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
        config_dict = config_data.dict(exclude_unset=True)
        configuration_data = await db.update_module_configuration(str(config_id), config_dict)
        
        if not configuration_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module configuration not found"
            )
        
        configuration = ModuleConfigurationResponse(
            id=str(configuration_data["id"]),
            module_name=configuration_data["module_name"],
            llm_setting_id=str(configuration_data["llm_setting_id"]) if configuration_data.get("llm_setting_id") else None,
            temperature=float(configuration_data["temperature"]) if configuration_data.get("temperature") else 0.7,
            max_tokens=configuration_data.get("max_tokens", 1000),
            max_completion_tokens=configuration_data.get("max_completion_tokens", 1000),
            active=configuration_data.get("active", True),
            created_at=configuration_data["created_at"].isoformat() if configuration_data.get("created_at") else "",
            updated_at=configuration_data["updated_at"].isoformat() if configuration_data.get("updated_at") else "",
            created_by=str(configuration_data["created_by"]) if configuration_data.get("created_by") else None,
            ai_personality_id=str(configuration_data["ai_personality_id"]) if configuration_data.get("ai_personality_id") else None,
            model=configuration_data.get("model", "")
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Delete a module configuration (Admin only)
    
    - **config_id**: UUID of the module configuration to delete
    
    Returns success confirmation
    """
    try:
        deleted = await db.delete_module_configuration(str(config_id))
        
        if not deleted:
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get active configuration for a specific module
    
    - **module_name**: Name of the module to get configuration for
    
    Returns the active configuration for the specified module
    """
    try:
        configuration_data = await db.get_active_configuration_for_module(module_name)
        
        if not configuration_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active configuration found for module {module_name}"
            )
        
        configuration = ModuleConfigurationResponse(
            id=str(configuration_data["id"]),
            module_name=configuration_data["module_name"],
            llm_setting_id=str(configuration_data["llm_setting_id"]) if configuration_data.get("llm_setting_id") else None,
            temperature=float(configuration_data["temperature"]) if configuration_data.get("temperature") else 0.7,
            max_tokens=configuration_data.get("max_tokens", 1000),
            max_completion_tokens=configuration_data.get("max_completion_tokens", 1000),
            active=configuration_data.get("active", True),
            created_at=configuration_data["created_at"].isoformat() if configuration_data.get("created_at") else "",
            updated_at=configuration_data["updated_at"].isoformat() if configuration_data.get("updated_at") else "",
            created_by=str(configuration_data["created_by"]) if configuration_data.get("created_by") else None,
            ai_personality_id=str(configuration_data["ai_personality_id"]) if configuration_data.get("ai_personality_id") else None,
            model=configuration_data.get("model", "")
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get available LLM settings for dropdown/selection (Admin only)
    
    Returns list of active LLM settings that can be used in module configurations
    """
    try:
        llm_settings_data = await db.get_active_llm_settings()
        
        llm_settings = [
            LLMSettingResponse(
                id=str(setting["id"]),
                provider=setting["provider"],
                active=setting.get("active", True),
                api_key_configured=bool(setting.get("api_key")),
                created_by=str(setting["created_by"]) if setting.get("created_by") else None,
                created_at=setting["created_at"] if setting.get("created_at") else None,
                updated_at=setting["updated_at"] if setting.get("updated_at") else None
            )
            for setting in llm_settings_data
        ]
        
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
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get list of supported module names
    
    Returns list of valid module names that can be configured
    """
    try:
        # Supported module names
        SUPPORTED_MODULES = [
            "semantic_search_engine", 
            "ai_chat_integration",
            "survey_suggestions_generation",
            "survey_builder"
        ]
        
        logger.info(f"Retrieved supported modules for user {current_user.id}")
        return {
            "modules": SUPPORTED_MODULES,
            "total": len(SUPPORTED_MODULES)
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
):
    """
    Validate if a module name is supported
    
    - **module_name**: Module name to validate
    
    Returns validation result
    """
    try:
        # Supported module names
        SUPPORTED_MODULES = [
            "semantic_search_engine", 
            "ai_chat_integration",
            "survey_suggestions_generation",
            "survey_builder"
        ]
        
        is_valid = module_name in SUPPORTED_MODULES
        
        return {
            "module_name": module_name,
            "is_valid": is_valid,
            "supported_modules": SUPPORTED_MODULES
        }
        
    except Exception as e:
        logger.error(f"Error validating module name {module_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate module name"
        )


@router.get("/survey-builder", response_model=ModuleConfigurationResponse)
async def get_survey_builder_configuration(
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get survey builder module configuration with AI personality and LLM settings
    
    This endpoint provides the complete configuration needed for the survey builder chat interface,
    including the AI personality prompts and LLM settings.
    
    Returns the survey builder module configuration or 404 if not configured
    """
    try:
        config = await survey_builder_service.get_survey_builder_config(db)
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey builder module not configured. Please contact an administrator."
            )
        
        if not config.active:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Survey builder module is currently disabled"
            )
        
        logger.info(f"Retrieved survey builder configuration for user {current_user.id}")
        return config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving survey builder configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve survey builder configuration"
        )