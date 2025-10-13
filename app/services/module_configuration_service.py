from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, text
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any
import uuid
import time
from datetime import datetime
from cachetools import TTLCache

from app.models.models import ModuleConfiguration, LLMSetting, AIPersonality
from app.models.schemas import (
    ModuleConfigurationCreate, 
    ModuleConfigurationUpdate, 
    ModuleConfigurationResponse,
    LLMSettingResponse
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ModuleConfigurationService:
    """Optimized service for managing module configurations with caching"""
    
    # Supported module names
    SUPPORTED_MODULES = [
        "data_processing_engine",
        "semantic_search_engine", 
        "ai_chat_integration",
        "survey_suggestions_generation"
    ]
    
    def __init__(self):
        # Configuration cache with TTL (Time To Live)
        self._config_cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes TTL
        
        # Complete configuration data cache (includes all related data)
        self._complete_data_cache = TTLCache(maxsize=20, ttl=300)  # 5 minutes TTL
        
        logger.info("ModuleConfigurationService initialized with caching")
    
    def get_supported_modules(self) -> List[str]:
        """Get list of supported module names"""
        return self.SUPPORTED_MODULES.copy()
    
    def validate_module_name(self, module_name: str) -> bool:
        """Validate if module name is supported"""
        return module_name in self.SUPPORTED_MODULES
    
    async def get_all_configurations(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[ModuleConfigurationResponse]:
        """Get all module configurations with joined data"""
        try:
            result = await db.execute(
                select(ModuleConfiguration)
                .options(
                    selectinload(ModuleConfiguration.llm_setting),
                    selectinload(ModuleConfiguration.ai_personality)
                )
                .order_by(ModuleConfiguration.module_name)
                .offset(skip)
                .limit(limit)
            )
            configurations = result.scalars().all()
            
            return [
                await self._build_configuration_response(config)
                for config in configurations
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving module configurations: {str(e)}")
            raise
    
    async def get_configuration_by_id(
        self, 
        db: AsyncSession, 
        config_id: uuid.UUID
    ) -> Optional[ModuleConfigurationResponse]:
        """Get module configuration by ID with joined data"""
        try:
            result = await db.execute(
                select(ModuleConfiguration)
                .options(
                    selectinload(ModuleConfiguration.llm_setting),
                    selectinload(ModuleConfiguration.ai_personality)
                )
                .where(ModuleConfiguration.id == config_id)
            )
            configuration = result.scalar_one_or_none()
            
            if not configuration:
                return None
                
            return await self._build_configuration_response(configuration)
            
        except Exception as e:
            logger.error(f"Error retrieving module configuration {config_id}: {str(e)}")
            raise
    
    async def get_active_configuration_for_module(
        self, 
        db: AsyncSession, 
        module_name: str
    ) -> Optional[ModuleConfigurationResponse]:
        """Get active configuration for a specific module"""
        try:
            if not self.validate_module_name(module_name):
                raise ValueError(f"Unsupported module name: {module_name}")
            
            result = await db.execute(
                select(ModuleConfiguration)
                .options(
                    selectinload(ModuleConfiguration.llm_setting),
                    selectinload(ModuleConfiguration.ai_personality)
                )
                .where(
                    and_(
                        ModuleConfiguration.module_name == module_name,
                        ModuleConfiguration.active == True
                    )
                )
                .order_by(ModuleConfiguration.created_at.desc())
                .limit(1)
            )
            configuration = result.scalar_one_or_none()
            
            if not configuration:
                return None
                
            return await self._build_configuration_response(configuration)
            
        except Exception as e:
            logger.error(f"Error retrieving active configuration for module {module_name}: {str(e)}")
            raise
    
    async def get_active_configuration_for_service(
        self, 
        db: AsyncSession, 
        module_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get active configuration for a specific module with caching
        Returns a dictionary with all configuration data for service use
        """
        cache_key = f"active_config:{module_name}"
        
        # Check cache first
        if cache_key in self._config_cache:
            logger.debug(f"Cache hit for active configuration: {cache_key}")
            return self._config_cache[cache_key]
        
        try:
            if not self.validate_module_name(module_name):
                raise ValueError(f"Unsupported module name: {module_name}")
            
            # Single optimized query with all necessary JOINs
            query = text("""
                SELECT 
                    mc.id,
                    mc.module_name,
                    mc.llm_setting_id,
                    mc.model,
                    mc.temperature,
                    mc.max_tokens,
                    mc.max_completion_tokens,
                    mc.active,
                    mc.ai_personality_id,
                    ls.provider,
                    ls.api_key,
                    ls.active as llm_active,
                    ap.name as personality_name,
                    ap.detailed_analysis_prompt,
                    ap.system_prompt,
                    ap.is_active as personality_active
                FROM module_configurations mc
                LEFT JOIN llm_settings ls ON mc.llm_setting_id = ls.id
                LEFT JOIN ai_personalities ap ON mc.ai_personality_id = ap.id
                WHERE mc.active = true AND mc.module_name = :module_name
                ORDER BY mc.created_at DESC
                LIMIT 1
            """)
            
            result = await db.execute(query, {"module_name": module_name})
            row = result.fetchone()
            
            if not row:
                logger.warning(f"No active configuration found for module: {module_name}")
                return None
            
            # Convert to dictionary format expected by services
            config = {
                "id": row[0],
                "module_name": row[1],
                "llm_setting_id": row[2],
                "model": row[3],
                "temperature": row[4],
                "max_tokens": row[5],
                "max_completion_tokens": row[6],
                "active": row[7],
                "ai_personality_id": row[8],
                "provider": row[9],
                "api_key": row[10]
            }
            
            # Add AI personality if available
            if row[11]:  # personality_name
                config["ai_personality"] = {
                    "name": row[11],
                    "detailed_analysis_prompt": row[12],
                    "system_prompt": row[13],
                    "active": row[14]
                }
            
            # Cache the result
            self._config_cache[cache_key] = config
            logger.debug(f"Cached active configuration: {cache_key}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error getting active configuration for {module_name}: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear all caches"""
        self._config_cache.clear()
        self._complete_data_cache.clear()
        logger.info("Cleared all configuration caches")
    
    def clear_module_cache(self, module_name: str):
        """Clear cache for specific module"""
        keys_to_remove = [
            key for key in self._config_cache.keys() 
            if module_name in str(key)
        ]
        for key in keys_to_remove:
            del self._config_cache[key]
        
        # Also clear complete data cache
        complete_keys = [
            key for key in self._complete_data_cache.keys()
            if module_name in str(key)
        ]
        for key in complete_keys:
            del self._complete_data_cache[key]
        
        logger.debug(f"Cleared cache for module: {module_name}")
    
    async def create_configuration(
        self, 
        db: AsyncSession, 
        config_data: ModuleConfigurationCreate,
        created_by: uuid.UUID
    ) -> ModuleConfigurationResponse:
        """Create a new module configuration"""
        try:
            # Validate module name
            if not self.validate_module_name(config_data.module_name):
                raise ValueError(f"Unsupported module name: {config_data.module_name}")
            
            # Validate required fields
            if not config_data.llm_setting_id:
                raise ValueError("LLM setting ID is required")
            
            if not config_data.model:
                raise ValueError("Model is required")
            
            # Validate LLM setting exists and is active
            llm_setting = await db.execute(
                select(LLMSetting)
                .where(
                    and_(
                        LLMSetting.id == config_data.llm_setting_id,
                        LLMSetting.active == True
                    )
                )
            )
            if not llm_setting.scalar_one_or_none():
                raise ValueError("LLM setting not found or inactive")
            
            # Validate AI personality if provided
            if config_data.ai_personality_id:
                ai_personality = await db.execute(
                    select(AIPersonality)
                    .where(
                        and_(
                            AIPersonality.id == config_data.ai_personality_id,
                            AIPersonality.is_active == True
                        )
                    )
                )
                if not ai_personality.scalar_one_or_none():
                    raise ValueError("AI personality not found or inactive")
            
            # Validate temperature range
            if config_data.temperature is not None and (config_data.temperature < 0 or config_data.temperature > 2):
                raise ValueError("Temperature must be between 0 and 2")
            
            # Validate token limits
            if config_data.max_tokens is not None and config_data.max_tokens < 1:
                raise ValueError("Max tokens must be greater than 0")
            
            if config_data.max_completion_tokens is not None and config_data.max_completion_tokens < 0:
                raise ValueError("Max completion tokens must be 0 or greater")
            
            # Create new configuration
            new_config = ModuleConfiguration(
                module_name=config_data.module_name,
                llm_setting_id=config_data.llm_setting_id,
                model=config_data.model,
                temperature=config_data.temperature,
                max_tokens=config_data.max_tokens,
                max_completion_tokens=config_data.max_completion_tokens,
                active=True,
                created_by=created_by,
                ai_personality_id=config_data.ai_personality_id
            )
            
            db.add(new_config)
            await db.commit()
            await db.refresh(new_config)
            
            # Load relationships
            await db.refresh(new_config, ['llm_setting', 'ai_personality'])
            
            logger.info(f"Created module configuration {new_config.id} for module {new_config.module_name}")
            
            return await self._build_configuration_response(new_config)
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating module configuration: {str(e)}")
            raise
    
    async def update_configuration(
        self, 
        db: AsyncSession, 
        config_id: uuid.UUID,
        config_data: ModuleConfigurationUpdate
    ) -> Optional[ModuleConfigurationResponse]:
        """Update an existing module configuration"""
        try:
            # Get existing configuration
            result = await db.execute(
                select(ModuleConfiguration)
                .options(
                    selectinload(ModuleConfiguration.llm_setting),
                    selectinload(ModuleConfiguration.ai_personality)
                )
                .where(ModuleConfiguration.id == config_id)
            )
            configuration = result.scalar_one_or_none()
            
            if not configuration:
                return None
            
            # Validate AI personality if provided
            if config_data.ai_personality_id:
                ai_personality = await db.execute(
                    select(AIPersonality)
                    .where(
                        and_(
                            AIPersonality.id == config_data.ai_personality_id,
                            AIPersonality.is_active == True
                        )
                    )
                )
                if not ai_personality.scalar_one_or_none():
                    raise ValueError("AI personality not found or inactive")
            
            # Update fields if provided
            update_data = {}
            
            if config_data.model is not None:
                update_data['model'] = config_data.model
            
            if config_data.temperature is not None:
                if config_data.temperature < 0 or config_data.temperature > 2:
                    raise ValueError("Temperature must be between 0 and 2")
                update_data['temperature'] = config_data.temperature
            
            if config_data.max_tokens is not None:
                if config_data.max_tokens < 1:
                    raise ValueError("Max tokens must be greater than 0")
                update_data['max_tokens'] = config_data.max_tokens
            
            if config_data.max_completion_tokens is not None:
                if config_data.max_completion_tokens < 0:
                    raise ValueError("Max completion tokens must be 0 or greater")
                update_data['max_completion_tokens'] = config_data.max_completion_tokens
            
            if config_data.active is not None:
                update_data['active'] = config_data.active
            
            if config_data.ai_personality_id is not None:
                update_data['ai_personality_id'] = config_data.ai_personality_id
            
            if update_data:
                update_data['updated_at'] = datetime.utcnow()
                
                await db.execute(
                    update(ModuleConfiguration)
                    .where(ModuleConfiguration.id == config_id)
                    .values(**update_data)
                )
                await db.commit()
                
                # Refresh the configuration with relationships
                await db.refresh(configuration, ['llm_setting', 'ai_personality'])
            
            logger.info(f"Updated module configuration {config_id}")
            
            return await self._build_configuration_response(configuration)
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating module configuration {config_id}: {str(e)}")
            raise
    
    async def upsert_configuration(
        self, 
        db: AsyncSession, 
        config_data: ModuleConfigurationCreate,
        created_by: uuid.UUID
    ) -> ModuleConfigurationResponse:
        """Create or update module configuration based on module name and LLM setting"""
        try:
            # Check if configuration exists for this module and LLM setting
            result = await db.execute(
                select(ModuleConfiguration)
                .where(
                    and_(
                        ModuleConfiguration.module_name == config_data.module_name,
                        ModuleConfiguration.llm_setting_id == config_data.llm_setting_id
                    )
                )
            )
            existing_config = result.scalar_one_or_none()
            
            if existing_config:
                # Update existing configuration
                update_data = ModuleConfigurationUpdate(
                    model=config_data.model,
                    temperature=config_data.temperature,
                    max_tokens=config_data.max_tokens,
                    max_completion_tokens=config_data.max_completion_tokens,
                    active=True,
                    ai_personality_id=config_data.ai_personality_id
                )
                return await self.update_configuration(db, existing_config.id, update_data)
            else:
                # Create new configuration
                return await self.create_configuration(db, config_data, created_by)
            
        except Exception as e:
            logger.error(f"Error upserting module configuration: {str(e)}")
            raise
    
    async def delete_configuration(
        self, 
        db: AsyncSession, 
        config_id: uuid.UUID
    ) -> bool:
        """Delete a module configuration"""
        try:
            # Check if configuration exists
            result = await db.execute(
                select(ModuleConfiguration)
                .where(ModuleConfiguration.id == config_id)
            )
            configuration = result.scalar_one_or_none()
            
            if not configuration:
                return False
            
            # Delete the configuration
            await db.execute(
                delete(ModuleConfiguration)
                .where(ModuleConfiguration.id == config_id)
            )
            await db.commit()
            
            logger.info(f"Deleted module configuration {config_id}")
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting module configuration {config_id}: {str(e)}")
            raise
    
    async def get_available_llm_settings(self, db: AsyncSession) -> List[LLMSettingResponse]:
        """Get available LLM settings for dropdown/selection"""
        try:
            result = await db.execute(
                select(LLMSetting)
                .where(LLMSetting.active == True)
                .order_by(LLMSetting.provider)
            )
            settings = result.scalars().all()
            
            return [
                LLMSettingResponse(
                    id=s.id,
                    provider=s.provider,
                    active=s.active,
                    api_key_configured=bool(s.api_key),
                    created_by=s.created_by,
                    created_at=s.created_at,
                    updated_at=s.updated_at
                )
                for s in settings
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving available LLM settings: {str(e)}")
            raise
    
    async def _build_configuration_response(
        self, 
        config: ModuleConfiguration
    ) -> ModuleConfigurationResponse:
        """Build configuration response with proper data types and joined data"""
        from app.models.schemas import LLMSettingNestedResponse, AIPersonalityNestedResponse
        
        # Build llm_settings nested object if available
        llm_settings_data = None
        if config.llm_setting:
            llm_settings_data = LLMSettingNestedResponse(
                id=str(config.llm_setting.id),
                provider=config.llm_setting.provider,
                api_key_configured=bool(config.llm_setting.api_key)
            )
        
        # Build ai_personality nested object if available
        ai_personality_data = None
        if config.ai_personality:
            ai_personality_data = AIPersonalityNestedResponse(
                name=config.ai_personality.name,
                detailed_analysis_prompt=config.ai_personality.detailed_analysis_prompt
            )
        
        return ModuleConfigurationResponse(
            id=config.id,
            module_name=config.module_name,
            llm_setting_id=config.llm_setting_id,
            model=config.model,
            temperature=float(config.temperature) if config.temperature else None,
            max_tokens=config.max_tokens,
            max_completion_tokens=config.max_completion_tokens,
            active=config.active,
            created_by=config.created_by,
            ai_personality_id=config.ai_personality_id,
            created_at=config.created_at,
            updated_at=config.updated_at,
            llm_settings=llm_settings_data,
            ai_personality=ai_personality_data
        )
    
    async def get_active_configuration_for_service(
        self, 
        db: AsyncSession, 
        module_name: str
    ) -> Optional[dict]:
        """Get active configuration for internal service use with decrypted API key"""
        try:
            if not self.validate_module_name(module_name):
                raise ValueError(f"Unsupported module name: {module_name}")
            
            result = await db.execute(
                select(ModuleConfiguration)
                .options(
                    selectinload(ModuleConfiguration.llm_setting),
                    selectinload(ModuleConfiguration.ai_personality)
                )
                .where(
                    and_(
                        ModuleConfiguration.module_name == module_name,
                        ModuleConfiguration.active == True
                    )
                )
                .order_by(ModuleConfiguration.created_at.desc())
                .limit(1)
            )
            configuration = result.scalar_one_or_none()
            
            if not configuration or not configuration.llm_setting:
                return None
            
            # Decrypt the API key for internal use
            from app.utils.encryption import encryption_service
            
            decrypted_api_key = ""
            if configuration.llm_setting.api_key:
                try:
                    decrypted_api_key = encryption_service.decrypt_api_key(
                        configuration.llm_setting.api_key
                    )
                except Exception as e:
                    logger.error(f"Failed to decrypt API key for {module_name}: {str(e)}")
                    return None
            
            return {
                "id": str(configuration.id),
                "module_name": configuration.module_name,
                "model": configuration.model,
                "temperature": float(configuration.temperature) if configuration.temperature else 0.7,
                "max_tokens": configuration.max_tokens or 1000,
                "max_completion_tokens": configuration.max_completion_tokens,
                "active": configuration.active,
                "provider": configuration.llm_setting.provider,
                "api_key": decrypted_api_key,
                "ai_personality": {
                    "id": str(configuration.ai_personality.id),
                    "name": configuration.ai_personality.name,
                    "detailed_analysis_prompt": configuration.ai_personality.detailed_analysis_prompt
                } if configuration.ai_personality else None
            }
            
        except Exception as e:
            logger.error(f"Error retrieving active service configuration for module {module_name}: {str(e)}")
            raise


# Create a singleton instance
module_configuration_service = ModuleConfigurationService()