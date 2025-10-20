from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from typing import List, Optional
import uuid
from datetime import datetime

from app.models.models import LLMSetting
from app.models.schemas import LLMSettingCreate, LLMSettingUpdate, LLMSettingResponse
from app.utils.encryption import api_key_manager
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LLMSettingsService:
    """Service for managing LLM settings with encrypted API keys"""
    
    async def get_all_settings(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[LLMSettingResponse]:
        """Get all LLM settings with pagination"""
        try:
            result = await db.execute(
                select(LLMSetting)
                .order_by(LLMSetting.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            settings = result.scalars().all()
            
            return [
                LLMSettingResponse(
                    id=s.id,
                    provider=s.provider,
                    active=s.active,
                    api_key='***MASKED***' if s.api_key else None,
                    created_by=s.created_by,
                    created_at=s.created_at,
                    updated_at=s.updated_at
                )
                for s in settings
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving LLM settings: {str(e)}")
            raise
    
    async def get_setting_by_id(
        self, 
        db: AsyncSession, 
        setting_id: uuid.UUID
    ) -> Optional[LLMSettingResponse]:
        """Get LLM setting by ID"""
        try:
            result = await db.execute(
                select(LLMSetting)
                .where(LLMSetting.id == setting_id)
            )
            setting = result.scalar_one_or_none()
            
            if not setting:
                return None
                
            return LLMSettingResponse(
                id=setting.id,
                provider=setting.provider,
                active=setting.active,
                api_key='***MASKED***' if setting.api_key else None,
                created_by=setting.created_by,
                created_at=setting.created_at,
                updated_at=setting.updated_at
            )
            
        except Exception as e:
            logger.error(f"Error retrieving LLM setting {setting_id}: {str(e)}")
            raise
    
    async def get_settings_by_provider(
        self, 
        db: AsyncSession, 
        provider: str
    ) -> Optional[LLMSettingResponse]:
        """Get LLM setting by provider (returns first active one)"""
        try:
            result = await db.execute(
                select(LLMSetting)
                .where(
                    and_(
                        LLMSetting.provider == provider,
                        LLMSetting.active == True
                    )
                )
                .order_by(LLMSetting.created_at.desc())
                .limit(1)
            )
            setting = result.scalar_one_or_none()
            
            if not setting:
                return None
                
            return LLMSettingResponse(
                id=setting.id,
                provider=setting.provider,
                active=setting.active,
                api_key='***MASKED***' if setting.api_key else None,
                created_by=setting.created_by,
                created_at=setting.created_at,
                updated_at=setting.updated_at
            )
            
        except Exception as e:
            logger.error(f"Error retrieving LLM setting for provider {provider}: {str(e)}")
            raise
    
    async def get_active_settings(self, db: AsyncSession) -> List[LLMSettingResponse]:
        """Get only active LLM settings"""
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
                    api_key='***MASKED***' if s.api_key else None,
                    created_by=s.created_by,
                    created_at=s.created_at,
                    updated_at=s.updated_at
                )
                for s in settings
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving active LLM settings: {str(e)}")
            raise
    
    async def create_setting(
        self, 
        db: AsyncSession, 
        setting_data: LLMSettingCreate,
        created_by: uuid.UUID
    ) -> LLMSettingResponse:
        """Create a new LLM setting"""
        try:
            # Validate required fields
            if not setting_data.provider:
                raise ValueError("Provider is required")
            
            if not setting_data.api_key:
                raise ValueError("API key is required")
            
            # Validate API key format
            if not api_key_manager.validate_api_key_format(setting_data.api_key, setting_data.provider):
                raise ValueError(f"Invalid API key format for provider {setting_data.provider}")
            
            # Check if provider setting already exists (for providers that should be unique)
            existing = await db.execute(
                select(LLMSetting)
                .where(LLMSetting.provider == setting_data.provider)
            )
            if existing.scalar_one_or_none():
                raise ValueError(f"LLM setting for provider '{setting_data.provider}' already exists. Use update or upsert instead.")
            
            # Encrypt the API key
            api_key = api_key_manager.store_api_key(setting_data.api_key)
            
            # Create new setting
            new_setting = LLMSetting(
                provider=setting_data.provider,
                api_key=api_key,
                active=setting_data.active if setting_data.active is not None else True,
                created_by=created_by
            )
            
            db.add(new_setting)
            await db.commit()
            await db.refresh(new_setting)
            
            logger.info(f"Created LLM setting {new_setting.id} for provider {new_setting.provider}")
            
            return LLMSettingResponse(
                id=new_setting.id,
                provider=new_setting.provider,
                active=new_setting.active,
                api_key='***MASKED***' if new_setting.api_key else None,
                created_by=new_setting.created_by,
                created_at=new_setting.created_at,
                updated_at=new_setting.updated_at
            )
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating LLM setting: {str(e)}")
            raise
    
    async def update_setting(
        self, 
        db: AsyncSession, 
        setting_id: uuid.UUID,
        setting_data: LLMSettingUpdate
    ) -> Optional[LLMSettingResponse]:
        """Update an existing LLM setting"""
        try:
            # Get existing setting
            result = await db.execute(
                select(LLMSetting)
                .where(LLMSetting.id == setting_id)
            )
            setting = result.scalar_one_or_none()
            
            if not setting:
                return None
            
            # Update fields if provided
            update_data = {}
            
            if setting_data.active is not None:
                update_data['active'] = setting_data.active
            
            if setting_data.api_key is not None:
                # Validate API key format if provided
                if setting_data.api_key and not api_key_manager.validate_api_key_format(
                    setting_data.api_key, setting.provider
                ):
                    raise ValueError(f"Invalid API key format for provider {setting.provider}")
                
                # Encrypt the new API key
                api_key = api_key_manager.store_api_key(setting_data.api_key)
                update_data['api_key'] = api_key
            
            if update_data:
                update_data['updated_at'] = datetime.utcnow()
                
                await db.execute(
                    update(LLMSetting)
                    .where(LLMSetting.id == setting_id)
                    .values(**update_data)
                )
                await db.commit()
                
                # Refresh the setting
                await db.refresh(setting)
            
            logger.info(f"Updated LLM setting {setting_id}")
            
            return LLMSettingResponse(
                id=setting.id,
                provider=setting.provider,
                active=setting.active,
                api_key='***MASKED***' if setting.api_key else None,
                created_by=setting.created_by,
                created_at=setting.created_at,
                updated_at=setting.updated_at
            )
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating LLM setting {setting_id}: {str(e)}")
            raise
    
    async def upsert_setting(
        self, 
        db: AsyncSession, 
        setting_data: LLMSettingCreate,
        created_by: uuid.UUID
    ) -> LLMSettingResponse:
        """Create or update LLM setting based on provider"""
        try:
            # Check if setting exists for this provider
            result = await db.execute(
                select(LLMSetting)
                .where(LLMSetting.provider == setting_data.provider)
            )
            existing_setting = result.scalar_one_or_none()
            
            if existing_setting:
                # Update existing setting - only update fields that are provided
                update_data = LLMSettingUpdate()
                
                if setting_data.api_key is not None:
                    update_data.api_key = setting_data.api_key
                
                if setting_data.active is not None:
                    update_data.active = setting_data.active
                else:
                    update_data.active = True  # Default to active
                    
                return await self.update_setting(db, existing_setting.id, update_data)
            else:
                # Create new setting - api_key is required for new settings
                if not setting_data.api_key:
                    raise ValueError("API key is required for new LLM settings")
                return await self.create_setting(db, setting_data, created_by)
                
        except Exception as e:
            logger.error(f"Error upserting LLM setting: {str(e)}")
            raise
    
    async def delete_setting(
        self, 
        db: AsyncSession, 
        setting_id: uuid.UUID
    ) -> bool:
        """Delete an LLM setting"""
        try:
            # Check if setting exists
            result = await db.execute(
                select(LLMSetting)
                .where(LLMSetting.id == setting_id)
            )
            setting = result.scalar_one_or_none()
            
            if not setting:
                return False
            
            # Delete the setting
            await db.execute(
                delete(LLMSetting)
                .where(LLMSetting.id == setting_id)
            )
            await db.commit()
            
            logger.info(f"Deleted LLM setting {setting_id}")
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting LLM setting {setting_id}: {str(e)}")
            raise
    
    async def get_decrypted_api_key_by_id(
        self, 
        db: AsyncSession, 
        setting_id: uuid.UUID
    ) -> Optional[str]:
        """Get decrypted API key for a specific setting"""
        try:
            result = await db.execute(
                select(LLMSetting)
                .where(LLMSetting.id == setting_id)
            )
            setting = result.scalar_one_or_none()
            
            if not setting or not setting.api_key:
                return None
            
            # Decrypt the API key
            decrypted_key = api_key_manager.retrieve_api_key(setting.api_key)
            
            logger.warning(f"API key decrypted for setting {setting_id}")
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Error getting decrypted API key for setting {setting_id}: {str(e)}")
            raise
    
    async def get_decrypted_api_key_by_provider(
        self, 
        db: AsyncSession, 
        provider: str
    ) -> Optional[str]:
        """Get decrypted API key for a specific provider"""
        try:
            result = await db.execute(
                select(LLMSetting)
                .where(
                    and_(
                        LLMSetting.provider == provider,
                        LLMSetting.active == True
                    )
                )
                .order_by(LLMSetting.created_at.desc())
                .limit(1)
            )
            setting = result.scalar_one_or_none()
            
            if not setting or not setting.api_key:
                return None
            
            # Decrypt the API key
            decrypted_key = api_key_manager.retrieve_api_key(setting.api_key)
            
            logger.warning(f"API key decrypted for provider {provider}")
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Error getting decrypted API key for provider {provider}: {str(e)}")
            raise


# Create a singleton instance
llm_settings_service = LLMSettingsService()