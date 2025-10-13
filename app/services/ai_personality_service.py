from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload
from typing import List, Optional
import uuid
from datetime import datetime

from app.models.models import AIPersonality as AIPersonalityModel
from app.models.schemas import AIPersonalityCreate, AIPersonalityUpdate, AIPersonality
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AIPersonalityService:
    """Service for managing AI personalities"""
    
    async def get_all_personalities(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[AIPersonality]:
        """Get all AI personalities with pagination"""
        try:
            result = await db.execute(
                select(AIPersonalityModel)
                .order_by(AIPersonalityModel.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            personalities = result.scalars().all()
            
            return [
                AIPersonality(
                    id=str(p.id),
                    name=p.name,
                    description=p.description,
                    detailed_analysis_prompt=p.detailed_analysis_prompt or "",
                    suggestions_prompt=p.suggestions_prompt or "",
                    model_override=None,
                    temperature_override=None,
                    is_default=getattr(p, 'is_default', False),
                    is_active=p.is_active,
                    created_at=p.created_at.isoformat() if p.created_at else "",
                    updated_at=p.updated_at.isoformat() if p.updated_at else ""
                )
                for p in personalities
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving AI personalities: {str(e)}")
            raise
    
    async def get_personality_by_id(
        self, 
        db: AsyncSession, 
        personality_id: uuid.UUID
    ) -> Optional[AIPersonality]:
        """Get AI personality by ID"""
        try:
            result = await db.execute(
                select(AIPersonalityModel)
                .where(AIPersonalityModel.id == personality_id)
            )
            personality = result.scalar_one_or_none()
            
            if not personality:
                return None
                
            return AIPersonality(
                id=str(personality.id),
                name=personality.name,
                description=personality.description,
                detailed_analysis_prompt=personality.detailed_analysis_prompt or "",
                suggestions_prompt=personality.suggestions_prompt or "",
                model_override=None,
                temperature_override=None,
                is_default=getattr(personality, 'is_default', False),
                is_active=personality.is_active,
                created_at=personality.created_at.isoformat() if personality.created_at else "",
                updated_at=personality.updated_at.isoformat() if personality.updated_at else ""
            )
            
        except Exception as e:
            logger.error(f"Error retrieving AI personality {personality_id}: {str(e)}")
            raise
    
    async def get_active_personalities(self, db: AsyncSession) -> List[AIPersonality]:
        """Get only active AI personalities"""
        try:
            result = await db.execute(
                select(AIPersonalityModel)
                .where(AIPersonalityModel.is_active == True)
                .order_by(AIPersonalityModel.name)
            )
            personalities = result.scalars().all()
            
            return [
                AIPersonality(
                    id=str(p.id),
                    name=p.name,
                    description=p.description,
                    detailed_analysis_prompt=p.detailed_analysis_prompt or "",
                    suggestions_prompt=p.suggestions_prompt or "",
                    model_override=None,
                    temperature_override=None,
                    is_default=getattr(p, 'is_default', False),
                    is_active=p.is_active,
                    created_at=p.created_at.isoformat() if p.created_at else "",
                    updated_at=p.updated_at.isoformat() if p.updated_at else ""
                )
                for p in personalities
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving active AI personalities: {str(e)}")
            raise
    
    async def create_personality(
        self, 
        db: AsyncSession, 
        personality_data: AIPersonalityCreate,
        created_by: uuid.UUID
    ) -> AIPersonality:
        """Create a new AI personality"""
        try:
            # Validate required fields
            if not personality_data.name or not personality_data.description:
                raise ValueError("Name and description are required")
            
            # Check if name already exists
            existing = await db.execute(
                select(AIPersonalityModel)
                .where(AIPersonalityModel.name == personality_data.name)
            )
            if existing.scalar_one_or_none():
                raise ValueError(f"AI personality with name '{personality_data.name}' already exists")
            
            # Create new personality
            new_personality = AIPersonalityModel(
                name=personality_data.name,
                description=personality_data.description,
                system_prompt=personality_data.system_prompt,
                detailed_analysis_prompt=personality_data.detailed_analysis_prompt,
                suggestions_prompt=personality_data.suggestions_prompt,
                model_override=personality_data.model_override,
                temperature_override=personality_data.temperature_override,
                is_active=True,
                created_by=created_by
            )
            
            db.add(new_personality)
            await db.commit()
            await db.refresh(new_personality)
            
            logger.info(f"Created AI personality {new_personality.id} - {new_personality.name}")
            
            return AIPersonality(
                id=str(new_personality.id),
                name=new_personality.name,
                description=new_personality.description,
                detailed_analysis_prompt=new_personality.detailed_analysis_prompt or "",
                suggestions_prompt=new_personality.suggestions_prompt or "",
                model_override=new_personality.model_override,
                temperature_override=float(new_personality.temperature_override) if new_personality.temperature_override else None,
                is_default=getattr(new_personality, 'is_default', False),
                is_active=new_personality.is_active,
                created_at=new_personality.created_at.isoformat() if new_personality.created_at else "",
                updated_at=new_personality.updated_at.isoformat() if new_personality.updated_at else ""
            )
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating AI personality: {str(e)}")
            raise
    
    async def update_personality(
        self, 
        db: AsyncSession, 
        personality_id: uuid.UUID,
        personality_data: AIPersonalityUpdate
    ) -> Optional[AIPersonality]:
        """Update an existing AI personality"""
        try:
            # Get existing personality
            result = await db.execute(
                select(AIPersonalityModel)
                .where(AIPersonalityModel.id == personality_id)
            )
            personality = result.scalar_one_or_none()
            
            if not personality:
                return None
            
            # Check if new name conflicts with existing personalities
            if personality_data.name and personality_data.name != personality.name:
                existing = await db.execute(
                    select(AIPersonalityModel)
                    .where(
                        and_(
                            AIPersonalityModel.name == personality_data.name,
                            AIPersonalityModel.id != personality_id
                        )
                    )
                )
                if existing.scalar_one_or_none():
                    raise ValueError(f"AI personality with name '{personality_data.name}' already exists")
            
            # Update fields if provided
            update_data = {}
            if personality_data.name is not None:
                update_data['name'] = personality_data.name
            if personality_data.description is not None:
                update_data['description'] = personality_data.description
            if personality_data.is_active is not None:
                update_data['is_active'] = personality_data.is_active
            if personality_data.system_prompt is not None:
                update_data['system_prompt'] = personality_data.system_prompt
            if personality_data.detailed_analysis_prompt is not None:
                update_data['detailed_analysis_prompt'] = personality_data.detailed_analysis_prompt
            if personality_data.suggestions_prompt is not None:
                update_data['suggestions_prompt'] = personality_data.suggestions_prompt
            if personality_data.model_override is not None:
                update_data['model_override'] = personality_data.model_override
            if personality_data.temperature_override is not None:
                update_data['temperature_override'] = personality_data.temperature_override
            
            if update_data:
                update_data['updated_at'] = datetime.utcnow()
                
                await db.execute(
                    update(AIPersonalityModel)
                    .where(AIPersonalityModel.id == personality_id)
                    .values(**update_data)
                )
                await db.commit()
                
                # Refresh the personality
                await db.refresh(personality)
            
            logger.info(f"Updated AI personality {personality_id}")
            
            return AIPersonality(
                id=str(personality.id),
                name=personality.name,
                description=personality.description,
                detailed_analysis_prompt=personality.detailed_analysis_prompt or "",
                suggestions_prompt=personality.suggestions_prompt or "",
                model_override=personality.model_override,
                temperature_override=float(personality.temperature_override) if personality.temperature_override else None,
                is_default=getattr(personality, 'is_default', False),
                is_active=personality.is_active,
                created_at=personality.created_at.isoformat() if personality.created_at else "",
                updated_at=personality.updated_at.isoformat() if personality.updated_at else ""
            )
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating AI personality {personality_id}: {str(e)}")
            raise
    
    async def delete_personality(
        self, 
        db: AsyncSession, 
        personality_id: uuid.UUID
    ) -> bool:
        """Delete an AI personality"""
        try:
            # Check if personality exists
            result = await db.execute(
                select(AIPersonalityModel)
                .where(AIPersonalityModel.id == personality_id)
            )
            personality = result.scalar_one_or_none()
            
            if not personality:
                return False
            
            # Delete the personality
            await db.execute(
                delete(AIPersonalityModel)
                .where(AIPersonalityModel.id == personality_id)
            )
            await db.commit()
            
            logger.info(f"Deleted AI personality {personality_id}")
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting AI personality {personality_id}: {str(e)}")
            raise
    
    async def set_as_default(
        self, 
        db: AsyncSession, 
        personality_id: uuid.UUID
    ) -> Optional[AIPersonality]:
        """Set an AI personality as default (this is a placeholder - implement based on your business logic)"""
        try:
            # For now, we'll just ensure the personality exists and is active
            # You may want to add a 'is_default' field to the model in the future
            
            result = await db.execute(
                select(AIPersonalityModel)
                .where(AIPersonalityModel.id == personality_id)
            )
            personality = result.scalar_one_or_none()
            
            if not personality:
                return None
            
            # Ensure it's active
            if not personality.is_active:
                await db.execute(
                    update(AIPersonalityModel)
                    .where(AIPersonalityModel.id == personality_id)
                    .values(is_active=True, updated_at=datetime.utcnow())
                )
                await db.commit()
                await db.refresh(personality)
            
            logger.info(f"Set AI personality {personality_id} as default")
            
            return AIPersonality(
                id=str(personality.id),
                name=personality.name,
                description=personality.description,
                detailed_analysis_prompt=personality.detailed_analysis_prompt or "",
                suggestions_prompt=personality.suggestions_prompt or "",
                model_override=personality.model_override,
                temperature_override=float(personality.temperature_override) if personality.temperature_override else None,
                is_default=getattr(personality, 'is_default', False),
                is_active=personality.is_active,
                created_at=personality.created_at.isoformat() if personality.created_at else "",
                updated_at=personality.updated_at.isoformat() if personality.updated_at else ""
            )
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error setting AI personality {personality_id} as default: {str(e)}")
            raise


# Create a singleton instance
ai_personality_service = AIPersonalityService()

