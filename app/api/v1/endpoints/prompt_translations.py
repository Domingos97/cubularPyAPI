from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_, func
import uuid

from app.core.dependencies import get_current_user, get_current_admin_user, get_db
from app.models.models import User, PromptTranslation, AIPersonality, SupportedLanguage
from app.models.schemas import (
    PromptTranslationCreate, 
    PromptTranslationUpdate, 
    PromptTranslationResponse,
    SuccessResponse
)
from app.services.language_service import LanguageService

router = APIRouter()

# Backward compatibility endpoints for frontend
@router.get("/translations", response_model=List[PromptTranslationResponse])
async def get_prompt_translations_compat(
    personality_id: Optional[str] = Query(None, description="Filter by personality ID"),
    language_code: Optional[str] = Query(None, description="Filter by language code"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get prompt translations (backward compatibility endpoint)
    """
    # This endpoint matches the frontend expectation: GET /api/prompts/translations?personality_id=...&language_code=...
    try:
        query = select(PromptTranslation).options(
            selectinload(PromptTranslation.personality),
            selectinload(PromptTranslation.language)
        )
        
        filters = []
        if personality_id:
            filters.append(PromptTranslation.personality_id == uuid.UUID(personality_id))
        if language_code:
            # Join with SupportedLanguage to filter by language code
            query = query.join(SupportedLanguage)
            filters.append(SupportedLanguage.language_code == language_code)
        
        if filters:
            query = query.where(and_(*filters))
        
        result = await db.execute(query)
        translations = result.scalars().all()
        
        return [PromptTranslationResponse.model_validate(t) for t in translations]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch prompt translations: {str(e)}"
        )

@router.post("/translations", response_model=PromptTranslationResponse)
async def create_prompt_translation_compat(
    translation_data: PromptTranslationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Create a new prompt translation (backward compatibility)
    """
    try:
        new_translation = PromptTranslation(
            id=uuid.uuid4(),
            **translation_data.model_dump()
        )
        
        db.add(new_translation)
        await db.commit()
        await db.refresh(new_translation)
        
        return PromptTranslationResponse.model_validate(new_translation)
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create prompt translation: {str(e)}"
        )

@router.put("/translations/{translation_id}", response_model=PromptTranslationResponse)  
async def update_prompt_translation_compat(
    translation_id: uuid.UUID,
    update_data: PromptTranslationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Update prompt translation (backward compatibility)
    """
    try:
        query = select(PromptTranslation).where(PromptTranslation.id == translation_id)
        result = await db.execute(query)
        translation = result.scalar_one_or_none()
        
        if not translation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prompt translation not found"
            )
        
        # Update fields
        for field, value in update_data.model_dump(exclude_unset=True).items():
            setattr(translation, field, value)
        
        await db.commit()
        await db.refresh(translation)
        
        return PromptTranslationResponse.model_validate(translation)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update prompt translation: {str(e)}"
        )

@router.delete("/translations/{translation_id}", response_model=SuccessResponse)
async def delete_prompt_translation_compat(
    translation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Delete prompt translation (backward compatibility)
    """
    try:
        query = select(PromptTranslation).where(PromptTranslation.id == translation_id)
        result = await db.execute(query)
        translation = result.scalar_one_or_none()
        
        if not translation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prompt translation not found"
            )
        
        await db.delete(translation)
        await db.commit()
        
        return SuccessResponse(message="Prompt translation deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete prompt translation: {str(e)}"
        )

class PersonalityPromptTranslation(PromptTranslationCreate):
    """Extended prompt translation for personality-specific prompts"""
    personality_id: uuid.UUID
    prompt_type: str  # 'system', 'detailed_analysis', 'suggestions'

class PersonalityPromptTranslationUpdate(PromptTranslationUpdate):
    """Update schema for personality prompt translations"""
    prompt_type: Optional[str] = None

class PersonalityPromptTranslationResponse(PromptTranslationResponse):
    """Response schema with personality and prompt type info"""
    personality_id: uuid.UUID
    prompt_type: str
    personality_name: Optional[str] = None
    language_name: Optional[str] = None
    language_native_name: Optional[str] = None

@router.get("/personalities/{personality_id}/translations", 
           response_model=List[PersonalityPromptTranslationResponse], 
           tags=["prompt-translations"])
async def get_personality_translations(
    personality_id: str,
    language_code: Optional[str] = Query(None, description="Filter by language code"),
    prompt_type: Optional[str] = Query(None, description="Filter by prompt type"),
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all prompt translations for a specific AI personality (admin only)
    """
    try:
        personality_uuid = uuid.UUID(personality_id)
        
        # Build query with filters
        query = (
            select(PromptTranslation, AIPersonality, SupportedLanguage)
            .join(AIPersonality, PromptTranslation.personality_id == AIPersonality.id)
            .outerjoin(SupportedLanguage, PromptTranslation.language_code == SupportedLanguage.code)
            .where(PromptTranslation.personality_id == personality_uuid)
        )
        
        if language_code:
            query = query.where(PromptTranslation.language_code == language_code)
        
        if prompt_type:
            query = query.where(PromptTranslation.prompt_type == prompt_type)
        
        query = query.order_by(PromptTranslation.language_code, PromptTranslation.prompt_type)
        
        result = await db.execute(query)
        translations_data = result.fetchall()
        
        translations = []
        for translation, personality, language in translations_data:
            translation_dict = {
                **translation.__dict__,
                "personality_id": personality.id,
                "prompt_type": translation.prompt_type,
                "personality_name": personality.name if personality else None,
                "language_name": language.name if language else None,
                "language_native_name": language.native_name if language else None
            }
            translations.append(PersonalityPromptTranslationResponse(**translation_dict))
        
        return translations
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid personality ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch personality translations: {str(e)}"
        )

@router.post("/personalities/{personality_id}/translations", 
            response_model=PersonalityPromptTranslationResponse, 
            status_code=status.HTTP_201_CREATED,
            tags=["prompt-translations"])
async def create_personality_translation(
    personality_id: str,
    translation_data: PersonalityPromptTranslation,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new prompt translation for an AI personality (admin only)
    """
    try:
        personality_uuid = uuid.UUID(personality_id)
        
        # Verify personality exists
        personality_query = select(AIPersonality).where(AIPersonality.id == personality_uuid)
        personality_result = await db.execute(personality_query)
        personality = personality_result.scalar_one_or_none()
        
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI Personality not found"
            )
        
        # Check if translation already exists
        existing_query = (
            select(PromptTranslation)
            .where(
                and_(
                    PromptTranslation.personality_id == personality_uuid,
                    PromptTranslation.language_code == translation_data.language_code,
                    PromptTranslation.prompt_type == translation_data.prompt_type
                )
            )
        )
        
        existing_result = await db.execute(existing_query)
        existing_translation = existing_result.scalar_one_or_none()
        
        if existing_translation:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Translation already exists for this personality, language, and prompt type"
            )
        
        # Create new translation
        new_translation = PromptTranslation(
            personality_id=personality_uuid,
            prompt_key=f"{personality.name}_{translation_data.prompt_type}",
            language_code=translation_data.language_code,
            translated_text=translation_data.translated_text,
            context=translation_data.context,
            prompt_type=translation_data.prompt_type,
            created_by=current_user.id
        )
        
        db.add(new_translation)
        await db.commit()
        await db.refresh(new_translation)
        
        # Return with additional info
        response_data = {
            **new_translation.__dict__,
            "personality_id": personality.id,
            "prompt_type": translation_data.prompt_type,
            "personality_name": personality.name
        }
        
        return PersonalityPromptTranslationResponse(**response_data)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid personality ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create personality translation: {str(e)}"
        )

@router.put("/translations/{translation_id}", 
           response_model=PersonalityPromptTranslationResponse,
           tags=["prompt-translations"])
async def update_personality_translation(
    translation_id: str,
    translation_data: PersonalityPromptTranslationUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing prompt translation (admin only)
    """
    try:
        translation_uuid = uuid.UUID(translation_id)
        
        # Get translation with personality info
        query = (
            select(PromptTranslation, AIPersonality)
            .join(AIPersonality, PromptTranslation.personality_id == AIPersonality.id)
            .where(PromptTranslation.id == translation_uuid)
        )
        
        result = await db.execute(query)
        translation_data_result = result.first()
        
        if not translation_data_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Translation not found"
            )
        
        translation, personality = translation_data_result
        
        # Update fields
        if translation_data.translated_text is not None:
            translation.translated_text = translation_data.translated_text
        
        if translation_data.context is not None:
            translation.context = translation_data.context
        
        if translation_data.prompt_type is not None:
            translation.prompt_type = translation_data.prompt_type
        
        await db.commit()
        await db.refresh(translation)
        
        # Return with additional info
        response_data = {
            **translation.__dict__,
            "personality_id": personality.id,
            "prompt_type": translation.prompt_type,
            "personality_name": personality.name
        }
        
        return PersonalityPromptTranslationResponse(**response_data)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid translation ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update translation: {str(e)}"
        )

@router.delete("/translations/{translation_id}", 
              status_code=status.HTTP_204_NO_CONTENT,
              tags=["prompt-translations"])
async def delete_personality_translation(
    translation_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a prompt translation (admin only)
    """
    try:
        translation_uuid = uuid.UUID(translation_id)
        
        query = select(PromptTranslation).where(PromptTranslation.id == translation_uuid)
        result = await db.execute(query)
        translation = result.scalar_one_or_none()
        
        if not translation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Translation not found"
            )
        
        await db.delete(translation)
        await db.commit()
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid translation ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete translation: {str(e)}"
        )

@router.get("/personalities/{personality_id}/prompts/{language_code}", 
           response_model=Dict[str, str],
           tags=["prompt-translations"])
async def get_personality_prompts_by_language(
    personality_id: str,
    language_code: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all prompts for a personality in a specific language
    Returns a dictionary with prompt_type as key and translated_text as value
    """
    try:
        personality_uuid = uuid.UUID(personality_id)
        
        query = (
            select(PromptTranslation)
            .where(
                and_(
                    PromptTranslation.personality_id == personality_uuid,
                    PromptTranslation.language_code == language_code
                )
            )
        )
        
        result = await db.execute(query)
        translations = result.scalars().all()
        
        prompts = {
            translation.prompt_type: translation.translated_text
            for translation in translations
        }
        
        return prompts
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid personality ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch personality prompts: {str(e)}"
        )

@router.get("/translations/statistics", 
           response_model=Dict[str, Any],
           tags=["prompt-translations"])
async def get_translation_statistics(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get translation statistics (admin only)
    """
    try:
        # Total translations
        total_query = select(func.count(PromptTranslation.id))
        total_result = await db.execute(total_query)
        total_translations = total_result.scalar()
        
        # Translations by language
        lang_query = (
            select(
                PromptTranslation.language_code,
                func.count(PromptTranslation.id).label('count')
            )
            .group_by(PromptTranslation.language_code)
            .order_by(func.count(PromptTranslation.id).desc())
        )
        
        lang_result = await db.execute(lang_query)
        languages_stats = [
            {"language_code": row.language_code, "translation_count": row.count}
            for row in lang_result.fetchall()
        ]
        
        # Translations by prompt type
        type_query = (
            select(
                PromptTranslation.prompt_type,
                func.count(PromptTranslation.id).label('count')
            )
            .group_by(PromptTranslation.prompt_type)
            .order_by(func.count(PromptTranslation.id).desc())
        )
        
        type_result = await db.execute(type_query)
        type_stats = [
            {"prompt_type": row.prompt_type, "translation_count": row.count}
            for row in type_result.fetchall()
        ]
        
        # Personalities with translations
        personality_query = (
            select(func.count(func.distinct(PromptTranslation.personality_id)))
        )
        personality_result = await db.execute(personality_query)
        personalities_with_translations = personality_result.scalar()
        
        return {
            "total_translations": total_translations,
            "personalities_with_translations": personalities_with_translations,
            "translations_by_language": languages_stats,
            "translations_by_prompt_type": type_stats
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch translation statistics: {str(e)}"
        )