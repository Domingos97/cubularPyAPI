from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_admin_user
from app.models.models import User
from app.models.schemas import (
    SupportedLanguageResponse,
    SupportedLanguageCreate,
    SupportedLanguageUpdate,
    PromptTranslationResponse,
    PromptTranslationCreate,
    PromptTranslationUpdate,
    LanguageDetectionResult,
    LanguageConfigResponse,
    UserLanguagePreference
)
from app.services.language_service import LanguageService, PromptTranslationService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# Root language endpoint to match TypeScript API
@router.get("/")
async def list_languages(
    db: Session = Depends(get_db)
):
    """
    List all available languages (matches TypeScript API root endpoint)
    """
    try:
        languages = await LanguageService.get_supported_languages(db, enabled_only=False)
        
        return {
            "languages": [
                {
                    "code": lang.code,
                    "name": lang.name,
                    "native_name": lang.native_name,
                    "enabled": lang.enabled,
                    "is_rtl": lang.is_rtl
                }
                for lang in languages
            ],
            "total_languages": len(languages)
        }
        
    except Exception as e:
        logger.error(f"Error getting languages: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve languages"
        )


# Public language endpoints (no authentication required)
@router.get("/enabled", response_model=Dict[str, Any])
async def get_enabled_languages(
    db: Session = Depends(get_db)
):
    """
    Get list of enabled/supported languages.
    Public endpoint for frontend language selection.
    """
    try:
        languages = await LanguageService.get_supported_languages(db, enabled_only=True)
        
        return {
            "languages": [
                {
                    "code": lang.code,
                    "name": lang.name,
                    "native_name": lang.native_name,
                    "is_rtl": lang.is_rtl
                }
                for lang in languages
            ],
            "total_languages": len(languages)
        }
        
    except Exception as e:
        logger.error(f"Error getting enabled languages: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch enabled languages"
        )


@router.get("/config", response_model=LanguageConfigResponse)
async def get_language_config(
    db: Session = Depends(get_db)
):
    """
    Get current language configuration including default language and available languages.
    Public endpoint for frontend configuration.
    """
    try:
        languages = await LanguageService.get_supported_languages(db, enabled_only=True)
        
        return LanguageConfigResponse(
            default_language="en",
            available_languages=[lang.code for lang in languages],
            languages=languages
        )
        
    except Exception as e:
        logger.error(f"Error getting language config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch language configuration"
        )


@router.post("/detect", response_model=LanguageDetectionResult)
async def detect_language(
    text: str,
    db: Session = Depends(get_db)
):
    """
    Detect language of given text.
    Public endpoint for language detection.
    """
    try:
        if not text or len(text.strip()) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text is required for language detection"
            )
        
        result = await LanguageService.detect_language(text)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect language"
        )


# User language preference endpoints (authentication required)
@router.post("/user/preference")
async def set_user_language_preference(
    preference: UserLanguagePreference,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Set user's preferred language.
    Requires authentication.
    """
    try:
        success = await LanguageService.set_user_language_preference(
            db, str(current_user.id), preference.language_code
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to set language preference"
            )
        
        return {
            "message": f"Language preference set to {preference.language_code}",
            "language_code": preference.language_code
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error setting user language preference: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set language preference"
        )


@router.get("/user/preference")
async def get_user_language_preference(
    current_user: User = Depends(get_current_user)
):
    """
    Get user's current language preference.
    Requires authentication.
    """
    try:
        return {
            "language_code": current_user.language,
            "message": f"Current language preference: {current_user.language}"
        }
        
    except Exception as e:
        logger.error(f"Error getting user language preference: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get language preference"
        )


# Admin language management endpoints (admin authentication required)
@router.get("/admin/all", response_model=List[SupportedLanguageResponse])
async def get_all_languages(
    include_disabled: bool = Query(False, description="Include disabled languages"),
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Get all languages including disabled ones.
    Admin only endpoint.
    """
    try:
        languages = await LanguageService.get_supported_languages(
            db, enabled_only=False, include_disabled=include_disabled
        )
        return languages
        
    except Exception as e:
        logger.error(f"Error getting all languages: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch languages"
        )


@router.post("/admin", response_model=SupportedLanguageResponse)
async def create_language(
    language_data: SupportedLanguageCreate,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Create new supported language.
    Admin only endpoint.
    """
    try:
        language = await LanguageService.create_language(db, language_data)
        return language
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating language: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create language"
        )


@router.put("/admin/{language_code}", response_model=SupportedLanguageResponse)
async def update_language(
    language_code: str,
    language_data: SupportedLanguageUpdate,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update existing language.
    Admin only endpoint.
    """
    try:
        language = await LanguageService.update_language(db, language_code, language_data)
        
        if not language:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Language '{language_code}' not found"
            )
        
        return language
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating language: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update language"
        )


@router.delete("/admin/{language_code}")
async def delete_language(
    language_code: str,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete (disable) language.
    Admin only endpoint.
    """
    try:
        success = await LanguageService.delete_language(db, language_code)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Language '{language_code}' not found"
            )
        
        return {"message": f"Language '{language_code}' has been disabled"}
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error deleting language: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete language"
        )


# Prompt translation endpoints
@router.get("/translations", response_model=List[PromptTranslationResponse])
async def get_prompt_translations(
    prompt_key: Optional[str] = Query(None, description="Filter by prompt key"),
    language_code: Optional[str] = Query(None, description="Filter by language code"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get prompt translations with optional filtering.
    Requires authentication.
    """
    try:
        translations = await PromptTranslationService.get_prompt_translations(
            db, prompt_key, language_code, limit, offset
        )
        return translations
        
    except Exception as e:
        logger.error(f"Error getting prompt translations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch prompt translations"
        )


@router.get("/translations/{language_code}", response_model=Dict[str, str])
async def get_translations_by_language(
    language_code: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all translations for a specific language as key-value pairs.
    Requires authentication.
    """
    try:
        translations = await PromptTranslationService.get_translations_by_language(
            db, language_code
        )
        return translations
        
    except Exception as e:
        logger.error(f"Error getting translations by language: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch translations"
        )


@router.post("/translations", response_model=PromptTranslationResponse)
async def create_prompt_translation(
    translation_data: PromptTranslationCreate,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Create new prompt translation.
    Admin only endpoint.
    """
    try:
        translation = await PromptTranslationService.create_prompt_translation(
            db, translation_data, str(current_admin.id)
        )
        return translation
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating prompt translation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create prompt translation"
        )


@router.put("/translations/{prompt_key}/{language_code}", response_model=PromptTranslationResponse)
async def update_prompt_translation(
    prompt_key: str,
    language_code: str,
    translation_data: PromptTranslationUpdate,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Update existing prompt translation.
    Admin only endpoint.
    """
    try:
        translation = await PromptTranslationService.update_prompt_translation(
            db, prompt_key, language_code, translation_data
        )
        
        if not translation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Translation not found for key '{prompt_key}' in language '{language_code}'"
            )
        
        return translation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating prompt translation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update prompt translation"
        )


@router.delete("/translations/{prompt_key}/{language_code}")
async def delete_prompt_translation(
    prompt_key: str,
    language_code: str,
    current_admin: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete prompt translation.
    Admin only endpoint.
    """
    try:
        success = await PromptTranslationService.delete_prompt_translation(
            db, prompt_key, language_code
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Translation not found for key '{prompt_key}' in language '{language_code}'"
            )
        
        return {"message": f"Translation deleted for key '{prompt_key}' in language '{language_code}'"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt translation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete prompt translation"
        )