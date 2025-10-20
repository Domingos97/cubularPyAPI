from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_
from typing import List, Optional, Dict, Any
from app.models.models import SupportedLanguage, User
from app.models.schemas import (
    SupportedLanguageCreate, 
    SupportedLanguageUpdate,
    LanguageDetectionResult
)
import logging
import re

logger = logging.getLogger(__name__)

class LanguageService:
    """Service for managing languages and translations"""
    
    @staticmethod
    async def get_supported_languages(
        db: Session,
        enabled_only: bool = True,
        include_disabled: bool = False
    ) -> List[SupportedLanguage]:
        """Get list of supported languages"""
        try:
            query = db.query(SupportedLanguage)
            
            if enabled_only and not include_disabled:
                query = query.filter(SupportedLanguage.enabled == True)
            
            languages = query.order_by(SupportedLanguage.sort_order, SupportedLanguage.name).all()
            
            # If no languages in database, return defaults
            if not languages:
                return await LanguageService._create_default_languages(db)
            
            return languages
            
        except Exception as e:
            logger.error(f"Error getting supported languages: {str(e)}")
            return await LanguageService._create_default_languages(db)
    
    @staticmethod
    async def get_language_by_code(db: Session, code: str) -> Optional[SupportedLanguage]:
        """Get language by code"""
        try:
            return db.query(SupportedLanguage).filter(
                SupportedLanguage.code == code.lower()
            ).first()
        except Exception as e:
            logger.error(f"Error getting language by code {code}: {str(e)}")
            return None
    
    @staticmethod
    async def create_language(
        db: Session, 
        language_data: SupportedLanguageCreate
    ) -> SupportedLanguage:
        """Create new supported language"""
        try:
            # Check if language already exists
            existing = await LanguageService.get_language_by_code(db, language_data.code)
            if existing:
                raise ValueError(f"Language with code '{language_data.code}' already exists")
            
            language = SupportedLanguage(
                code=language_data.code.lower(),
                name=language_data.name,
                native_name=language_data.native_name,
                enabled=language_data.enabled,
                is_rtl=language_data.is_rtl,
                sort_order=language_data.sort_order
            )
            
            db.add(language)
            db.commit()
            db.refresh(language)
            
            logger.info(f"Created language: {language.code} - {language.name}")
            return language
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error creating language: {str(e)}")
            raise ValueError("Language code must be unique")
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating language: {str(e)}")
            raise
    
    @staticmethod
    async def update_language(
        db: Session,
        language_code: str,
        language_data: SupportedLanguageUpdate
    ) -> Optional[SupportedLanguage]:
        """Update existing language"""
        try:
            language = await LanguageService.get_language_by_code(db, language_code)
            if not language:
                return None
            
            # Update fields if provided
            update_data = language_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(language, field, value)
            
            db.commit()
            db.refresh(language)
            
            logger.info(f"Updated language: {language.code}")
            return language
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating language {language_code}: {str(e)}")
            raise
    
    @staticmethod
    async def delete_language(db: Session, language_code: str) -> bool:
        """Delete language (soft delete by disabling)"""
        try:
            language = await LanguageService.get_language_by_code(db, language_code)
            if not language:
                return False
            
            # Don't allow deletion of default language
            if language_code.lower() == 'en':
                raise ValueError("Cannot delete default English language")
            
            # Soft delete by disabling
            language.enabled = False
            db.commit()
            
            logger.info(f"Disabled language: {language.code}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deleting language {language_code}: {str(e)}")
            raise
    
    @staticmethod
    async def set_user_language_preference(
        db: Session,
        user_id: str,
        language_code: str
    ) -> bool:
        """Set user's preferred language"""
        try:
            # Verify language exists and is enabled
            language = await LanguageService.get_language_by_code(db, language_code)
            if not language or not language.enabled:
                raise ValueError(f"Language '{language_code}' is not available")
            
            # Update user's language preference
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            user.language = language_code.lower()
            db.commit()
            
            logger.info(f"Set user {user_id} language to {language_code}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error setting user language preference: {str(e)}")
            raise
    
    @staticmethod
    async def detect_language(text: str) -> LanguageDetectionResult:
        """Simple language detection using basic patterns"""
        try:
            if not text or len(text.strip()) < 3:
                return LanguageDetectionResult(language="en", confidence=0.5)
            
            text = text.lower().strip()
            
            # Basic language detection patterns
            language_patterns = {
                'es': [
                    r'\b(el|la|los|las|un|una|de|en|y|que|es|por|para|con|su|se|no|te|le|da|ya|va|ha|si|al)\b',
                    r'[ñáéíóúü]',
                    r'\b(hola|gracias|por favor|buenos días|buenas tardes|buenas noches)\b'
                ],
                'pt': [
                    r'\b(o|a|os|as|um|uma|de|em|e|que|é|por|para|com|seu|sua|se|não|te|lhe|já|vai|há|se|ao)\b',
                    r'[ãõçáéíóúâêîôû]',
                    r'\b(olá|obrigado|obrigada|por favor|bom dia|boa tarde|boa noite)\b'
                ],
                'sv': [
                    r'\b(den|det|de|en|ett|och|att|är|för|med|på|av|till|från|som|var|kan|ska|inte|om|när)\b',
                    r'[åäöé]',
                    r'\b(hej|tack|ursäkta|god morgon|god kväll|god natt)\b'
                ],
                'fr': [
                    r'\b(le|la|les|un|une|de|du|des|et|que|est|pour|avec|son|sa|se|ne|te|lui|ça|va|a|si|au)\b',
                    r'[àâäéèêëïîôöùûüÿç]',
                    r'\b(bonjour|merci|s\'il vous plaît|bonsoir|bonne nuit|salut)\b'
                ]
            }
            
            language_scores = {}
            
            for lang, patterns in language_patterns.items():
                score = 0
                total_patterns = len(patterns)
                
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    if matches > 0:
                        score += min(matches / 10, 1.0)  # Normalize to max 1.0 per pattern
                
                language_scores[lang] = score / total_patterns
            
            # Default to English if no clear match
            if not language_scores or max(language_scores.values()) < 0.2:
                return LanguageDetectionResult(language="en", confidence=0.6)
            
            # Return language with highest score
            detected_lang = max(language_scores, key=language_scores.get)
            confidence = min(language_scores[detected_lang], 1.0)
            
            return LanguageDetectionResult(
                language=detected_lang, 
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return LanguageDetectionResult(language="en", confidence=0.5)
    
    @staticmethod
    async def _create_default_languages(db: Session) -> List[SupportedLanguage]:
        """Create default languages if none exist"""
        try:
            default_languages = [
                {"code": "en", "name": "English", "native_name": "English", "sort_order": 1},
                {"code": "es", "name": "Spanish", "native_name": "Español", "sort_order": 2},
                {"code": "pt", "name": "Portuguese", "native_name": "Português", "sort_order": 3},
                {"code": "sv", "name": "Swedish", "native_name": "Svenska", "sort_order": 4},
                {"code": "fr", "name": "French", "native_name": "Français", "sort_order": 5},
                {"code": "de", "name": "German", "native_name": "Deutsch", "sort_order": 6},
                {"code": "it", "name": "Italian", "native_name": "Italiano", "sort_order": 7},
                {"code": "ru", "name": "Russian", "native_name": "Русский", "sort_order": 8}
            ]
            
            created_languages = []
            for lang_data in default_languages:
                # Check if language already exists
                existing = db.query(SupportedLanguage).filter(
                    SupportedLanguage.code == lang_data["code"]
                ).first()
                
                if not existing:
                    language = SupportedLanguage(**lang_data)
                    db.add(language)
                    created_languages.append(language)
            
            if created_languages:
                db.commit()
                for lang in created_languages:
                    db.refresh(lang)
                logger.info(f"Created {len(created_languages)} default languages")
            
            # Return all languages
            return db.query(SupportedLanguage).filter(
                SupportedLanguage.enabled == True
            ).order_by(SupportedLanguage.sort_order).all()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating default languages: {str(e)}")
            return []
