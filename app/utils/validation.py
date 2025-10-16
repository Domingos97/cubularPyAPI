"""
Input Validation Utilities
==========================
Simple validation helpers for API endpoints
"""

from typing import List
import uuid
import re

from app.utils.error_handlers import validation_error


class ValidationHelpers:
    """Common validation helper functions"""
    
    @staticmethod
    def validate_uuid(value: str, field_name: str = "id") -> uuid.UUID:
        """Validate and convert string to UUID"""
        try:
            return uuid.UUID(value)
        except (ValueError, TypeError):
            raise validation_error(f"Invalid {field_name}: must be a valid UUID")
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email.lower()):
            raise validation_error("Invalid email format")
        return email.lower()
    
    @staticmethod
    def validate_string_length(value: str, min_length: int = 1, max_length: int = 255, field_name: str = "field") -> str:
        """Validate string length"""
        if len(value) < min_length:
            raise validation_error(f"{field_name} must be at least {min_length} characters long")
        if len(value) > max_length:
            raise validation_error(f"{field_name} must be no more than {max_length} characters long")
        return value.strip()
    
    @staticmethod
    def validate_survey_ids(survey_ids: List[str]) -> List[uuid.UUID]:
        """Validate list of survey IDs"""
        if not survey_ids:
            raise validation_error("At least one survey ID is required")
        
        validated_ids = []
        for i, survey_id in enumerate(survey_ids):
            try:
                validated_ids.append(uuid.UUID(survey_id))
            except (ValueError, TypeError):
                raise validation_error(f"Invalid survey ID at index {i}: must be a valid UUID")
        
        return validated_ids
    
    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 10000) -> str:
        """Sanitize text input by removing dangerous characters"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()