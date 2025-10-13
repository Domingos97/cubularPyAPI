"""
Error logging service for detailed error tracking
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.utils.logging import get_logger

logger = get_logger(__name__)

class ErrorLoggingService:
    """Service for logging detailed error information"""
    
    @staticmethod
    async def log_chat_error(
        db: AsyncSession,
        error_type: str,
        error_message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        question: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a chat-related error to database for debugging and monitoring
        Returns the error_log_id for reference
        """
        try:
            error_log_id = str(uuid.uuid4())
            
            # Prepare error data
            error_data = {
                'error_type': error_type,
                'error_message': error_message,
                'user_id': user_id,
                'session_id': session_id,
                'question': question[:500] if question else None,  # Truncate long questions
                'details': details or {},
                'timestamp': datetime.utcnow().isoformat(),
                'error_log_id': error_log_id
            }
            
            # Log to application logs
            logger.error(f"Chat Error [{error_log_id}]: {error_type} - {error_message}", 
                        extra={'error_data': error_data})
            
            # Store in database (you can create an error_logs table later if needed)
            # For now, we'll just ensure comprehensive logging
            
            return error_log_id
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
            return "unknown"
    
    @staticmethod
    def format_error_for_frontend(
        error_type: str,
        user_message: str,
        error_log_id: str,
        show_details: bool = False,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format error information for frontend display
        """
        error_response = {
            'error': True,
            'error_type': error_type,
            'message': user_message,
            'error_id': error_log_id,
            'timestamp': datetime.utcnow().isoformat(),
            'support_message': f"If this issue persists, please contact support with error ID: {error_log_id}"
        }
        
        if show_details and details:
            error_response['details'] = details
            
        return error_response

# Create singleton instance
error_logging_service = ErrorLoggingService()