"""
Error logging service for detailed error tracking
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.models.models import Log
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
            # Create log entry in database
            log_entry = Log(
                level="ERROR",
                action=f"chat_error_{error_type}",
                user_id=uuid.UUID(user_id) if user_id else None,
                session_id=uuid.UUID(session_id) if session_id else None,
                error_message=error_message,
                details={
                    'error_type': error_type,
                    'question': question[:500] if question else None,  # Truncate long questions
                    'details': details or {}
                },
                timestamp=datetime.utcnow()
                # Note: Removed ip_address=None to avoid INET type issues
            )
            
            db.add(log_entry)
            await db.commit()
            await db.refresh(log_entry)
            
            # Log to console for reference (not file)
            logger.error(f"Chat Error [{log_entry.id}]: {error_type} - {error_message}", 
                        extra={'error_data': {
                            'error_type': error_type,
                            'user_id': user_id,
                            'session_id': session_id,
                            'question': question,
                            'details': details
                        }})
            
            return str(log_entry.id)
            
        except Exception as e:
            # Log to console only if database logging fails
            logger.error(f"Failed to log error to database: {e}")
            logger.error(f"Original error: {error_type} - {error_message}")
            return "db_logging_failed"
    
    @staticmethod
    async def log_api_error(
        db: AsyncSession,
        error_message: str,
        endpoint: str,
        method: str,
        status_code: int,
        user_id: Optional[str] = None,
        request_body: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None
    ) -> str:
        """
        Log API-related errors to database
        Returns the log entry ID for reference
        """
        try:
            log_entry = Log(
                level="ERROR",
                action=f"api_error_{method}_{endpoint}",
                user_id=uuid.UUID(user_id) if user_id else None,
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                error_message=error_message,
                request_body=request_body,
                stack_trace=stack_trace,
                timestamp=datetime.utcnow()
                # Note: Removed ip_address=None to avoid INET type issues
            )
            
            db.add(log_entry)
            await db.commit()
            await db.refresh(log_entry)
            
            # Log to console for reference (not file)
            logger.error(f"API Error [{log_entry.id}]: {method} {endpoint} - {error_message}")
            
            return str(log_entry.id)
            
        except Exception as e:
            logger.error(f"Failed to log API error to database: {e}")
            logger.error(f"Original error: {error_message}")
            return "db_logging_failed"
    
    @staticmethod
    async def log_general_error(
        db: AsyncSession,
        error_message: str,
        error_type: str = "general_error",
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None
    ) -> str:
        """
        Log general application errors to database
        Returns the log entry ID for reference
        """
        try:
            log_entry = Log(
                level="ERROR",
                action=error_type,
                user_id=uuid.UUID(user_id) if user_id else None,
                error_message=error_message,
                details=details,
                stack_trace=stack_trace,
                timestamp=datetime.utcnow()
                # Note: Removed ip_address=None to avoid INET type issues
            )
            
            db.add(log_entry)
            await db.commit()
            await db.refresh(log_entry)
            
            # Log to console for reference (not file)
            logger.error(f"General Error [{log_entry.id}]: {error_type} - {error_message}")
            
            return str(log_entry.id)
            
        except Exception as e:
            logger.error(f"Failed to log general error to database: {e}")
            logger.error(f"Original error: {error_message}")
            return "db_logging_failed"
    
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