from typing import Optional, Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import time
import json

from app.core.dependencies import get_db
from app.models.models import ChatSession, ChatMessage, MessageRole
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MessagePersistenceMiddleware:
    """
    Middleware to handle message persistence for chat endpoints.
    Matches TypeScript API messagePersistenceMiddleware functionality.
    """
    
    @staticmethod
    async def process_request(
        request: Request,
        call_next,
        db: AsyncSession
    ):
        """Process request with message persistence tracking"""
        start_time = time.time()
        
        try:
            # Get request body for message extraction
            if hasattr(request.state, "body"):
                body = request.state.body
            else:
                body = await request.body()
                request.state.body = body
            
            # Parse request data
            request_data = {}
            if body:
                try:
                    request_data = json.loads(body.decode())
                except json.JSONDecodeError:
                    pass
            
            # Store request info for later use
            request.state.message_data = {
                "timestamp": start_time,
                "question": request_data.get("question"),
                "session_id": request_data.get("sessionId"),
                "create_session": request_data.get("createSession", True)
            }
            
            # Call the actual endpoint
            response = await call_next(request)
            
            # Log the interaction
            processing_time = time.time() - start_time
            logger.info(f"Message processed in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in message persistence middleware: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Message persistence error", "details": str(e)}
            )


class SessionValidationMiddleware:
    """
    Middleware to validate chat sessions.
    Matches TypeScript API sessionValidationMiddleware functionality.
    """
    
    @staticmethod
    async def validate_session(
        request: Request,
        user_id: str,
        session_id: Optional[str],
        db: AsyncSession
    ) -> bool:
        """Validate if session exists and belongs to user"""
        if not session_id:
            return True  # No session to validate
        
        try:
            session_uuid = uuid.UUID(session_id)
            user_uuid = uuid.UUID(user_id)
            
            # Check if session exists and belongs to user
            from sqlalchemy import select
            query = select(ChatSession).where(
                ChatSession.id == session_uuid,
                ChatSession.user_id == user_uuid
            )
            result = await db.execute(query)
            session = result.scalar_one_or_none()
            
            return session is not None
            
        except (ValueError, Exception) as e:
            logger.warning(f"Session validation error: {str(e)}")
            return False
    
    @staticmethod
    async def process_request(
        request: Request,
        call_next,
        db: AsyncSession,
        user_id: str
    ):
        """Process request with session validation"""
        try:
            # Get session ID from request
            session_id = None
            if hasattr(request.state, "message_data"):
                session_id = request.state.message_data.get("session_id")
            
            # Validate session if provided
            if session_id:
                is_valid = await SessionValidationMiddleware.validate_session(
                    request, user_id, session_id, db
                )
                
                if not is_valid:
                    return JSONResponse(
                        status_code=status.HTTP_404_NOT_FOUND,
                        content={"error": "Chat session not found or access denied"}
                    )
            
            # Session is valid, proceed
            return await call_next(request)
            
        except Exception as e:
            logger.error(f"Error in session validation middleware: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Session validation error", "details": str(e)}
            )


class EnhancedChatMiddleware:
    """
    Combined middleware for enhanced chat functionality.
    Provides both message persistence and session validation.
    """
    
    @staticmethod
    async def ensure_chat_session(
        user_id: str,
        survey_ids: list,
        category: str,
        title: str,
        personality_id: Optional[str],
        db: AsyncSession
    ) -> str:
        """Ensure a chat session exists, create if needed"""
        try:
            from app.services import ai_service
            
            # Create optimized chat session
            session = await ai_service.create_optimized_chat_session(
                db=db,
                user_id=uuid.UUID(user_id),
                title=title,
                survey_ids=[uuid.UUID(sid) for sid in survey_ids],
                personality_id=uuid.UUID(personality_id) if personality_id else None
            )
            
            return str(session.id)
            
        except Exception as e:
            logger.error(f"Error ensuring chat session: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat session"
            )
    
    @staticmethod
    async def log_chat_interaction(
        session_id: str,
        question: str,
        response: str,
        metadata: Dict[str, Any],
        db: AsyncSession
    ):
        """Log chat interaction for tracking and analytics"""
        try:
            # Save user message
            user_message = ChatMessage(
                session_id=uuid.UUID(session_id),
                content=question,
                sender=MessageRole.USER,
                metadata={"timestamp": time.time()}
            )
            db.add(user_message)
            
            # Save AI response
            ai_message = ChatMessage(
                session_id=uuid.UUID(session_id),
                content=response,
                sender=MessageRole.ASSISTANT,
                metadata=metadata
            )
            db.add(ai_message)
            
            await db.commit()
            
        except Exception as e:
            logger.error(f"Error logging chat interaction: {str(e)}")
            # Don't raise exception here to avoid breaking the main flow


# Middleware factory functions to match FastAPI dependency pattern

async def message_persistence_middleware():
    """Factory for message persistence middleware"""
    return MessagePersistenceMiddleware()

async def session_validation_middleware():
    """Factory for session validation middleware"""
    return SessionValidationMiddleware()

async def enhanced_chat_middleware():
    """Factory for enhanced chat middleware"""
    return EnhancedChatMiddleware()