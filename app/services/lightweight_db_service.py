"""
Lightweight Database Service - Direct SQL Queries
================================================
Replaces heavy SQLAlchemy ORM with direct asyncpg connections for maximum performance.
Mimics TypeScript API's approach with raw SQL queries and minimal overhead.
"""

import asyncpg
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LightweightDBService:
    """
    Ultra-fast database service using direct asyncpg connections
    No ORM overhead, just raw SQL queries like the TypeScript API
    """
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        """Build asyncpg connection string from settings"""
        return f"postgresql://{settings.database_user}:{settings.database_password}@{settings.database_host}:{settings.database_port}/{settings.database_name}"
    
    async def initialize(self):
        """Initialize connection pool - lightweight configuration"""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=5,          # Minimal pool size
                max_size=10,         # Reduced from 20
                command_timeout=10,   # Quick timeout
                max_inactive_connection_lifetime=300,  # 5 minutes
            )
            logger.info("Lightweight DB pool initialized")
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute_query(self, query: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results as list of dicts"""
        async with self.get_connection() as conn:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)
            
            # Convert asyncpg Records to dicts
            return [dict(row) for row in rows]
    
    async def execute_command(self, query: str, params: List[Any] = None) -> str:
        """Execute INSERT/UPDATE/DELETE and return status"""
        try:
            async with self.get_connection() as conn:
                if params:
                    result = await conn.execute(query, *params)
                else:
                    result = await conn.execute(query)
                return result
        except Exception as e:
            logger.error(f"Database command execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise e
    
    async def execute_fetchrow(self, query: str, params: List[Any] = None) -> Optional[Dict[str, Any]]:
        """Execute query and return single row"""
        async with self.get_connection() as conn:
            if params:
                row = await conn.fetchrow(query, *params)
            else:
                row = await conn.fetchrow(query)
            
            return dict(row) if row else None
    
    # ==========================================
    # CHAT-SPECIFIC LIGHTWEIGHT OPERATIONS
    # ==========================================
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID - simple query"""
        query = "SELECT id, email, roleid as role, language FROM users WHERE id = $1"
        return await self.execute_fetchrow(query, [user_id])
    
    async def get_chat_session(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get chat session without messages - lightweight"""
        query = """
        SELECT id, user_id, title, survey_ids, category, personality_id, 
               selected_file_ids, created_at, updated_at
        FROM chat_sessions 
        WHERE id = $1 AND user_id = $2
        """
        return await self.execute_fetchrow(query, [session_id, user_id])
    
    async def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages for context - efficient query with full message data"""
        query = """
        SELECT id, session_id, content, sender, timestamp, 
               data_snapshot, confidence, personality_used, created_at
        FROM chat_messages 
        WHERE session_id = $1 
        ORDER BY created_at DESC 
        LIMIT $2
        """
        messages = await self.execute_query(query, [session_id, limit])
        # Reverse to get chronological order
        return list(reversed(messages))
    
    async def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all messages for a session with full details"""
        try:
            query = """
            SELECT id, session_id, content, sender, timestamp, 
                   data_snapshot, confidence, personality_used, created_at
            FROM chat_messages 
            WHERE session_id = $1 
            ORDER BY created_at ASC 
            LIMIT $2
            """
            messages = await self.execute_query(query, [session_id, limit])
            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {str(e)}")
            raise e
    
    async def save_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Save single message - fast insert"""
        try:
            message_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Extract metadata fields
            data_snapshot = None
            confidence = None
            personality_used = None
            
            if metadata:
                data_snapshot = metadata.get('data_snapshot')
                confidence = metadata.get('confidence') 
                personality_used = metadata.get('personality_used')
            
            # Convert to JSON strings for PostgreSQL JSONB columns
            import json
            data_snapshot_json = json.dumps(data_snapshot) if data_snapshot is not None else None
            confidence_json = json.dumps(confidence) if confidence is not None else None
            
            query = """
            INSERT INTO chat_messages (id, session_id, sender, content, data_snapshot, confidence, personality_used, created_at, timestamp)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $8)
            """
            params = [
                message_id,
                session_id,
                role,
                content,
                data_snapshot_json,
                confidence_json,
                personality_used,
                now
            ]
            await self.execute_command(query, params)
            logger.info(f"Successfully saved message {message_id} for session {session_id} with metadata: snapshot={bool(data_snapshot)}, confidence={bool(confidence)}")
            return message_id
        except Exception as e:
            logger.error(f"Failed to save message for session {session_id}: {str(e)}")
            logger.error(f"Message content length: {len(content) if content else 0}")
            logger.error(f"Sender: {role}")
            raise e
    
    async def save_message_pair(self, session_id: str, user_content: str, ai_content: str, 
                              ai_metadata: Dict[str, Any] = None) -> tuple[str, str]:
        """Save user and AI message in single transaction - optimal performance"""
        try:
            user_id = str(uuid.uuid4())
            ai_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Insert user message
                    await conn.execute("""
                        INSERT INTO chat_messages (id, session_id, sender, content, created_at)
                        VALUES ($1, $2, $3, $4, $5)
                    """, user_id, session_id, "user", user_content, now)
                    
                    # Insert AI message
                    await conn.execute("""
                        INSERT INTO chat_messages (id, session_id, sender, content, created_at)
                        VALUES ($1, $2, $3, $4, $5)
                    """, ai_id, session_id, "assistant", ai_content, now)
            
            logger.info(f"Successfully saved message pair for session {session_id}: user={user_id}, ai={ai_id}")
            return user_id, ai_id
        except Exception as e:
            logger.error(f"Failed to save message pair for session {session_id}: {str(e)}")
            logger.error(f"User content length: {len(user_content) if user_content else 0}")
            logger.error(f"AI content length: {len(ai_content) if ai_content else 0}")
            raise e
    
    async def update_session_title(self, session_id: str, title: str):
        """Update session title - simple update"""
        query = "UPDATE chat_sessions SET title = $1, updated_at = $2 WHERE id = $3"
        await self.execute_command(query, [title, datetime.utcnow(), session_id])
    
    async def create_chat_session(self, user_id: str, title: str = "New Chat", 
                                survey_ids: List[str] = None, category: str = None,
                                personality_id: str = None, selected_file_ids: List[str] = None) -> str:
        """Create new chat session - fast insert"""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Convert to proper arrays for PostgreSQL
        # For survey_ids: convert to list of strings (not UUID objects) 
        survey_array = survey_ids or []
        selected_files_array = selected_file_ids or []
        
        query = """
        INSERT INTO chat_sessions (id, user_id, title, survey_ids, category, 
                                 personality_id, selected_file_ids, created_at, updated_at)
        VALUES ($1, $2, $3, $4::text[], $5, $6, $7::text[], $8, $9)
        """
        params = [
            session_id,
            user_id,
            title,
            survey_array,
            category,
            personality_id,
            selected_files_array,
            now,
            now
        ]
        
        await self.execute_command(query, params)
        return session_id
    
    async def get_ai_personality(self, personality_id: str) -> Optional[Dict[str, Any]]:
        """Get AI personality - simple query"""
        query = """
        SELECT id, name, description, system_prompt, detailed_analysis_prompt
        FROM ai_personalities 
        WHERE id = $1
        """
        return await self.execute_fetchrow(query, [personality_id])
    
    async def get_chat_sessions_by_survey(self, user_id: str, survey_id: str, 
                                        limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat sessions for a specific survey and user"""
        query = """
        SELECT id, title, survey_ids, category, personality_id, created_at, updated_at
        FROM chat_sessions 
        WHERE user_id = $1 
        AND $2::uuid = ANY(survey_ids)
        ORDER BY updated_at DESC 
        LIMIT $3
        """
        return await self.execute_query(query, [user_id, survey_id, limit])
    
    async def get_user_chat_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user"""
        query = """
        SELECT id, title, survey_ids, category, personality_id, created_at, updated_at
        FROM chat_sessions 
        WHERE user_id = $1 
        ORDER BY updated_at DESC 
        LIMIT $2
        """
        return await self.execute_query(query, [user_id, limit])
    
    async def delete_chat_session(self, session_id: str, user_id: str) -> bool:
        """Delete a chat session and all its messages (user must own the session)"""
        try:
            # First verify the session exists and user owns it
            session = await self.get_chat_session(session_id, user_id)
            if not session:
                logger.warning(f"Cannot delete session {session_id}: not found or access denied for user {user_id}")
                return False
            
            # Delete messages first (foreign key constraint)
            delete_messages_query = "DELETE FROM chat_messages WHERE session_id = $1"
            await self.execute_query(delete_messages_query, [session_id])
            
            # Delete the session
            delete_session_query = "DELETE FROM chat_sessions WHERE id = $1 AND user_id = $2"
            result = await self.execute_query(delete_session_query, [session_id, user_id])
            
            logger.info(f"Successfully deleted chat session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chat session {session_id}: {e}")
            return False


# Global instance - singleton pattern like TypeScript API
lightweight_db = LightweightDBService()


async def get_lightweight_db() -> LightweightDBService:
    """Dependency injection for FastAPI"""
    if not lightweight_db.pool:
        await lightweight_db.initialize()
    return lightweight_db


# Background task for analytics (non-blocking)
async def log_chat_analytics_background(session_id: str, question: str, response_time: float, 
                                      search_strategy: str = None, total_matches: int = 0):
    """Log analytics in background - don't block response"""
    try:
        # Simplified analytics logging
        query = """
        INSERT INTO chat_analytics (session_id, question_length, response_time, 
                                  search_strategy, total_matches, created_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        params = [
            session_id,
            len(question),
            response_time,
            search_strategy or "direct",
            total_matches,
            datetime.utcnow()
        ]
        await lightweight_db.execute_command(query, params)
    except Exception as e:
        # Don't let analytics failures affect main flow
        logger.warning(f"Analytics logging failed: {e}")