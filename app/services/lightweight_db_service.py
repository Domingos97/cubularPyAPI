"""
Lightweight Database Service - Direct SQL Queries
================================================
Replaces heavy SQLAlchemy ORM with direct asyncpg connections for maximum performance.
Mimics TypeScript API's approach with raw SQL queries and minimal overhead.
"""

import asyncpg
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
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
    
    async def _setup_connection(self, conn):
        """Setup individual connection for optimal performance with Phase 2C enhancements"""
        # PERFORMANCE OPTIMIZATION: Connection-level settings for AI chat workload
        await conn.execute("SET enable_seqscan = off")  # Prefer index scans
        await conn.execute("SET random_page_cost = 1.1")  # Optimize for SSD
        await conn.execute("SET effective_cache_size = '1GB'")  # Assume decent cache
        await conn.execute("SET work_mem = '16MB'")  # More memory for sorts/hashes
        await conn.execute("SET maintenance_work_mem = '64MB'")  # Faster index ops
        # NOTE: shared_preload_libraries requires server restart - commented out to prevent runtime errors
        # await conn.execute("SET shared_preload_libraries = 'pg_stat_statements'")  # Query stats
        
        # Phase 2C: Enhanced connection settings for configuration queries
        await conn.execute("SET statement_timeout = '30s'")  # Prevent long-running queries
        await conn.execute("SET lock_timeout = '10s'")  # Prevent lock waits
        await conn.execute("SET idle_in_transaction_session_timeout = '60s'")  # Clean up idle transactions
        await conn.execute("SET log_min_duration_statement = 1000")  # Log slow queries (1s+)
    
    async def initialize(self):
        """Initialize connection pool - OPTIMIZED for high performance concurrent operations"""
        if self.pool is None:
            # PERFORMANCE OPTIMIZATION: Enhanced pool settings for concurrent AI chat requests
            self.pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=5,           # Increased for concurrent chat requests
                max_size=15,          # Higher max for burst capacity 
                command_timeout=10,   # Reasonable timeout for complex queries
                max_inactive_connection_lifetime=300,  # 5 minutes for better reuse
                max_queries=50000,    # High query limit per connection
                max_cached_statement_lifetime=900,  # 15 min cached statement lifetime
                setup=self._setup_connection  # Connection-level optimizations
            )
            logger.info("🚀 OPTIMIZED DB pool initialized: 5-15 connections, enhanced for AI chat concurrency")
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool performance statistics"""
        if not self.pool:
            return {"error": "Pool not initialized"}
        
        return {
            "min_size": self.pool.get_min_size(),
            "max_size": self.pool.get_max_size(),
            "current_size": self.pool.get_size(),
            "idle_connections": self.pool.get_idle_size(),
            "status": "healthy" if self.pool.get_size() > 0 else "degraded"
        }
    
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
        """Execute query and return single row - optimized for speed"""
        async with self.get_connection() as conn:
            if params:
                row = await conn.fetchrow(query, *params)
            else:
                row = await conn.fetchrow(query)
            
            return dict(row) if row else None
    
    # ==========================================
    # CHAT-SPECIFIC LIGHTWEIGHT OPERATIONS
    # ==========================================
    
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
        """Save user and AI message in single transaction with proper column separation"""
        try:
            logger.info(f"🔵 save_message_pair called for session {session_id}")
            
            user_id = str(uuid.uuid4())
            ai_id = str(uuid.uuid4())
            now = datetime.utcnow()
            # Ensure AI message has a later timestamp for proper chronological ordering
            from datetime import timedelta
            ai_time = now + timedelta(microseconds=1000)
            
            # Extract structured data for separate columns
            data_snapshot = None
            confidence = None
            personality_used = None
            
            if ai_metadata:
                data_snapshot_raw = ai_metadata.get('data_snapshot')
                confidence_raw = ai_metadata.get('confidence')
                personality_id_raw = ai_metadata.get('personality_used')
                
                # Convert to JSON strings for JSONB columns
                import json
                if data_snapshot_raw is not None:
                    if isinstance(data_snapshot_raw, (dict, list)):
                        data_snapshot = json.dumps(data_snapshot_raw)
                    else:
                        data_snapshot = str(data_snapshot_raw)
                
                if confidence_raw is not None:
                    if isinstance(confidence_raw, (dict, list)):
                        confidence = json.dumps(confidence_raw)
                    else:
                        confidence = str(confidence_raw)
                
                logger.info(f"🔵 Extracted data: snapshot={bool(data_snapshot)}, confidence={bool(confidence)}")
                
                # Convert personality ID to UUID if needed
                if personality_id_raw:
                    try:
                        import uuid as uuid_module
                        if isinstance(personality_id_raw, str):
                            personality_used = uuid_module.UUID(personality_id_raw)
                        else:
                            personality_used = personality_id_raw
                    except (ValueError, TypeError):
                        personality_used = None
            
            logger.info(f"🔵 About to insert messages to database")
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Insert user message
                    await conn.execute("""
                        INSERT INTO chat_messages (id, session_id, sender, content, created_at, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $5)
                    """, user_id, session_id, "user", user_content, now)
                    
                    # Insert AI message with structured data in separate columns
                    # Use ::jsonb cast for proper JSONB handling
                    await conn.execute("""
                        INSERT INTO chat_messages (id, session_id, sender, content, data_snapshot, confidence, personality_used, created_at, timestamp)
                        VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $8)
                    """, ai_id, session_id, "assistant", ai_content, data_snapshot, confidence, personality_used, ai_time)
            
            logger.info(f"✅ Successfully saved message pair for session {session_id}")
            return user_id, ai_id
            
        except Exception as e:
            logger.error(f"Failed to save message pair for session {session_id}: {str(e)}")
            raise e
    
    async def update_session_title(self, session_id: str, title: str):
        """Update session title - simple update"""
        query = "UPDATE chat_sessions SET title = $1, updated_at = $2 WHERE id = $3"
        await self.execute_command(query, [title, datetime.utcnow(), session_id])
    
    async def create_chat_session(self, user_id: str, title: str = "New Chat", 
                                survey_ids: List[str] = None, category: str = None,
                                personality_id: str = None, selected_file_ids: List[str] = None) -> str:
        """Create new chat session - fast insert"""
        try:
            session_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Convert to proper arrays for PostgreSQL
            survey_array = survey_ids or []
            selected_files_array = selected_file_ids or []
            
            # Ensure user_id and personality_id are strings
            user_id_str = str(user_id) if user_id else None
            personality_id_str = str(personality_id) if personality_id else None
            
            # Log the incoming parameters for debugging
            logger.info(f"Creating chat session: session_id={session_id}, user_id={user_id_str}, "
                       f"title={title}, survey_ids={survey_array}, category={category}, "
                       f"personality_id={personality_id_str}, selected_file_ids={selected_files_array}")
            
            try:
                # Convert UUIDs properly - user_id and personality_id are UUID columns
                user_uuid = uuid.UUID(user_id_str) if user_id_str else None
                logger.info(f"Converted user_id to UUID: {user_uuid}")
            except ValueError as e:
                logger.error(f"Invalid user_id UUID format: {user_id_str}, error: {e}")
                raise ValueError(f"Invalid user_id format: {user_id_str}")
            
            try:
                personality_uuid = uuid.UUID(personality_id_str) if personality_id_str else None
                logger.info(f"Converted personality_id to UUID: {personality_uuid}")
            except ValueError as e:
                logger.error(f"Invalid personality_id UUID format: {personality_id_str}, error: {e}")
                # For personality_id, we can set it to None if invalid
                personality_uuid = None
                logger.info(f"Set personality_id to None due to invalid format")
            
            session_uuid = uuid.UUID(session_id)
            
            query = """
            INSERT INTO chat_sessions (id, user_id, title, survey_ids, category, 
                                     personality_id, selected_file_ids, created_at, updated_at)
            VALUES ($1, $2, $3, $4::text[], $5, $6, $7::text[], $8, $9)
            """
            params = [
                session_uuid,  # UUID type
                user_uuid,     # UUID type
                title,
                survey_array,
                category,
                personality_uuid,  # UUID type or None
                selected_files_array,
                now,
                now
            ]
            
            logger.info(f"Executing query with params types: {[type(p).__name__ for p in params]}")
            await self.execute_command(query, params)
            logger.info(f"Successfully created chat session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating chat session: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Parameters: user_id={user_id_str}, title={title}, survey_ids={survey_ids}, "
                        f"category={category}, personality_id={personality_id_str}, selected_file_ids={selected_file_ids}")
            raise e
    
    async def get_ai_personality(self, personality_id: str) -> Optional[Dict[str, Any]]:
        """Get AI personality - simple query"""
        query = """
        SELECT id, name, description, is_active, detailed_analysis_prompt, suggestions_prompt, 
               created_by, created_at, updated_at, is_default
        FROM ai_personalities WHERE id = $1
        """
        return await self.execute_fetchrow(query, [personality_id])

    async def get_all_ai_personalities(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all AI personalities with pagination"""
        query = """
        SELECT id, name, description, is_active, detailed_analysis_prompt, suggestions_prompt, 
               created_by, created_at, updated_at, is_default
        FROM ai_personalities
        ORDER BY created_at DESC
        OFFSET $1 LIMIT $2
        """
        return await self.execute_query(query, [skip, limit])

    async def get_active_ai_personalities(self) -> List[Dict[str, Any]]:
        """Get only active AI personalities"""
        query = """
        SELECT id, name, description, is_active, detailed_analysis_prompt, suggestions_prompt, 
               created_by, created_at, updated_at, is_default
        FROM ai_personalities 
        WHERE is_active = true
        ORDER BY created_at DESC
        """
        return await self.execute_query(query)

    async def create_ai_personality(self, personality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new AI personality"""
        query = """
        INSERT INTO ai_personalities (name, description, detailed_analysis_prompt, suggestions_prompt,
                                     is_active, created_by, is_default)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id, name, description, is_active, detailed_analysis_prompt, suggestions_prompt,
                  created_by, created_at, updated_at, is_default
        """
        params = [
            personality_data['name'],
            personality_data.get('description', ''),
            personality_data.get('detailed_analysis_prompt', ''),
            personality_data.get('suggestions_prompt', ''),
            personality_data.get('is_active', True),
            personality_data.get('created_by'),
            personality_data.get('is_default', False)
        ]
        return await self.execute_fetchrow(query, params)

    async def update_ai_personality(self, personality_id: str, personality_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing AI personality"""
        # Build update query dynamically based on provided fields
        set_clauses = []
        params = []
        param_count = 1
        
        if 'name' in personality_data:
            set_clauses.append(f"name = ${param_count}")
            params.append(personality_data['name'])
            param_count += 1
            
        if 'description' in personality_data:
            set_clauses.append(f"description = ${param_count}")
            params.append(personality_data['description'])
            param_count += 1
            
        if 'detailed_analysis_prompt' in personality_data:
            set_clauses.append(f"detailed_analysis_prompt = ${param_count}")
            params.append(personality_data['detailed_analysis_prompt'])
            param_count += 1
            
        if 'suggestions_prompt' in personality_data:
            set_clauses.append(f"suggestions_prompt = ${param_count}")
            params.append(personality_data['suggestions_prompt'])
            param_count += 1
            
        if 'is_active' in personality_data:
            set_clauses.append(f"is_active = ${param_count}")
            params.append(personality_data['is_active'])
            param_count += 1
            
        if 'is_default' in personality_data:
            set_clauses.append(f"is_default = ${param_count}")
            params.append(personality_data['is_default'])
            param_count += 1
        
        # Add personality_id as the last parameter
        params.append(personality_id)
        
        query = f"""
        UPDATE ai_personalities 
        SET {', '.join(set_clauses)}, updated_at = NOW()
        WHERE id = ${param_count}
        RETURNING id, name, description, is_active, detailed_analysis_prompt, suggestions_prompt,
                  created_by, created_at, updated_at, is_default
        """
        
        return await self.execute_fetchrow(query, params)

    async def delete_ai_personality(self, personality_id: str) -> bool:
        """Delete an AI personality"""
        query = "DELETE FROM ai_personalities WHERE id = $1"
        result = await self.execute_query(query, [personality_id])
        return len(result) > 0 if result else False

    async def set_ai_personality_as_default(self, personality_id: str) -> Dict[str, Any]:
        """Set an AI personality as default (unset others first)"""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # First, unset all other defaults
                await conn.execute("UPDATE ai_personalities SET is_default = false")
                
                # Then set the specified one as default
                result = await conn.fetchrow("""
                    UPDATE ai_personalities 
                    SET is_default = true, updated_at = NOW()
                    WHERE id = $1
                    RETURNING id, name, description, is_active, detailed_analysis_prompt, suggestions_prompt,
                              created_by, created_at, updated_at, is_default
                """, personality_id)
                
                return dict(result) if result else None

    # ==========================================
    # PLANS METHODS
    # ==========================================
    
    async def get_all_plans(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all plans with pagination"""
        query = """
        SELECT id, name, display_name, description, price, currency, billing, features,
               max_surveys, max_responses, priority_support, api_access, is_active,
               created_at, updated_at
        FROM plans
        ORDER BY created_at DESC
        OFFSET $1 LIMIT $2
        """
        return await self.execute_query(query, [skip, limit])

    async def get_plan_by_id(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get plan by ID"""
        query = """
        SELECT id, name, display_name, description, price, currency, billing, features,
               max_surveys, max_responses, priority_support, api_access, is_active,
               created_at, updated_at
        FROM plans WHERE id = $1
        """
        return await self.execute_fetchrow(query, [plan_id])

    async def get_active_plans(self) -> List[Dict[str, Any]]:
        """Get only active plans"""
        query = """
        SELECT id, name, display_name, description, price, currency, billing, features,
               max_surveys, max_responses, priority_support, api_access, is_active,
               created_at, updated_at
        FROM plans 
        WHERE is_active = true
        ORDER BY price ASC
        """
        return await self.execute_query(query)

    async def create_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new plan"""
        import json
        query = """
        INSERT INTO plans (name, display_name, description, price, currency, billing, features,
                          max_surveys, max_responses, priority_support, api_access, is_active)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id, name, display_name, description, price, currency, billing, features,
                  max_surveys, max_responses, priority_support, api_access, is_active,
                  created_at, updated_at
        """
        
        # Ensure features is properly serialized as JSON
        features_json = json.dumps(plan_data.get('features', [])) if plan_data.get('features') else '[]'
        
        params = [
            plan_data['name'],
            plan_data.get('display_name'),
            plan_data.get('description'),
            plan_data.get('price'),
            plan_data.get('currency', 'USD'),
            plan_data.get('billing'),
            features_json,
            plan_data.get('max_surveys'),
            plan_data.get('max_responses'),
            plan_data.get('priority_support', False),
            plan_data.get('api_access', False),
            plan_data.get('is_active', True)
        ]
        
        return await self.execute_fetchrow(query, params)

    async def update_plan(self, plan_id: str, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing plan"""
        import json
        
        # Build update query dynamically based on provided fields
        set_clauses = []
        params = []
        param_count = 1
        
        if 'name' in plan_data:
            set_clauses.append(f"name = ${param_count}")
            params.append(plan_data['name'])
            param_count += 1
            
        if 'display_name' in plan_data:
            set_clauses.append(f"display_name = ${param_count}")
            params.append(plan_data['display_name'])
            param_count += 1
            
        if 'description' in plan_data:
            set_clauses.append(f"description = ${param_count}")
            params.append(plan_data['description'])
            param_count += 1
            
        if 'price' in plan_data:
            set_clauses.append(f"price = ${param_count}")
            params.append(plan_data['price'])
            param_count += 1
            
        if 'currency' in plan_data:
            set_clauses.append(f"currency = ${param_count}")
            params.append(plan_data['currency'])
            param_count += 1
            
        if 'billing' in plan_data:
            set_clauses.append(f"billing = ${param_count}")
            params.append(plan_data['billing'])
            param_count += 1
            
        if 'features' in plan_data:
            set_clauses.append(f"features = ${param_count}")
            params.append(json.dumps(plan_data['features']) if plan_data['features'] else '[]')
            param_count += 1
            
        if 'max_surveys' in plan_data:
            set_clauses.append(f"max_surveys = ${param_count}")
            params.append(plan_data['max_surveys'])
            param_count += 1
            
        if 'max_responses' in plan_data:
            set_clauses.append(f"max_responses = ${param_count}")
            params.append(plan_data['max_responses'])
            param_count += 1
            
        if 'priority_support' in plan_data:
            set_clauses.append(f"priority_support = ${param_count}")
            params.append(plan_data['priority_support'])
            param_count += 1
            
        if 'api_access' in plan_data:
            set_clauses.append(f"api_access = ${param_count}")
            params.append(plan_data['api_access'])
            param_count += 1
            
        if 'is_active' in plan_data:
            set_clauses.append(f"is_active = ${param_count}")
            params.append(plan_data['is_active'])
            param_count += 1
        
        # Add plan_id as the last parameter
        params.append(plan_id)
        
        query = f"""
        UPDATE plans 
        SET {', '.join(set_clauses)}, updated_at = NOW()
        WHERE id = ${param_count}
        RETURNING id, name, display_name, description, price, currency, billing, features,
                  max_surveys, max_responses, priority_support, api_access, is_active,
                  created_at, updated_at
        """
        
        return await self.execute_fetchrow(query, params)

    # ==========================================
    # LLM SETTINGS METHODS
    # ==========================================
    
    async def get_all_llm_settings(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all LLM settings with pagination"""
        query = """
        SELECT id, provider, active, api_key, created_by, created_at, updated_at
        FROM llm_settings
        ORDER BY created_at DESC
        OFFSET $1 LIMIT $2
        """
        return await self.execute_query(query, [skip, limit])

    async def get_llm_setting_by_id(self, setting_id: str) -> Optional[Dict[str, Any]]:
        """Get LLM setting by ID"""
        query = """
        SELECT id, provider, active, api_key, created_by, created_at, updated_at
        FROM llm_settings WHERE id = $1
        """
        return await self.execute_fetchrow(query, [setting_id])

    async def get_active_llm_settings(self) -> List[Dict[str, Any]]:
        """Get only active LLM settings"""
        query = """
        SELECT id, provider, active, api_key, created_by, created_at, updated_at
        FROM llm_settings 
        WHERE active = true
        ORDER BY created_at DESC
        """
        return await self.execute_query(query)

    async def create_llm_setting(self, setting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new LLM setting"""
        query = """
        INSERT INTO llm_settings (provider, api_key, active, created_by)
        VALUES ($1, $2, $3, $4)
        RETURNING id, provider, active, api_key, created_by, created_at, updated_at
        """
        return await self.execute_fetchrow(query, [
            setting_data['provider'],
            setting_data.get('api_key'),
            setting_data.get('active', True),
            setting_data.get('created_by')
        ])

    async def update_llm_setting(self, setting_id: str, setting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing LLM setting"""
        set_clauses = []
        params = []
        param_count = 1
        
        if 'provider' in setting_data:
            set_clauses.append(f"provider = ${param_count}")
            params.append(setting_data['provider'])
            param_count += 1
            
        if 'api_key' in setting_data:
            set_clauses.append(f"api_key = ${param_count}")
            params.append(setting_data['api_key'])
            param_count += 1
            
        if 'active' in setting_data:
            set_clauses.append(f"active = ${param_count}")
            params.append(setting_data['active'])
            param_count += 1
        
        if not set_clauses:
            raise ValueError("No fields to update")
        
        # Add setting_id as the last parameter
        params.append(setting_id)
        
        query = f"""
        UPDATE llm_settings 
        SET {', '.join(set_clauses)}, updated_at = NOW()
        WHERE id = ${param_count}
        RETURNING id, provider, active, api_key, created_by, created_at, updated_at
        """
        
        return await self.execute_fetchrow(query, params)

    async def delete_llm_setting(self, setting_id: str) -> bool:
        """Delete an LLM setting"""
        query = "DELETE FROM llm_settings WHERE id = $1"
        result = await self.execute_query(query, [setting_id])
        return result is not None

    async def upsert_llm_setting(self, setting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update LLM setting based on provider"""
        # First check if setting exists for this provider
        existing = await self.execute_fetchrow(
            "SELECT id FROM llm_settings WHERE provider = $1",
            [setting_data['provider']]
        )
        
        if existing:
            return await self.update_llm_setting(str(existing['id']), setting_data)
        else:
            return await self.create_llm_setting(setting_data)

    # ==========================================
    # MODULE CONFIGURATIONS METHODS
    # ==========================================
    
    async def get_all_module_configurations(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all module configurations with pagination"""
        query = """
        SELECT id, module_name, llm_setting_id, temperature, max_tokens, max_completion_tokens,
               active, created_at, updated_at, created_by, ai_personality_id, model
        FROM module_configurations
        ORDER BY created_at DESC
        OFFSET $1 LIMIT $2
        """
        return await self.execute_query(query, [skip, limit])

    async def get_module_configuration_by_id(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get module configuration by ID"""
        query = """
        SELECT id, module_name, llm_setting_id, temperature, max_tokens, max_completion_tokens,
               active, created_at, updated_at, created_by, ai_personality_id, model
        FROM module_configurations WHERE id = $1
        """
        return await self.execute_fetchrow(query, [config_id])

    async def get_active_module_configurations(self) -> List[Dict[str, Any]]:
        """Get only active module configurations"""
        query = """
        SELECT id, module_name, llm_setting_id, temperature, max_tokens, max_completion_tokens,
               active, created_at, updated_at, created_by, ai_personality_id, model
        FROM module_configurations 
        WHERE active = true
        ORDER BY created_at DESC
        """
        return await self.execute_query(query)

    async def upsert_module_configuration(
        self, 
        module_name: str, 
        llm_setting_id: str, 
        model: str, 
        temperature: float = 0.7,
        max_tokens: int = 1000,
        max_completion_tokens: int = 1000,
        active: bool = True,
        ai_personality_id: str = None,
        created_by: str = None
    ) -> Dict[str, Any]:
        """Create or update module configuration"""
        config_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # First, try to update existing configuration
        update_query = """
        UPDATE module_configurations 
        SET llm_setting_id = $2, model = $3, temperature = $4, max_tokens = $5, 
            max_completion_tokens = $6, active = $7, ai_personality_id = $8, updated_at = $9
        WHERE module_name = $1
        RETURNING id, module_name, llm_setting_id, model, temperature, max_tokens, max_completion_tokens,
                  active, ai_personality_id, created_by, created_at, updated_at
        """
        
        result = await self.execute_query(
            update_query, 
            [module_name, llm_setting_id, model, temperature, max_tokens, 
             max_completion_tokens, active, ai_personality_id, now]
        )
        
        if result:
            return result[0]
        
        # If no existing configuration, create new one
        insert_query = """
        INSERT INTO module_configurations 
        (id, module_name, llm_setting_id, model, temperature, max_tokens, max_completion_tokens, 
         active, ai_personality_id, created_by, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        RETURNING id, module_name, llm_setting_id, model, temperature, max_tokens, max_completion_tokens,
                  active, ai_personality_id, created_by, created_at, updated_at
        """
        
        result = await self.execute_query(
            insert_query, 
            [config_id, module_name, llm_setting_id, model, temperature, max_tokens, 
             max_completion_tokens, active, ai_personality_id, created_by, now, now]
        )
        
        if result:
            return result[0]
        else:
            raise Exception("Failed to create module configuration")

    async def update_module_configuration(self, config_id: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing module configuration"""
        # Build update query dynamically based on provided fields
        set_clauses = []
        params = []
        param_count = 1
        
        if 'module_name' in config_data:
            set_clauses.append(f"module_name = ${param_count}")
            params.append(config_data['module_name'])
            param_count += 1
            
        if 'llm_setting_id' in config_data:
            set_clauses.append(f"llm_setting_id = ${param_count}")
            params.append(config_data['llm_setting_id'])
            param_count += 1
            
        if 'model' in config_data:
            set_clauses.append(f"model = ${param_count}")
            params.append(config_data['model'])
            param_count += 1
            
        if 'temperature' in config_data:
            set_clauses.append(f"temperature = ${param_count}")
            params.append(config_data['temperature'])
            param_count += 1
            
        if 'max_tokens' in config_data:
            set_clauses.append(f"max_tokens = ${param_count}")
            params.append(config_data['max_tokens'])
            param_count += 1
            
        if 'max_completion_tokens' in config_data:
            set_clauses.append(f"max_completion_tokens = ${param_count}")
            params.append(config_data['max_completion_tokens'])
            param_count += 1
            
        if 'active' in config_data:
            set_clauses.append(f"active = ${param_count}")
            params.append(config_data['active'])
            param_count += 1
            
        if 'ai_personality_id' in config_data:
            set_clauses.append(f"ai_personality_id = ${param_count}")
            params.append(config_data['ai_personality_id'])
            param_count += 1
        
        # Add config_id as the last parameter
        params.append(config_id)
        
        query = f"""
        UPDATE module_configurations 
        SET {', '.join(set_clauses)}, updated_at = NOW()
        WHERE id = ${param_count}
        RETURNING id, module_name, llm_setting_id, temperature, max_tokens, max_completion_tokens,
                  active, created_at, updated_at, created_by, ai_personality_id, model
        """
        
        return await self.execute_fetchrow(query, params)

    async def delete_module_configuration(self, config_id: str) -> bool:
        """Delete a module configuration"""
        query = "DELETE FROM module_configurations WHERE id = $1"
        result = await self.execute_query(query, [config_id])
        return len(result) > 0 if result else False

    async def get_active_configuration_for_module(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get active configuration for a specific module"""
        query = """
        SELECT id, module_name, llm_setting_id, temperature, max_tokens, max_completion_tokens,
               active, created_at, updated_at, created_by, ai_personality_id, model
        FROM module_configurations 
        WHERE module_name = $1 AND active = true
        ORDER BY created_at DESC
        LIMIT 1
        """
        return await self.execute_fetchrow(query, [module_name])

    # ==========================================
    # LOGS METHODS
    # ==========================================
    
    async def get_all_logs(self, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all logs with pagination"""
        query = """
        SELECT id, user_id, action, resource, resource_id, details, ip_address, user_agent,
               timestamp, level, method, endpoint, status_code, session_id, request_body,
               response_body, response_time, error_message, stack_trace, api_key_used,
               provider, model, tokens_used, cost, priority, created_at
        FROM logs
        ORDER BY created_at DESC
        OFFSET $1 LIMIT $2
        """
        return await self.execute_query(query, [skip, limit])

    async def get_log_by_id(self, log_id: str) -> Optional[Dict[str, Any]]:
        """Get log by ID"""
        query = """
        SELECT id, user_id, action, resource, resource_id, details, ip_address, user_agent,
               timestamp, level, method, endpoint, status_code, session_id, request_body,
               response_body, response_time, error_message, stack_trace, api_key_used,
               provider, model, tokens_used, cost, priority, created_at
        FROM logs WHERE id = $1
        """
        return await self.execute_fetchrow(query, [log_id])

    async def get_logs_by_user(self, user_id: str, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get logs for a specific user"""
        query = """
        SELECT id, user_id, action, resource, resource_id, details, ip_address, user_agent,
               timestamp, level, method, endpoint, status_code, session_id, request_body,
               response_body, response_time, error_message, stack_trace, api_key_used,
               provider, model, tokens_used, cost, priority, created_at
        FROM logs 
        WHERE user_id = $1
        ORDER BY created_at DESC
        OFFSET $2 LIMIT $3
        """
        return await self.execute_query(query, [user_id, skip, limit])

    async def get_logs(self, filters: dict) -> List[Dict[str, Any]]:
        """Get logs with comprehensive filtering"""
        conditions = ["1=1"]  # Base condition
        params = []
        param_count = 0
        
        # Build WHERE clause based on filters
        if filters.get("level"):
            param_count += 1
            conditions.append(f"level = ${param_count}")
            params.append(filters["level"])
            
        if filters.get("user_id"):
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(str(filters["user_id"]))
            
        if filters.get("endpoint"):
            param_count += 1
            conditions.append(f"endpoint ILIKE ${param_count}")
            params.append(f"%{filters['endpoint']}%")
            
        if filters.get("method"):
            param_count += 1
            conditions.append(f"method = ${param_count}")
            params.append(filters["method"])
            
        if filters.get("start_date"):
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(filters["start_date"])
            
        if filters.get("end_date"):
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(filters["end_date"])
        
        # Add pagination
        skip = filters.get("skip", 0)
        limit = filters.get("limit", 50)
        param_count += 2
        
        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT id, user_id, action, resource, resource_id, details, ip_address, user_agent,
               timestamp, level, method, endpoint, status_code, session_id, request_body,
               response_body, response_time, error_message, stack_trace, api_key_used,
               provider, model, tokens_used, cost, priority, created_at
        FROM logs
        WHERE {where_clause}
        ORDER BY created_at DESC
        OFFSET ${param_count-1} LIMIT ${param_count}
        """
        params.extend([skip, limit])
        return await self.execute_query(query, params)

    async def get_logs_count(self, filters: dict) -> int:
        """Get count of logs matching filters"""
        conditions = ["1=1"]
        params = []
        param_count = 0
        
        # Build WHERE clause based on filters
        if filters.get("level"):
            param_count += 1
            conditions.append(f"level = ${param_count}")
            params.append(filters["level"])
            
        if filters.get("user_id"):
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(str(filters["user_id"]))
            
        if filters.get("endpoint"):
            param_count += 1
            conditions.append(f"endpoint ILIKE ${param_count}")
            params.append(f"%{filters['endpoint']}%")
            
        if filters.get("method"):
            param_count += 1
            conditions.append(f"method = ${param_count}")
            params.append(filters["method"])
            
        if filters.get("start_date"):
            param_count += 1
            conditions.append(f"created_at >= ${param_count}")
            params.append(filters["start_date"])
            
        if filters.get("end_date"):
            param_count += 1
            conditions.append(f"created_at <= ${param_count}")
            params.append(filters["end_date"])
        
        where_clause = " AND ".join(conditions)
        query = f"""
        SELECT COUNT(*) as count FROM logs WHERE {where_clause}
        """
        result = await self.execute_fetchrow(query, params)
        return result["count"] if result else 0

    async def get_log_statistics(self, start_date, end_date, user_id=None):
        """Get comprehensive log statistics"""
        user_filter = ""
        params = [start_date, end_date]
        if user_id:
            user_filter = "AND user_id = $3"
            params.append(str(user_id))
        
        # Get log level counts
        levels_query = f"""
        SELECT level, COUNT(*) as count
        FROM logs
        WHERE created_at >= $1 AND created_at <= $2 {user_filter}
        GROUP BY level
        ORDER BY count DESC
        """
        levels = await self.execute_query(levels_query, params)
        
        # Get top endpoints
        endpoints_query = f"""
        SELECT endpoint, COUNT(*) as count
        FROM logs
        WHERE created_at >= $1 AND created_at <= $2 {user_filter}
        AND endpoint IS NOT NULL
        GROUP BY endpoint
        ORDER BY count DESC
        LIMIT 10
        """
        endpoints = await self.execute_query(endpoints_query, params)
        
        # Get error trends (hourly)
        errors_query = f"""
        SELECT DATE_TRUNC('hour', created_at) as hour, COUNT(*) as count
        FROM logs
        WHERE created_at >= $1 AND created_at <= $2 {user_filter}
        AND level IN ('ERROR', 'CRITICAL')
        GROUP BY hour
        ORDER BY hour
        """
        errors = await self.execute_query(errors_query, params)
        
        return {
            "levels": levels,
            "top_endpoints": endpoints,
            "error_trends": errors,
            "period": {"start": start_date, "end": end_date}
        }

    def get_log_levels(self):
        """Get available log levels"""
        return ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    async def get_recent_logs(self, limit: int = 20, level: str = None) -> List[Dict[str, Any]]:
        """Get most recent logs"""
        level_filter = ""
        params = [limit]
        if level:
            level_filter = "WHERE level = $2"
            params = [limit, level]
        
        query = f"""
        SELECT id, user_id, action, resource, resource_id, details, ip_address, user_agent,
               timestamp, level, method, endpoint, status_code, session_id, request_body,
               response_body, response_time, error_message, stack_trace, api_key_used,
               provider, model, tokens_used, cost, priority, created_at
        FROM logs
        {level_filter}
        ORDER BY created_at DESC
        LIMIT $1
        """
        return await self.execute_query(query, params)

    async def get_error_logs(self, start_date, end_date, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get error logs within time period"""
        query = """
        SELECT id, user_id, action, resource, resource_id, details, ip_address, user_agent,
               timestamp, level, method, endpoint, status_code, session_id, request_body,
               response_body, response_time, error_message, stack_trace, api_key_used,
               provider, model, tokens_used, cost, priority, created_at
        FROM logs
        WHERE created_at >= $1 AND created_at <= $2
        AND level IN ('ERROR', 'CRITICAL')
        ORDER BY created_at DESC
        OFFSET $3 LIMIT $4
        """
        return await self.execute_query(query, [start_date, end_date, skip, limit])

    async def create_log_entry(self, log_data) -> Dict[str, Any]:
        """Create a new log entry"""
        log_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        query = """
        INSERT INTO logs (id, level, action, resource, details, user_id, endpoint, method,
                         status_code, response_time, ip_address, user_agent, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING *
        """
        params = [
            log_id,
            log_data.level,
            log_data.message,  # Using message as action
            log_data.endpoint,
            log_data.message,  # Using message as details too
            str(log_data.user_id) if log_data.user_id else None,
            log_data.endpoint,
            log_data.method,
            log_data.status_code,
            log_data.response_time,
            log_data.ip_address,
            log_data.user_agent,
            now
        ]
        result = await self.execute_fetchrow(query, params)
        return result

    async def count_logs_for_cleanup(self, cutoff_date, level: str = None) -> int:
        """Count logs that would be deleted in cleanup"""
        level_filter = ""
        params = [cutoff_date]
        if level:
            level_filter = "AND level = $2"
            params.append(level)
        
        query = f"""
        SELECT COUNT(*) as count
        FROM logs
        WHERE created_at < $1 {level_filter}
        """
        result = await self.execute_fetchrow(query, params)
        return result["count"] if result else 0

    async def cleanup_old_logs(self, cutoff_date, level: str = None) -> int:
        """Delete old logs"""
        level_filter = ""
        params = [cutoff_date]
        if level:
            level_filter = "AND level = $2"
            params.append(level)
        
        query = f"""
        DELETE FROM logs
        WHERE created_at < $1 {level_filter}
        """
        # For DELETE operations, we need to count first
        count = await self.count_logs_for_cleanup(cutoff_date, level)
        await self.execute_command(query, params)
        return count
    
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

    # ==========================================
    # AUTH METHODS - Direct SQL for Authentication
    # ==========================================
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email with role information - optimized"""
        query_with_flag = """
        SELECT u.id, u.email, u.username, u.password, u.language, 
           u.email_confirmed, u.welcome_popup_dismissed, u.last_login,
       u.created_at, u.updated_at, u.preferred_personality, u.has_ai_personalities_access, u.avatar, r.role as role_name
        FROM users u
        LEFT JOIN roles r ON u.roleid = r.id
        WHERE u.email = $1
        """

        # Some deployments may not have run the migration that adds
        # `has_ai_personalities_access`. If the column is missing the
        # SELECT will raise asyncpg.UndefinedColumnError; in that case
        # retry with a fallback query that omits the column and set the
        # flag to False by default.
        try:
            return await self.execute_fetchrow(query_with_flag, [email])
        except asyncpg.UndefinedColumnError:
            logger.warning("Users table missing 'has_ai_personalities_access' column, using fallback query")
            query_fallback = """
            SELECT u.id, u.email, u.username, u.password, u.language, 
                   u.email_confirmed, u.welcome_popup_dismissed, u.last_login,
             u.created_at, u.updated_at, u.preferred_personality, r.role as role_name
            FROM users u
            LEFT JOIN roles r ON u.roleid = r.id
            WHERE u.email = $1
            """
            row = await self.execute_fetchrow(query_fallback, [email])
            if row is not None:
                row.setdefault('has_ai_personalities_access', False)
                row.setdefault('avatar', None)
            return row
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID with role information"""
        query_with_flag = """
        SELECT u.id, u.email, u.username, u.password, u.language, 
           u.email_confirmed, u.welcome_popup_dismissed, u.last_login,
       u.created_at, u.updated_at, u.preferred_personality, u.has_ai_personalities_access, u.avatar, r.role as role_name
        FROM users u
        LEFT JOIN roles r ON u.roleid = r.id
        WHERE u.id = $1
        """

        try:
            return await self.execute_fetchrow(query_with_flag, [user_id])
        except asyncpg.UndefinedColumnError:
            logger.warning("Users table missing 'has_ai_personalities_access' column, using fallback query")
            query_fallback = """
            SELECT u.id, u.email, u.username, u.password, u.language, 
                   u.email_confirmed, u.welcome_popup_dismissed, u.last_login,
             u.created_at, u.updated_at, u.preferred_personality, r.role as role_name
            FROM users u
            LEFT JOIN roles r ON u.roleid = r.id
            WHERE u.id = $1
            """
            row = await self.execute_fetchrow(query_fallback, [user_id])
            if row is not None:
                row.setdefault('has_ai_personalities_access', False)
                row.setdefault('avatar', None)
            return row
    
    async def create_user(self, email: str, username: str, hashed_password: str, language: str = "en-US") -> Dict[str, Any]:
        """Create a new user with default role"""
        import uuid
        from datetime import datetime
        
        # Get default role ID (user role)
        role_query = "SELECT id FROM roles WHERE role = 'user' LIMIT 1"
        role_result = await self.execute_fetchrow(role_query)
        role_id = role_result["id"] if role_result else None
        
        if not role_id:
            raise ValueError("Default user role not found")
        
        user_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        query = """
        INSERT INTO users (id, email, username, password, language, roleid, 
                          email_confirmed, welcome_popup_dismissed, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, false, false, $7, $8)
        RETURNING id, email, username, language, email_confirmed, welcome_popup_dismissed, created_at, updated_at
        """
        
        result = await self.execute_fetchrow(query, [user_id, email, username, hashed_password, language, role_id, now, now])
        
        # Add role information
        if result:
            result["role_name"] = "user"
        
        return result
    
    async def update_user_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        from datetime import datetime
        query = "UPDATE users SET last_login = $1 WHERE id = $2"
        await self.execute_command(query, [datetime.utcnow(), user_id])
    
    async def create_refresh_token(self, user_id: str, refresh_token: str, expires_at: datetime) -> str:
        """Create a new refresh token"""
        import uuid
        token_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO refresh_tokens (id, user_id, token, expires_at, created_at)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """
        
        result = await self.execute_fetchrow(query, [token_id, user_id, refresh_token, expires_at, datetime.utcnow()])
        return result["id"] if result else None
    
    async def get_refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Get refresh token with user information"""
        query = """
        SELECT rt.id, rt.user_id, rt.token, rt.expires_at, rt.is_revoked,
         u.email, u.username, u.language, u.welcome_popup_dismissed, u.has_ai_personalities_access, u.avatar, r.role as role_name
        FROM refresh_tokens rt
        JOIN users u ON rt.user_id = u.id
        LEFT JOIN roles r ON u.roleid = r.id
        WHERE rt.token = $1 AND rt.is_revoked = false
        """

        try:
            return await self.execute_fetchrow(query, [refresh_token])
        except asyncpg.UndefinedColumnError:
            # Fallback for older DB schemas without the flag
            logger.warning("Users table missing 'has_ai_personalities_access' column in refresh token lookup, falling back")
            query_fallback = """
            SELECT rt.id, rt.user_id, rt.token, rt.expires_at, rt.is_revoked,
                   u.email, u.username, u.language, u.welcome_popup_dismissed, r.role as role_name
            FROM refresh_tokens rt
            JOIN users u ON rt.user_id = u.id
            LEFT JOIN roles r ON u.roleid = r.id
            WHERE rt.token = $1 AND rt.is_revoked = false
            """
            row = await self.execute_fetchrow(query_fallback, [refresh_token])
            if row is not None:
                row.setdefault('has_ai_personalities_access', False)
                row.setdefault('avatar', None)
            return row
    
    async def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke a specific refresh token"""
        query = "UPDATE refresh_tokens SET is_revoked = true WHERE token = $1"
        result = await self.execute_command(query, [refresh_token])
        return result is not None
    
    async def revoke_all_user_tokens(self, user_id: str) -> int:
        """Revoke all refresh tokens for a user"""
        query = "UPDATE refresh_tokens SET is_revoked = true WHERE user_id = $1 AND is_revoked = false"
        # Note: execute_command doesn't return row count, so we'll estimate
        await self.execute_command(query, [user_id])
        
        # Count how many were affected (rough estimate)
        count_query = "SELECT COUNT(*) as count FROM refresh_tokens WHERE user_id = $1 AND is_revoked = true"
        result = await self.execute_fetchrow(count_query, [user_id])
        return result["count"] if result else 0
    
    # ==========================================
    # USER MANAGEMENT METHODS - Lightweight replacements for auth_service
    # ==========================================
    
    async def get_all_users(self, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all users with role information"""
        query_with_flag = """
        SELECT u.id, u.email, u.username, u.language, 
         u.email_confirmed, u.welcome_popup_dismissed, u.last_login, u.has_ai_personalities_access, u.avatar,
         u.created_at, u.updated_at, u.preferred_personality, r.role as role_name, u.roleid
        FROM users u
        LEFT JOIN roles r ON u.roleid = r.id
        ORDER BY u.created_at DESC
        LIMIT $1 OFFSET $2
        """

        try:
            rows = await self.execute_query(query_with_flag, [limit, skip])
            return rows
        except asyncpg.UndefinedColumnError:
            logger.warning("Users table missing 'has_ai_personalities_access' column, using fallback query for get_all_users")
            query_fallback = """
            SELECT u.id, u.email, u.username, u.language, 
             u.email_confirmed, u.welcome_popup_dismissed, u.last_login,
             u.created_at, u.updated_at, u.preferred_personality, r.role as role_name, u.roleid
            FROM users u
            LEFT JOIN roles r ON u.roleid = r.id
            ORDER BY u.created_at DESC
            LIMIT $1 OFFSET $2
            """
            rows = await self.execute_query(query_fallback, [limit, skip])
            # Ensure backward compatibility: add missing flag default False and avatar default
            for r in rows:
                r.setdefault('has_ai_personalities_access', False)
                r.setdefault('avatar', None)
            return rows
    
    async def update_user_password(self, user_id: str, hashed_password: str) -> bool:
        """Update user password"""
        query = "UPDATE users SET password = $1, updated_at = $2 WHERE id = $3"
        result = await self.execute_command(query, [hashed_password, datetime.utcnow(), user_id])
        return result is not None
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user and all related data"""
        try:
            # Delete in order to respect foreign key constraints
            # Need to delete records that reference the user first
            queries = [
                # Delete user-created data first
                "DELETE FROM llm_settings WHERE created_by = $1",
                "DELETE FROM module_configurations WHERE created_by = $1",
                "DELETE FROM ai_personalities WHERE created_by = $1",
                "DELETE FROM prompt_translations WHERE created_by = $1",
                
                # Delete user access records 
                "DELETE FROM user_survey_access WHERE user_id = $1 OR granted_by = $1",
                "DELETE FROM user_survey_file_access WHERE user_id = $1 OR granted_by = $1",
                
                # Delete user sessions and related data
                "DELETE FROM refresh_tokens WHERE user_id = $1",
                "DELETE FROM chat_messages WHERE session_id IN (SELECT id FROM chat_sessions WHERE user_id = $1)",
                "DELETE FROM chat_sessions WHERE user_id = $1",
                "DELETE FROM notifications WHERE user_id = $1",
                
                # Delete user plan associations
                "DELETE FROM user_plans WHERE user_id = $1",
                
                # Delete logs associated with the user
                "DELETE FROM logs WHERE user_id = $1",
                
                # Finally delete the user
                "DELETE FROM users WHERE id = $1"
            ]
            
            for query in queries:
                await self.execute_command(query, [user_id])
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    async def confirm_email(self, user_id: str) -> bool:
        """Confirm user's email address"""
        query = "UPDATE users SET email_confirmed = true WHERE id = $1"
        result = await self.execute_command(query, [user_id])
        return result is not None

    async def update_user_profile(self, user_id: str, user_update) -> Optional[Dict[str, Any]]:
        """Update user profile information"""
        try:
            # Build dynamic update query based on provided fields
            set_clauses = []
            params = []
            param_count = 0
            
            if hasattr(user_update, 'username') and user_update.username is not None:
                param_count += 1
                set_clauses.append(f"username = ${param_count}")
                params.append(user_update.username)
                
            if hasattr(user_update, 'email') and user_update.email is not None:
                param_count += 1
                set_clauses.append(f"email = ${param_count}")
                params.append(user_update.email)
                
            if hasattr(user_update, 'password') and user_update.password is not None:
                # Hash password before storing
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                hashed_password = pwd_context.hash(user_update.password)
                param_count += 1
                set_clauses.append(f"password = ${param_count}")
                params.append(hashed_password)
                
            if hasattr(user_update, 'language') and user_update.language is not None:
                param_count += 1
                set_clauses.append(f"language = ${param_count}")
                params.append(user_update.language)
                
            if hasattr(user_update, 'preferred_personality') and user_update.preferred_personality is not None:
                param_count += 1
                set_clauses.append(f"preferred_personality = ${param_count}")
                params.append(str(user_update.preferred_personality))
                
            if hasattr(user_update, 'welcome_popup_dismissed') and user_update.welcome_popup_dismissed is not None:
                param_count += 1
                set_clauses.append(f"welcome_popup_dismissed = ${param_count}")
                params.append(user_update.welcome_popup_dismissed)

            if hasattr(user_update, 'avatar') and user_update.avatar is not None:
                param_count += 1
                set_clauses.append(f"avatar = ${param_count}")
                params.append(user_update.avatar)

            # Allow updating the user's role by role name. We'll translate role name to roleid.
            role_id_to_set = None
            if hasattr(user_update, 'role') and user_update.role is not None:
                # Lookup role id
                try:
                    role_row = await self.execute_fetchrow("SELECT id FROM roles WHERE role = $1 LIMIT 1", [user_update.role])
                    if role_row and role_row.get('id'):
                        role_id_to_set = role_row['id']
                except Exception as e:
                    logger.warning(f"Failed to lookup role '{user_update.role}': {e}")
                if role_id_to_set:
                    param_count += 1
                    set_clauses.append(f"roleid = ${param_count}")
                    params.append(role_id_to_set)
            
            # Allow updating the AI personalities access flag
            if hasattr(user_update, 'has_ai_personalities_access') and user_update.has_ai_personalities_access is not None:
                param_count += 1
                set_clauses.append(f"has_ai_personalities_access = ${param_count}")
                params.append(bool(user_update.has_ai_personalities_access))
            
            if not set_clauses:
                # No fields to update
                return await self.get_user_by_id(user_id)
            
            # Add updated_at timestamp
            param_count += 1
            set_clauses.append(f"updated_at = ${param_count}")
            params.append(datetime.utcnow())
            
            # Add user_id for WHERE clause
            param_count += 1
            params.append(user_id)
            
            set_clause = ", ".join(set_clauses)
            query = f"""
            UPDATE users 
            SET {set_clause}
            WHERE id = ${param_count}
            RETURNING *
            """
            
            try:
                result = await self.execute_fetchrow(query, params)
                return result
            except asyncpg.UndefinedColumnError:
                # Database may not have the has_ai_personalities_access column yet.
                # Retry the UPDATE without that column to avoid failing the whole request.
                logger.warning("Users table missing 'has_ai_personalities_access' column, retrying update without it")

                # Rebuild query omitting has_ai_personalities_access
                set_clauses2 = []
                params2 = []
                param_count2 = 0

                if hasattr(user_update, 'username') and user_update.username is not None:
                    param_count2 += 1
                    set_clauses2.append(f"username = ${param_count2}")
                    params2.append(user_update.username)
                if hasattr(user_update, 'email') and user_update.email is not None:
                    param_count2 += 1
                    set_clauses2.append(f"email = ${param_count2}")
                    params2.append(user_update.email)
                if hasattr(user_update, 'password') and user_update.password is not None:
                    from passlib.context import CryptContext
                    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                    hashed_password = pwd_context.hash(user_update.password)
                    param_count2 += 1
                    set_clauses2.append(f"password = ${param_count2}")
                    params2.append(hashed_password)
                if hasattr(user_update, 'language') and user_update.language is not None:
                    param_count2 += 1
                    set_clauses2.append(f"language = ${param_count2}")
                    params2.append(user_update.language)
                if hasattr(user_update, 'preferred_personality') and user_update.preferred_personality is not None:
                    param_count2 += 1
                    set_clauses2.append(f"preferred_personality = ${param_count2}")
                    params2.append(str(user_update.preferred_personality))
                if hasattr(user_update, 'welcome_popup_dismissed') and user_update.welcome_popup_dismissed is not None:
                    param_count2 += 1
                    set_clauses2.append(f"welcome_popup_dismissed = ${param_count2}")
                    params2.append(user_update.welcome_popup_dismissed)
                if hasattr(user_update, 'avatar') and user_update.avatar is not None:
                    param_count2 += 1
                    set_clauses2.append(f"avatar = ${param_count2}")
                    params2.append(user_update.avatar)

                if not set_clauses2:
                    return await self.get_user_by_id(user_id)

                param_count2 += 1
                set_clauses2.append(f"updated_at = ${param_count2}")
                params2.append(datetime.utcnow())

                param_count2 += 1
                params2.append(user_id)

                set_clause2 = ", ".join(set_clauses2)
                query2 = f"""
                UPDATE users 
                SET {set_clause2}
                WHERE id = ${param_count2}
                RETURNING *
                """

                try:
                    result2 = await self.execute_fetchrow(query2, params2)
                    return result2
                except Exception as e2:
                    logger.error(f"Retry update_user_profile failed without ai flag: {e2}")
                    return None
            
        except Exception as e:
            logger.error(f"Failed to update user profile {user_id}: {e}")
            return None

    async def get_user_with_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user with additional statistics"""
        # Get basic user info
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
            
        # Get user statistics (surveys, responses, etc.)
        stats_query = """
        SELECT 
            (SELECT COUNT(*) FROM chat_sessions WHERE user_id = $1) as total_sessions,
            (SELECT COUNT(*) FROM chat_messages cm 
             JOIN chat_sessions cs ON cm.session_id = cs.id 
             WHERE cs.user_id = $1) as total_messages,
            (SELECT MAX(last_login) FROM users WHERE id = $1) as last_activity
        """
        stats = await self.execute_fetchrow(stats_query, [user_id])
        
        return {
            "user": user,
            "stats": {
                "total_sessions": stats.get("total_sessions", 0) if stats else 0,
                "total_messages": stats.get("total_messages", 0) if stats else 0,
                "last_activity": stats.get("last_activity") if stats else None
            }
        }

    async def get_multi_users(self, skip: int = 0, limit: int = 50, options=None) -> List[Dict[str, Any]]:
        """Get multiple users with filtering options"""
        # For now, just return all users with pagination
        return await self.get_all_users(skip=skip, limit=limit)

    async def get_user_by_id_simple(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Alias for get_user_by_id for compatibility"""
        return await self.get_user_by_id(user_id)

    async def delete_user_by_id(self, user_id: str) -> bool:
        """Delete user by ID - alias for compatibility"""
        return await self.delete_user(user_id)

    # ==========================================
    # USER ACCESS METHODS
    # ==========================================
    
    async def get_user_survey_access(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all survey access permissions for a user"""
        query = """
        SELECT usa.id, usa.user_id, usa.survey_id, usa.access_type, 
               usa.granted_at, usa.expires_at, usa.is_active,
               s.title, s.category
        FROM user_survey_access usa
        LEFT JOIN surveys s ON usa.survey_id = s.id
        WHERE usa.user_id = $1
        ORDER BY usa.granted_at DESC
        """
        return await self.execute_query(query, [user_id])
    
    async def get_user_file_access(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all file access permissions for a user"""
        query = """
        SELECT usfa.id, usfa.user_id, usfa.survey_file_id, usfa.access_type,
               usfa.granted_at, usfa.expires_at, usfa.is_active,
               sf.filename, s.id as survey_id, s.title, s.category
        FROM user_survey_file_access usfa
        LEFT JOIN survey_files sf ON usfa.survey_file_id = sf.id
        LEFT JOIN surveys s ON sf.survey_id = s.id
        WHERE usfa.user_id = $1
        ORDER BY usfa.granted_at DESC
        """
        return await self.execute_query(query, [user_id])

    # ==========================================
    # LOGGING METHODS
    # ==========================================
    
    async def log_api_error(
        self, 
        error_message: str, 
        endpoint: str, 
        method: str, 
        status_code: int,
        user_id: Optional[str] = None,
        request_body: Optional[str] = None,
        stack_trace: Optional[str] = None
    ) -> str:
        """Log API error to database"""
        try:
            import uuid
            from datetime import datetime
            
            log_id = str(uuid.uuid4())
            
            query = """
            INSERT INTO logs (id, level, action, user_id, method, endpoint, status_code, 
                             error_message, stack_trace, request_body, created_at)
            VALUES ($1, 'ERROR', $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            """
            
            action = f"api_error_{method}_{endpoint}"
            
            result = await self.execute_fetchrow(query, [
                log_id, action, user_id, method, endpoint, status_code,
                error_message, stack_trace, request_body, datetime.utcnow()
            ])
            return str(result["id"]) if result else log_id
        except Exception as e:
            logger.error(f"Failed to log API error to database: {e}")
            return "logging_failed"

    async def log_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        response_time: Optional[float] = None,
        user_id: Optional[str] = None,
        request_body: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Log API request to database"""
        try:
            import uuid
            from datetime import datetime
            
            log_id = str(uuid.uuid4())
            
            query = """
            INSERT INTO logs (id, level, action, user_id, method, endpoint, status_code, 
                             response_time, request_body, ip_address, user_agent, created_at)
            VALUES ($1, 'INFO', $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            RETURNING id
            """
            
            action = f"api_request_{method}_{endpoint}"
            
            result = await self.execute_fetchrow(query, [
                log_id, action, user_id, method, endpoint, status_code,
                response_time, request_body, ip_address, user_agent, datetime.utcnow()
            ])
            return str(result["id"]) if result else log_id
        except Exception as e:
            logger.error(f"Failed to log API request to database: {e}")
            return "logging_failed"

    # ==========================================
    # LANGUAGE METHODS
    # ==========================================
    
    async def get_enabled_languages(self) -> List[Dict[str, Any]]:
        """Get all enabled languages"""
        query = """
        SELECT code, name, native_name, enabled, created_at, updated_at
        FROM supported_languages
        WHERE enabled = true
        ORDER BY name
        """
        return await self.execute_query(query)
    
    async def get_all_languages(self) -> List[Dict[str, Any]]:
        """Get all languages (enabled and disabled)"""
        query = """
        SELECT code, name, native_name, enabled, created_at, updated_at
        FROM supported_languages
        ORDER BY name
        """
        return await self.execute_query(query)
    
    # ==========================================
    # NOTIFICATION METHODS
    # ==========================================
    
    async def get_user_notifications(
        self, 
        user_id: str, 
        unread_only: bool = False, 
        notification_type: Optional[str] = None,
        skip: int = 0, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get notifications for a specific user"""
        query = """
        SELECT id, user_id, title, message, type, is_read, status, priority,
               admin_response, responded_by, responded_at, created_at, updated_at
        FROM notifications
        WHERE user_id = $1
        """
        params = [user_id]
        param_count = 1
        
        if unread_only:
            param_count += 1
            query += f" AND is_read = ${param_count}"
            params.append(False)
        
        if notification_type:
            param_count += 1
            query += f" AND type = ${param_count}"
            params.append(notification_type)
        
        query += " ORDER BY created_at DESC"
        
        if skip > 0:
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(skip)
            
        param_count += 1
        query += f" LIMIT ${param_count}"
        params.append(limit)
        
        return await self.execute_query(query, params)

    async def get_all_notifications(
        self, 
        unread_only: bool = False, 
        notification_type: Optional[str] = None,
        skip: int = 0, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get all notifications for admin users"""
        query = """
        SELECT id, user_id, title, message, type, is_read, status, priority,
               admin_response, responded_by, responded_at, created_at, updated_at
        FROM notifications
        WHERE 1=1
        """
        params = []
        param_count = 0
        
        if unread_only:
            param_count += 1
            query += f" AND is_read = ${param_count}"
            params.append(False)
        
        if notification_type:
            param_count += 1
            query += f" AND type = ${param_count}"
            params.append(notification_type)
        
        query += " ORDER BY created_at DESC"
        
        if skip > 0:
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(skip)
            
        param_count += 1
        query += f" LIMIT ${param_count}"
        params.append(limit)
        
        return await self.execute_query(query, params)

    async def create_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        notification_type: str = "general",
        priority: int = 1,
        metadata: dict = None
    ) -> str:
        """Create a new notification"""
        try:
            notification_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Convert metadata to JSON string if provided
            metadata_json = None
            if metadata:
                import json
                metadata_json = json.dumps(metadata)
            
            # Convert user_id to UUID
            user_uuid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
            
            query = """
            INSERT INTO notifications (id, user_id, title, message, type, priority, is_read, status, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9)
            """
            
            await self.execute_command(query, [
                notification_id,
                user_uuid,
                title,
                message,
                notification_type,
                priority,
                False,  # is_read
                "pending",  # status
                now
            ])
            
            logger.info(f"Created notification {notification_id} for user {user_id}")
            return notification_id
            
        except Exception as e:
            logger.error(f"Failed to create notification: {e}")
            raise e

    async def update_notification(
        self,
        notification_id: str,
        status: Optional[str] = None,
        admin_response: Optional[str] = None,
        responded_by: Optional[str] = None,
        responded_at: Optional[datetime] = None
    ) -> bool:
        """Update a notification"""
        try:
            # Build dynamic update query based on provided fields
            set_clauses = []
            params = []
            param_count = 0
            
            if status is not None:
                param_count += 1
                set_clauses.append(f"status = ${param_count}")
                params.append(status)
                
            if admin_response is not None:
                param_count += 1
                set_clauses.append(f"admin_response = ${param_count}")
                params.append(admin_response)
                
            if responded_by is not None:
                param_count += 1
                set_clauses.append(f"responded_by = ${param_count}")
                params.append(uuid.UUID(responded_by) if isinstance(responded_by, str) else responded_by)
                
            if responded_at is not None:
                param_count += 1
                set_clauses.append(f"responded_at = ${param_count}")
                params.append(responded_at)
            
            if not set_clauses:
                return False
            
            # Always update the updated_at timestamp
            param_count += 1
            set_clauses.append(f"updated_at = ${param_count}")
            params.append(datetime.utcnow())
            
            # Add notification_id for WHERE clause
            param_count += 1
            params.append(uuid.UUID(notification_id) if isinstance(notification_id, str) else notification_id)
            
            query = f"""
            UPDATE notifications 
            SET {', '.join(set_clauses)}
            WHERE id = ${param_count}
            """
            
            result = await self.execute_command(query, params)
            
            if result:
                logger.info(f"Updated notification {notification_id}")
                return True
            else:
                logger.warning(f"No notification found with id {notification_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update notification {notification_id}: {e}")
            raise e

    async def mark_notification_read(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Mark a specific notification as read for a user"""
        try:
            now = datetime.utcnow()
            
            query = """
            UPDATE notifications 
            SET is_read = true, updated_at = $1
            WHERE id = $2 AND user_id = $3 AND is_read = false
            """
            
            result = await self.execute_command(query, [
                now,
                uuid.UUID(notification_id) if isinstance(notification_id, str) else notification_id,
                uuid.UUID(user_id) if isinstance(user_id, str) else user_id
            ])
            
            if result:
                logger.info(f"Marked notification {notification_id} as read for user {user_id}")
                return True
            else:
                logger.warning(f"No unread notification found with id {notification_id} for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to mark notification {notification_id} as read: {e}")
            raise e

    async def mark_all_notifications_read(self, user_id: str) -> int:
        """Mark all unread notifications as read for a user"""
        try:
            now = datetime.utcnow()
            
            # First count unread notifications
            count_query = """
            SELECT COUNT(*) FROM notifications 
            WHERE user_id = $1 AND is_read = false
            """
            
            count_result = await self.execute_fetchrow(count_query, [
                uuid.UUID(user_id) if isinstance(user_id, str) else user_id
            ])
            
            count = count_result["count"] if count_result else 0
            
            if count > 0:
                # Mark all as read
                update_query = """
                UPDATE notifications 
                SET is_read = true, updated_at = $1
                WHERE user_id = $2 AND is_read = false
                """
                
                await self.execute_command(update_query, [
                    now,
                    uuid.UUID(user_id) if isinstance(user_id, str) else user_id
                ])
                
                logger.info(f"Marked {count} notifications as read for user {user_id}")
            
            return count
                
        except Exception as e:
            logger.error(f"Failed to mark all notifications as read for user {user_id}: {e}")
            raise e

    # ===========================
    # USER PLAN MANAGEMENT METHODS
    # ===========================

    async def assign_plan_to_user(self, user_id: str, plan_id: str) -> Dict[str, Any]:
        """Assign a plan to a user - either create new or update existing"""
        try:
            # Check if user has an existing active plan
            existing_plan_query = """
            SELECT id FROM user_plans 
            WHERE user_id = $1 AND status = 'active'
            LIMIT 1
            """
            existing_plan = await self.execute_fetchrow(existing_plan_query, [user_id])
            
            if existing_plan:
                # Update existing plan instead of cancelling and creating new
                update_query = """
                UPDATE user_plans 
                SET plan_id = $1, updated_at = NOW()
                WHERE user_id = $2 AND status = 'active'
                RETURNING id, user_id, plan_id, status, start_date, created_at
                """
                result = await self.execute_fetchrow(update_query, [plan_id, user_id])
                logger.info(f"Updated existing plan for user {user_id} to plan {plan_id}")
            else:
                # Create new user plan (no existing plan)
                user_plan_id = str(uuid.uuid4())
                create_query = """
                INSERT INTO user_plans (id, user_id, plan_id, status, start_date, auto_renew, created_at, updated_at)
                VALUES ($1, $2, $3, 'active', NOW(), true, NOW(), NOW())
                RETURNING id, user_id, plan_id, status, start_date, created_at
                """
                result = await self.execute_fetchrow(create_query, [user_plan_id, user_id, plan_id])
                logger.info(f"Created new plan assignment: plan {plan_id} to user {user_id}")
            
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Failed to assign plan {plan_id} to user {user_id}: {e}")
            raise

    async def revoke_plan_from_user(self, user_id: str, user_plan_id: Optional[str] = None) -> bool:
        """Completely revoke/cancel a plan from a user (sets status to cancelled)"""
        try:
            if user_plan_id:
                # Revoke specific plan
                query = """
                UPDATE user_plans 
                SET status = 'cancelled', updated_at = NOW()
                WHERE id = $1 AND user_id = $2
                """
                result = await self.execute_command(query, [user_plan_id, user_id])
            else:
                # Revoke current active plan
                query = """
                UPDATE user_plans 
                SET status = 'cancelled', updated_at = NOW()
                WHERE user_id = $1 AND status = 'active'
                """
                result = await self.execute_command(query, [user_id])
            
            logger.info(f"Revoked plan for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke plan for user {user_id}: {e}")
            return False

    async def remove_user_plan_access(self, user_id: str) -> bool:
        """Completely remove plan access for a user (deletes the record)"""
        try:
            query = """
            DELETE FROM user_plans 
            WHERE user_id = $1 AND status = 'active'
            """
            result = await self.execute_command(query, [user_id])
            logger.info(f"Removed plan access for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove plan access for user {user_id}: {e}")
            return False

    async def get_user_plan(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the active plan for a user"""
        try:
            query = """
            SELECT up.id, up.user_id, up.plan_id, up.status, up.start_date, up.created_at,
                   p.name as plan_name, p.description as plan_description, p.features as plan_features,
                   p.price as plan_price, p.billing_cycle as plan_billing_cycle
            FROM user_plans up
            JOIN plans p ON up.plan_id = p.id
            WHERE up.user_id = $1 AND up.status = 'active'
            ORDER BY up.start_date DESC
            LIMIT 1
            """
            
            result = await self.execute_fetchrow(query, [user_id])
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Failed to get user plan for user {user_id}: {e}")
            return None

    async def cancel_user_plan(self, user_id: str) -> bool:
        """Cancel a user's active plan"""
        try:
            query = """
            UPDATE user_plans 
            SET status = 'cancelled', updated_at = NOW()
            WHERE user_id = $1 AND status = 'active'
            """
            
            await self.execute_command(query, [user_id])
            logger.info(f"Cancelled plan for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel plan for user {user_id}: {e}")
            return False

    async def get_plan_users(self, plan_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all users with a specific plan"""
        try:
            if active_only:
                query = """
                SELECT up.id, up.user_id, up.plan_id, up.status, up.start_date,
                       u.username, u.email, u.full_name
                FROM user_plans up
                JOIN users u ON up.user_id = u.id
                WHERE up.plan_id = $1 AND up.status = 'active'
                ORDER BY up.start_date DESC
                """
            else:
                query = """
                SELECT up.id, up.user_id, up.plan_id, up.status, up.start_date,
                       u.username, u.email, u.full_name
                FROM user_plans up
                JOIN users u ON up.user_id = u.id
                WHERE up.plan_id = $1
                ORDER BY up.start_date DESC
                """
            
            results = await self.execute_query(query, [plan_id])
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get users for plan {plan_id}: {e}")
            return []

    async def get_plan_usage_stats(self, plan_id: str) -> Dict[str, Any]:
        """Get usage statistics for a plan"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_users,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_users,
                COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_users
            FROM user_plans
            WHERE plan_id = $1
            """
            
            result = await self.execute_fetchrow(query, [plan_id])
            if result:
                return {
                    "plan_id": plan_id,
                    "total_users": result['total_users'],
                    "active_users": result['active_users'],
                    "cancelled_users": result['cancelled_users']
                }
            else:
                return {
                    "plan_id": plan_id,
                    "total_users": 0,
                    "active_users": 0,
                    "cancelled_users": 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get usage stats for plan {plan_id}: {e}")
            return {
                "plan_id": plan_id,
                "total_users": 0,
                "active_users": 0,
                "cancelled_users": 0
            }

    async def get_user_plans_with_details(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user plans with plan details for admin interface"""
        try:
            query = """
            SELECT up.id, up.user_id, up.plan_id, up.status, up.start_date, up.end_date,
                   up.trial_ends_at, up.auto_renew, up.payment_method_id, up.stripe_subscription_id,
                   up.created_at, up.updated_at,
                   p.id as plan_detail_id, p.name as plan_name, p.display_name as plan_display_name,
                   p.description as plan_description, p.price as plan_price, p.currency as plan_currency,
                   p.billing as plan_billing, p.features as plan_features, p.max_surveys as plan_max_surveys,
                   p.max_responses as plan_max_responses, p.priority_support as plan_priority_support,
                   p.api_access as plan_api_access, p.is_active as plan_is_active,
                   p.created_at as plan_created_at, p.updated_at as plan_updated_at
            FROM user_plans up
            JOIN plans p ON up.plan_id = p.id
            WHERE up.user_id = $1
            ORDER BY 
                CASE WHEN up.status = 'active' THEN 0 ELSE 1 END,
                up.start_date DESC
            """
            
            results = await self.execute_query(query, [user_id])
            
            user_plans = []
            for row in results:
                # Parse plan features
                plan_features = []
                if row.get('plan_features'):
                    import json
                    try:
                        plan_features = json.loads(row['plan_features'])
                        if not isinstance(plan_features, list):
                            plan_features = []
                    except (json.JSONDecodeError, TypeError):
                        plan_features = []
                
                user_plan = {
                    "id": str(row["id"]),
                    "user_id": str(row["user_id"]),
                    "plan_id": str(row["plan_id"]),
                    "status": row["status"],
                    "start_date": row["start_date"].isoformat() if row["start_date"] else None,
                    "end_date": row["end_date"].isoformat() if row["end_date"] else None,
                    "trial_ends_at": row["trial_ends_at"].isoformat() if row["trial_ends_at"] else None,
                    "auto_renew": row["auto_renew"],
                    "payment_method_id": row["payment_method_id"],
                    "stripe_subscription_id": row["stripe_subscription_id"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                    "plans": {
                        "id": str(row["plan_detail_id"]),
                        "name": row["plan_name"],
                        "display_name": row["plan_display_name"],
                        "description": row["plan_description"],
                        "price": float(row["plan_price"]) if row["plan_price"] else None,
                        "currency": row["plan_currency"],
                        "billing": row["plan_billing"],
                        "features": plan_features,
                        "max_surveys": row["plan_max_surveys"],
                        "max_responses": row["plan_max_responses"],
                        "priority_support": row["plan_priority_support"],
                        "api_access": row["plan_api_access"],
                        "is_active": row["plan_is_active"],
                        "created_at": row["plan_created_at"].isoformat() if row["plan_created_at"] else None,
                        "updated_at": row["plan_updated_at"].isoformat() if row["plan_updated_at"] else None
                    }
                }
                user_plans.append(user_plan)
                
            return user_plans
            
        except Exception as e:
            logger.error(f"Failed to get user plans with details for user {user_id}: {e}")
            return []

    async def check_plan_feature(self, user_id: str, feature_name: str) -> bool:
        """Check if user's plan includes a specific feature"""
        try:
            user_plan = await self.get_user_plan(user_id)
            if not user_plan:
                return False
            
            # Parse features JSON
            features_json = user_plan.get('plan_features', '[]')
            if not features_json:
                return False
                
            import json
            try:
                features = json.loads(features_json)
                if isinstance(features, dict):
                    return features.get(feature_name, False)
                elif isinstance(features, list):
                    return feature_name in features
                else:
                    return False
            except (json.JSONDecodeError, TypeError):
                return False
                
        except Exception as e:
            logger.error(f"Failed to check feature {feature_name} for user {user_id}: {e}")
            return False


# Create service instances for compatibility
class LightweightUserService:
    """Lightweight user service to replace the old SQLAlchemy user_service"""
    
    def __init__(self, db_service: LightweightDBService):
        self.db = db_service
    
    async def get_multi(self, db, skip: int = 0, limit: int = 50, options=None):
        """Get multiple users - compatibility method"""
        return await self.db.get_all_users(skip=skip, limit=limit)
    
    async def get_user_with_stats(self, db, user_id):
        """Get user with stats - simplified version"""
        user_data = await self.db.get_user_by_id(str(user_id))
        if user_data:
            # Create a simple user object with stats
            return {
                "user": type('SimpleUser', (), user_data)(),  # Convert dict to object
                "stats": {"surveys_completed": 0, "total_responses": 0}  # Placeholder stats
            }
        return None
    
    async def delete_by_id(self, db, user_id):
        """Delete user by ID"""
        return await self.db.delete_user(str(user_id))

    # ==========================================
    # SURVEY METHODS
    # ==========================================
    
    async def get_user_surveys(self, user_id: str, skip: int = 0, limit: int = 100, include_pending: bool = False) -> List[Dict[str, Any]]:
        """
        Get surveys for a specific user
        
        Args:
            user_id: The user ID (currently not used for filtering, but available for future access control)
            skip: Number of records to skip
            limit: Maximum number of records to return
            include_pending: Whether to include pending surveys (default False for regular users)
        """
        if include_pending:
            # Admin users see all surveys
            query = """
            SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
                   s.number_participants, s.total_files, s.processing_status, s.created_at, s.updated_at
            FROM surveys s
            ORDER BY s.created_at DESC
            OFFSET $1 LIMIT $2
            """
        else:
            # Regular users only see completed surveys
            query = """
            SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
                   s.number_participants, s.total_files, s.processing_status, s.created_at, s.updated_at
            FROM surveys s
            WHERE s.processing_status = 'completed' OR s.processing_status IS NULL
            ORDER BY s.created_at DESC
            OFFSET $1 LIMIT $2
            """
        return await self.execute_query(query, [skip, limit])
    
    async def get_survey_by_id(self, survey_id: str) -> Optional[Dict[str, Any]]:
        """Get survey by ID"""
        query = """
        SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
               s.number_participants, s.total_files, s.created_at, s.updated_at
        FROM surveys s
        WHERE s.id = $1
        """
        return await self.execute_fetchrow(query, [survey_id])

    # ==========================================
    # LANGUAGE METHODS
    # ==========================================
    
    async def get_enabled_languages(self) -> List[Dict[str, Any]]:
        """Get all enabled languages"""
        query = """
        SELECT code, name, native_name, enabled, created_at, updated_at
        FROM supported_languages
        WHERE enabled = true
        ORDER BY name
        """
        return await self.execute_query(query)
    
    async def get_all_languages(self) -> List[Dict[str, Any]]:
        """Get all languages (enabled and disabled)"""
        query = """
        SELECT code, name, native_name, enabled, created_at, updated_at
        FROM supported_languages
        ORDER BY name
        """
        return await self.execute_query(query)

    # ==========================================
    # NOTIFICATION METHODS
    # ==========================================
    
    async def get_user_notifications(
        self, 
        user_id: str, 
        unread_only: bool = False, 
        notification_type: Optional[str] = None,
        skip: int = 0, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get notifications for a specific user"""
        query = """
        SELECT id, user_id, title, message, type, is_read, status, priority,
               admin_response, responded_by, responded_at, created_at, updated_at
        FROM notifications
        WHERE user_id = $1
        """
        params = [user_id]
        param_count = 1
        
        if unread_only:
            param_count += 1
            query += f" AND is_read = ${param_count}"
            params.append(False)
        
        if notification_type:
            param_count += 1
            query += f" AND type = ${param_count}"
            params.append(notification_type)
        
        query += " ORDER BY created_at DESC"
        
        if skip > 0:
            param_count += 1
            query += f" OFFSET ${param_count}"
            params.append(skip)
            
        param_count += 1
        query += f" LIMIT ${param_count}"
        params.append(limit)
        
        return await self.execute_query(query, params)


# ==========================================
# SERVICE COMPATIBILITY LAYER
# ==========================================

class LightweightUserService:
    """Lightweight user service to replace the old SQLAlchemy user_service"""
    
    def __init__(self, db_service: LightweightDBService):
        self.db = db_service
    
    async def get_multi(self, db, skip: int = 0, limit: int = 50, options=None):
        """Get multiple users - compatibility method"""
        return await self.db.get_all_users(skip=skip, limit=limit)
    
    async def get_user_with_stats(self, db, user_id):
        """Get user with stats - simplified version"""
        user_data = await self.db.get_user_by_id(str(user_id))
        if user_data:
            # Create a simple user object with stats
            return {
                "user": type('SimpleUser', (), user_data)(),  # Convert dict to object
                "stats": {"surveys_completed": 0, "total_responses": 0}  # Placeholder stats
            }
        return None
    
    async def delete_by_id(self, db, user_id):
        """Delete user by ID"""
        return await self.db.delete_user(str(user_id))


# ==========================================
# GLOBAL INSTANCES & DEPENDENCY INJECTION
# ==========================================

# Global instance - singleton pattern like TypeScript API
lightweight_db = LightweightDBService()
user_service = LightweightUserService(lightweight_db)


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


# Phase 2C: Enhanced session management utilities
async def optimize_database_sessions():
    """Optimize database sessions for better performance."""
    try:
        # Clean up idle sessions
        cleanup_query = """
        SELECT pg_terminate_backend(pid) 
        FROM pg_stat_activity 
        WHERE state = 'idle in transaction' 
        AND state_change < NOW() - INTERVAL '5 minutes'
        """
        await lightweight_db.execute_command(cleanup_query)
        
        # Update connection statistics
        stats_query = """
        SELECT 
            count(*) as total_connections,
            count(*) FILTER (WHERE state = 'active') as active_connections,
            count(*) FILTER (WHERE state = 'idle') as idle_connections
        FROM pg_stat_activity 
        WHERE datname = current_database()
        """
        stats = await lightweight_db.fetch_one_dict(stats_query)
        logger.info(f"Session optimization complete: {stats}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Session optimization failed: {e}")
        return None


async def get_database_performance_metrics():
    """Get enhanced database performance metrics for monitoring."""
    try:
        metrics = {}
        
        # Connection pool stats
        metrics['pool'] = lightweight_db.get_pool_stats()
        
        # Query performance stats
        query_stats = """
        SELECT 
            calls,
            total_time,
            mean_time,
            query
        FROM pg_stat_statements 
        ORDER BY mean_time DESC 
        LIMIT 10
        """
        try:
            metrics['slow_queries'] = await lightweight_db.fetch_all_dict(query_stats)
        except:
            metrics['slow_queries'] = "pg_stat_statements not available"
        
        # Active connections
        conn_query = """
        SELECT 
            count(*) as total,
            count(*) FILTER (WHERE state = 'active') as active,
            count(*) FILTER (WHERE state = 'idle') as idle
        FROM pg_stat_activity 
        WHERE datname = current_database()
        """
        metrics['connections'] = await lightweight_db.fetch_one_dict(conn_query)
        
        # Cache hit ratios
        cache_query = """
        SELECT 
            sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) * 100 as cache_hit_ratio
        FROM pg_statio_user_tables
        """
        cache_result = await lightweight_db.fetch_one_dict(cache_query)
        metrics['cache_hit_ratio'] = cache_result['cache_hit_ratio'] if cache_result else 0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return {"error": str(e)}
