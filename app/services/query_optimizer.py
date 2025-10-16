"""
Database Query Optimization Service
==================================
Provides optimized database queries with prepared statements and batching.
"""

from typing import List, Dict, Any, Optional
from app.utils.logging import get_logger

logger = get_logger(__name__)


class QueryOptimizer:
    """
    Service for optimized database queries with caching and prepared statements
    """
    
    def __init__(self):
        # Store frequently used query templates
        self.prepared_queries = {
            'chat_messages_recent': """
                SELECT sender_type, content 
                FROM chat_messages 
                WHERE session_id = $1 
                ORDER BY created_at DESC
                LIMIT $2
            """,
            'module_config_with_personality': """
                SELECT 
                    mc.model, 
                    mc.temperature, 
                    mc.max_tokens, 
                    mc.ai_personality_id,
                    ls.provider, 
                    ls.api_key,
                    ap.name as personality_name,
                    ap.detailed_analysis_prompt
                FROM module_configurations mc
                LEFT JOIN llm_settings ls ON mc.llm_setting_id = ls.id
                LEFT JOIN ai_personalities ap ON mc.ai_personality_id = ap.id AND ap.is_active = true
                WHERE mc.module_name = $1 AND mc.active = true AND ls.active = true
            """,
            'personality_prompt_only': """
                SELECT detailed_analysis_prompt 
                FROM ai_personalities 
                WHERE id = $1 AND is_active = true
            """,
            'multiple_module_configs': """
                SELECT 
                    mc.module_name,
                    mc.model, 
                    mc.temperature, 
                    mc.max_tokens, 
                    mc.ai_personality_id,
                    ls.provider, 
                    ls.api_key,
                    ap.name as personality_name,
                    ap.detailed_analysis_prompt
                FROM module_configurations mc
                LEFT JOIN llm_settings ls ON mc.llm_setting_id = ls.id
                LEFT JOIN ai_personalities ap ON mc.ai_personality_id = ap.id AND ap.is_active = true
                WHERE mc.module_name = ANY($1) AND mc.active = true AND ls.active = true
            """
        }
        
        # Performance tracking
        self.query_count = 0
        self.bulk_query_count = 0
        self.cache_hits = 0
    
    async def get_recent_chat_messages(self, db, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent chat messages for a session with optimized query
        """
        self.query_count += 1
        try:
            messages = await db.execute_fetchall(
                self.prepared_queries['chat_messages_recent'],
                [session_id, limit]
            )
            return messages or []
        except Exception as e:
            logger.error(f"Failed to fetch recent chat messages: {str(e)}")
            return []
    
    async def get_module_config_with_personality(self, db, module_name: str) -> Optional[Dict[str, Any]]:
        """
        Get module configuration with personality data in single query
        """
        self.query_count += 1
        try:
            result = await db.execute_fetchrow(
                self.prepared_queries['module_config_with_personality'],
                [module_name]
            )
            return result
        except Exception as e:
            logger.error(f"Failed to fetch module config: {str(e)}")
            return None
    
    async def get_multiple_module_configs(self, db, module_names: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get multiple module configurations in a single optimized query
        """
        self.bulk_query_count += 1
        results = {}
        
        try:
            rows = await db.execute_fetchall(
                self.prepared_queries['multiple_module_configs'],
                [module_names]
            )
            
            # Process results and handle decryption
            for row in rows:
                module_name = row["module_name"]
                
                # Decrypt API key
                api_key = None
                if row.get("api_key"):
                    try:
                        from app.utils.encryption import encryption_service
                        api_key = encryption_service.decrypt_api_key(row["api_key"])
                    except Exception as e:
                        logger.error(f"Failed to decrypt API key for {module_name}: {str(e)}")
                        api_key = None
                
                results[module_name] = {
                    "provider": row["provider"],
                    "model": row["model"],
                    "temperature": float(row.get("temperature", 0.7)) if row.get("temperature") is not None else 0.7,
                    "max_tokens": row.get("max_tokens", 2000),
                    "api_key": api_key,
                    "ai_personality_id": row.get("ai_personality_id"),
                    "personality_name": row.get("personality_name"),
                    "detailed_analysis_prompt": row.get("detailed_analysis_prompt")
                }
            
            # Ensure all requested modules have entries (None if not found)
            for module_name in module_names:
                if module_name not in results:
                    results[module_name] = None
                    
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch multiple module configs: {str(e)}")
            # Return None for all requested modules on error
            return {module_name: None for module_name in module_names}
    
    async def get_personality_prompt(self, db, personality_id: str) -> Optional[str]:
        """
        Get personality prompt only
        """
        self.query_count += 1
        try:
            result = await db.execute_fetchrow(
                self.prepared_queries['personality_prompt_only'],
                [personality_id]
            )
            return result.get("detailed_analysis_prompt") if result else None
        except Exception as e:
            logger.error(f"Failed to fetch personality prompt: {str(e)}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get query optimizer performance statistics."""
        return {
            "total_queries": self.query_count,
            "bulk_queries": self.bulk_query_count,
            "cache_hits": self.cache_hits,
            "query_efficiency": f"{(self.bulk_query_count / max(self.query_count, 1) * 100):.1f}% bulk operations"
        }
    
    async def batch_execute(self, db, queries: List[tuple]) -> List[Any]:
        """
        Execute multiple queries in a batch for better performance
        """
        results = []
        try:
            for query, params in queries:
                if query in self.prepared_queries:
                    query = self.prepared_queries[query]
                result = await db.execute_fetchrow(query, params)
                results.append(result)
        except Exception as e:
            logger.error(f"Batch query execution failed: {str(e)}")
        
        return results


# Global query optimizer instance
query_optimizer = QueryOptimizer()