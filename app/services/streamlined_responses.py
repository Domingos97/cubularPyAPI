"""
Lightweight Response Handlers - No Pydantic Overhead
===================================================
Simple dict-based responses that match TypeScript API patterns.
No schema validation overhead, just clean data transformation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


class StreamlinedResponses:
    """
    Simple response builders without Pydantic overhead
    Matches TypeScript API response patterns exactly
    """
    
    @staticmethod
    def chat_completion_response(
        user_message_id: str,
        ai_response: str,
        ai_message_id: str,
        processing_time: float,
        search_results: Optional[Dict] = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        cached: bool = False
    ) -> Dict[str, Any]:
        """
        Build chat completion response - Enhanced to match TypeScript API
        Attempts to parse structured JSON response from AI
        """
        # Try to parse structured response from AI
        parsed_response = StreamlinedResponses._parse_ai_response(ai_response)
        
        # Base response structure matching TypeScript API
        response = {
            "conversationalResponse": parsed_response.get("conversationalResponse", ai_response),
            "dataSnapshot": parsed_response.get("dataSnapshot", {}),
            "confidence": parsed_response.get("confidence", {
                "score": 50,
                "reliability": "medium",
                "factors": {
                    "sampleSize": len(search_results.get("responses", [])) if search_results else 0,
                    "dataRelevance": "moderate",
                    "questionSpecificity": "general",
                    "dataCompleteness": "partial"
                }
            }),
            "sessionId": user_message_id,  # Will be updated by caller if needed
            "context": {
                "totalMatches": search_results.get("metadata", {}).get("total_matches", 0) if search_results else 0,
                "exactMatches": len(search_results.get("responses", [])) if search_results else 0,
                "responses": search_results.get("responses", []) if search_results else []
            },
            "processingType": "python-integrated",
            "searchMetadata": {
                "strategy": search_results.get("metadata", {}).get("search_strategy", "fast_direct_search") if search_results else "none",
                "processingTime": search_results.get("metadata", {}).get("processing_time", processing_time) if search_results else processing_time,
                "cacheHit": cached
            }
        }
        
        return response
    
    @staticmethod
    def _parse_ai_response(ai_response: str) -> Dict[str, Any]:
        """
        Try to parse structured JSON response from AI
        AI should always return JSON with conversationalResponse, dataSnapshot, and confidence
        """
        import json
        import re
        
        try:
            # Clean the response of any markdown or extra formatting
            cleaned_response = ai_response.strip()
            
            # Remove markdown code blocks if present
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Try to parse directly first
            try:
                parsed = json.loads(cleaned_response)
                
                # Validate the expected structure
                if isinstance(parsed, dict) and "conversationalResponse" in parsed:
                    return parsed
                elif isinstance(parsed, str):
                    # AI returned just a string instead of JSON structure
                    return {
                        "conversationalResponse": parsed,
                        "dataSnapshot": {},
                        "confidence": {"score": 30, "reliability": "low"}
                    }
            except json.JSONDecodeError:
                pass
            
            # Try to find JSON block in the response using regex
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate structure
                if isinstance(parsed, dict) and "conversationalResponse" in parsed:
                    return parsed
            
            # Fallback: wrap plain text response
            return {
                "conversationalResponse": cleaned_response,
                "dataSnapshot": {},
                "confidence": {"score": 20, "reliability": "low"}
            }
            
        except Exception as e:
            # If all else fails, return plain text wrapped
            print(f"Error parsing AI response: {e}")
            return {
                "conversationalResponse": ai_response,
                "dataSnapshot": {},
                "confidence": {"score": 10, "reliability": "low"}
            }
    
    @staticmethod
    def session_response(
        session_id: str,
        title: str,
        user_id: str,
        survey_ids: List[str] = None,
        category: str = None,
        personality_id: str = None,
        created_at: datetime = None,
        updated_at: datetime = None
    ) -> Dict[str, Any]:
        """Build session response - consistent snake_case format"""
        response = {
            "id": session_id,
            "title": title,
            "user_id": user_id,
            "survey_ids": survey_ids or [],
            "category": category,
            "personality_id": personality_id,
            "created_at": (created_at or datetime.utcnow()).isoformat(),
            "updated_at": (updated_at or datetime.utcnow()).isoformat()
        }
        return response
    
    @staticmethod
    def session_with_messages_response(
        session: Dict[str, Any],
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build session with messages - TypeScript style with full message data"""
        formatted_messages = []
        
        for msg in messages:
            # Create base message structure
            formatted_msg = {
                "id": msg.get("id", str(uuid.uuid4())),
                "session_id": msg.get("session_id"),
                "content": msg.get("content", ""),
                "sender": msg.get("sender", "user"),  # Map role -> sender for frontend compatibility
                "timestamp": msg.get("created_at", datetime.utcnow()).isoformat() if isinstance(msg.get("created_at"), datetime) else msg.get("created_at"),
                "message_language": msg.get("message_language"),
                "personality_used": msg.get("personality_used")
            }
            
            # Handle data_snapshot - ensure it's properly included
            if msg.get("data_snapshot") is not None:
                formatted_msg["data_snapshot"] = msg["data_snapshot"]
            
            # Handle confidence - ensure it's properly included
            if msg.get("confidence") is not None:
                formatted_msg["confidence"] = msg["confidence"]
                
            # Include any additional metadata
            if msg.get("metadata"):
                formatted_msg["metadata"] = msg["metadata"]
                
            formatted_messages.append(formatted_msg)
        
        return {
            "session": session,
            "messages": formatted_messages,
            "total_messages": len(messages)
        }
    
    @staticmethod
    def error_response(
        message: str,
        error_code: str = "internal_error",
        status_code: int = 500,
        details: str = None
    ) -> Dict[str, Any]:
        """Build error response - TypeScript API style"""
        return {
            "success": False,
            "error": message,
            "error_code": error_code,
            "status_code": status_code,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def success_response(
        message: str,
        data: Any = None
    ) -> Dict[str, Any]:
        """Build success response - TypeScript style"""
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if data is not None:
            response["data"] = data
        
        return response
    
    @staticmethod
    def cache_stats_response(
        ai_cache_size: int,
        ai_cache_hits: int = 0,
        ai_cache_misses: int = 0,
        search_cache_size: int = 0,
        embedding_cache_size: int = 0
    ) -> Dict[str, Any]:
        """Build cache statistics response"""
        return {
            "ai_cache": {
                "size": ai_cache_size,
                "hits": ai_cache_hits,
                "misses": ai_cache_misses,
                "hit_rate": round(ai_cache_hits / max(ai_cache_hits + ai_cache_misses, 1) * 100, 2)
            },
            "search_cache": {
                "survey_cache_size": search_cache_size,
                "embedding_cache_size": embedding_cache_size
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class StreamingResponseHelper:
    """
    Helper for streaming responses - TypeScript API style
    No overhead, just clean streaming
    """
    
    @staticmethod
    async def stream_chat_response(ai_response_generator, session_id: str, user_message_id: str):
        """
        Stream chat response like TypeScript API
        Yields JSON chunks for client-side processing
        """
        ai_message_id = str(uuid.uuid4())
        full_response = ""
        
        # Initial response with metadata
        yield f"data: {{\n"
        yield f'  "type": "start",\n'
        yield f'  "session_id": "{session_id}",\n'
        yield f'  "user_message_id": "{user_message_id}",\n'
        yield f'  "ai_message_id": "{ai_message_id}"\n'
        yield f"}}\n\n"
        
        # Stream content
        async for chunk in ai_response_generator:
            if chunk:
                full_response += chunk
                yield f"data: {{\n"
                yield f'  "type": "content",\n'
                yield f'  "content": "{chunk.replace('"', '\\"')}"\n'
                yield f"}}\n\n"
        
        # Final response
        yield f"data: {{\n"
        yield f'  "type": "complete",\n'
        yield f'  "full_response": "{full_response.replace('"', '\\"')}",\n'
        yield f'  "ai_message_id": "{ai_message_id}"\n'
        yield f"}}\n\n"


def convert_datetime_to_string(obj: Any) -> Any:
    """
    Convert datetime and UUID objects to strings for JSON serialization
    Simple utility without Pydantic overhead
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_datetime_to_string(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_string(item) for item in obj]
    else:
        return obj


def clean_db_response(db_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean database response for JSON serialization
    Handles datetime, UUID conversion and None values
    Also parses JSONB fields that come as strings from PostgreSQL
    """
    if not db_row:
        return {}
    
    import json
    
    cleaned = {}
    for key, value in db_row.items():
        if value is None:
            cleaned[key] = None
        elif isinstance(value, datetime):
            cleaned[key] = value.isoformat()
        elif isinstance(value, uuid.UUID):
            cleaned[key] = str(value)
        elif isinstance(value, list):
            # Handle lists that might contain UUIDs
            cleaned[key] = [str(item) if isinstance(item, uuid.UUID) else item for item in value]
        elif isinstance(value, dict):
            cleaned[key] = convert_datetime_to_string(value)
        elif key in ['data_snapshot', 'confidence'] and isinstance(value, str):
            # Parse JSONB fields that come as strings from PostgreSQL
            try:
                cleaned[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, keep as string
                cleaned[key] = value
        else:
            cleaned[key] = value
    
    return cleaned