"""
Streamlined Chat Endpoint - Ultra-Fast Performance
=================================================
Question → Search → AI → Response with minimal overhead.
No Pydantic validation, direct dict responses like TypeScript API.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import time
import asyncio
from datetime import datetime

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService, log_chat_analytics_background
from app.services.simple_ai_service import get_simple_ai, SimpleAIService, process_analytics_background
from app.services.fast_search_service import FastSearchService
from app.services.streamlined_responses import StreamlinedResponses, StreamingResponseHelper, clean_db_response
from app.core.lightweight_dependencies import get_current_user, SimpleUser
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Initialize services
fast_search = FastSearchService()


class CreateSessionRequest(BaseModel):
    title: str = "New Chat"
    survey_ids: List[str] = Field(default=[], alias="surveyIds")
    category: Optional[str] = None
    personality_id: Optional[str] = Field(default=None, alias="personalityId")
    selected_file_ids: List[str] = Field(default=[], alias="fileIds")
    
    class Config:
        populate_by_name = True  # Allow both field names and aliases


@router.post("/quick-stream")
async def quick_chat_stream(
    session_id: str,
    message: str,
    include_search: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db),
    ai: SimpleAIService = Depends(get_simple_ai),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Ultra-fast streaming chat completion - TypeScript API style
    Returns streaming JSON for real-time response building
    """
    
    async def generate_stream():
        start_time = time.time()
        
        try:
            # Fast validation - parallel with search
            tasks = [db.get_chat_session(session_id, current_user.id)]
            
            if include_search:
                tasks.append(_perform_search_if_needed(db, session_id, message))
            
            results = await asyncio.gather(*tasks)
            session = results[0]
            search_context = results[1] if include_search else None
            
            if not session:
                yield f"data: {StreamlinedResponses.error_response('Session not found', 'session_not_found', 404)}\n\n"
                return
            
            # Get conversation history and build context
            conversation_history = await db.get_recent_messages(session_id, limit=6)
            
            # Build AI messages
            system_prompt = ai.default_system_prompt
            search_text = ""
            
            if search_context and search_context.get("responses"):
                top_results = search_context["responses"][:5]
                search_text = "\n".join([f"- {r['text'][:200]}..." for r in top_results])
            
            messages = ai.build_messages(
                system_prompt=system_prompt,
                conversation_history=conversation_history,
                current_question=message,
                search_context=search_text if search_text else None
            )
            
            # Stream AI response
            user_msg_id = await db.save_message(session_id, "user", message)
            
            # Create streaming generator
            ai_response = ""
            ai_msg_id = None
            
            # For now, get full response and simulate streaming
            # TODO: Implement actual streaming from AI providers
            full_ai_response = await ai.generate_response(
                messages=messages,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=1000
            )
            
            # Save AI message
            ai_msg_id = await db.save_message(
                session_id, "assistant", full_ai_response,
                {"provider": provider, "model": model}
            )
            
            # Stream the response in chunks (simulate real streaming)
            chunk_size = 20
            for i in range(0, len(full_ai_response), chunk_size):
                chunk = full_ai_response[i:i + chunk_size]
                ai_response += chunk
                
                yield f"data: {{\n"
                yield f'  "type": "content",\n'
                yield f'  "content": "{chunk.replace('"', '\\"')}",\n'
                yield f'  "user_message_id": "{user_msg_id}",\n'
                yield f'  "ai_message_id": "{ai_msg_id}"\n'
                yield f"}}\n\n"
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.05)
            
            # Final completion
            processing_time = (time.time() - start_time) * 1000
            
            yield f"data: {{\n"
            yield f'  "type": "complete",\n'
            yield f'  "full_response": "{full_ai_response.replace('"', '\\"')}",\n'
            yield f'  "processing_time": {processing_time:.2f},\n'
            yield f'  "search_results": {search_context or "null"},\n'
            yield f'  "user_message_id": "{user_msg_id}",\n'
            yield f'  "ai_message_id": "{ai_msg_id}"\n'
            yield f"}}\n\n"
            
            # Background analytics
            background_tasks.add_task(
                process_analytics_background,
                session_id, message, full_ai_response, processing_time, provider, model
            )
            
        except Exception as e:
            yield f"data: {StreamlinedResponses.error_response(f'Stream error: {str(e)}', 'streaming_error', 500)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/plain; charset=utf-8"
        }
    )


@router.post("/quick-completion")
async def quick_chat_completion(
    session_id: str,
    message: str,
    include_search: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db),
    ai: SimpleAIService = Depends(get_simple_ai),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Ultra-fast chat completion endpoint
    
    Flow: Question → Search → AI → Response
    - Minimal validation
    - Parallel operations where possible
    - Background analytics
    - Response caching
    """
    start_time = time.time()
    
    try:
        # Step 1: Parallel session validation and search (if needed)
        tasks = [
            db.get_chat_session(session_id, current_user.id)
        ]
        
        if include_search:
            # Add search task to run in parallel
            tasks.append(_perform_search_if_needed(db, session_id, message))
        
        results = await asyncio.gather(*tasks)
        session = results[0]
        search_context = results[1] if include_search else None
        
        if not session:
            return JSONResponse(
                status_code=404,
                content=StreamlinedResponses.error_response(
                    "Chat session not found",
                    "session_not_found",
                    404
                )
            )
        
        # Step 2: Get conversation history (lightweight query)
        conversation_history = await db.get_recent_messages(session_id, limit=6)  # Reduced from 10
        
        # Step 3: Build AI messages (enhanced context)
        system_prompt = ai.default_system_prompt
        search_text = ""
        
        if search_context and search_context.get("responses"):
            # Build structured context similar to TypeScript API
            responses = search_context["responses"]
            total_matches = search_context.get("metadata", {}).get("total_matches", len(responses))
            
            search_text = f"Processing complete survey dataset - found {total_matches} total data points for analysis\n"
            search_text += f"Most relevant responses (top {len(responses)} matches):\n\n"
            search_text += "Survey responses:\n"
            
            for i, response in enumerate(responses):
                similarity = response.get('value', 0) * 100
                search_text += f"{i + 1}. \"{response.get('text', '')}\" (relevance: {similarity:.1f}%)\n"
            
            # Add demographics if available
            if search_context.get("demographics"):
                search_text += f"\nDemographics: {search_context['demographics']}\n"
                
            # Add psychology insights if available  
            if search_context.get("psychology"):
                search_text += f"\nPsychological insights: {search_context['psychology']}\n"
        
        messages = ai.build_messages(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            current_question=message,
            search_context=search_text if search_text else None
        )
        
        # Step 4: Generate AI response (cached)
        ai_response = await ai.generate_response(
            messages=messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=1000
        )
        
        # Step 5: Save messages (single transaction)
        user_msg_id, ai_msg_id = await db.save_message_pair(
            session_id=session_id,
            user_content=message,
            ai_content=ai_response,
            ai_metadata={
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "search_results_count": len(search_context.get("responses", [])) if search_context else 0,
                "processing_time": time.time() - start_time
            }
        )
        
        # Step 6: Update session title if first exchange (background)
        if len(conversation_history) == 0:
            title = message[:50] + "..." if len(message) > 50 else message
            background_tasks.add_task(db.update_session_title, session_id, title)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Step 7: Background analytics (non-blocking)
        background_tasks.add_task(
            process_analytics_background,
            session_id, message, ai_response, processing_time, provider, model
        )
        
        background_tasks.add_task(
            log_chat_analytics_background,
            session_id, message, processing_time, "fast_search" if include_search else "direct",
            len(search_context.get("responses", [])) if search_context else 0
        )
        
        # Return streamlined response - no Pydantic overhead
        return JSONResponse(
            content=StreamlinedResponses.chat_completion_response(
                user_message_id=user_msg_id,
                ai_response=ai_response,
                ai_message_id=ai_msg_id,
                processing_time=processing_time,
                search_results=search_context,
                provider=provider,
                model=model,
                cached=False  # TODO: Track cache hits
            )
        )
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Quick chat completion error: {str(e)} (took {processing_time:.1f}ms)")
        
        return JSONResponse(
            status_code=500,
            content=StreamlinedResponses.error_response(
                "Failed to generate response",
                "chat_completion_error",
                500,
                str(e)
            )
        )


async def _perform_search_if_needed(db: LightweightDBService, session_id: str, query: str) -> Optional[dict]:
    """
    Perform search if session has survey_ids
    Uses FastSearchService for optimal performance
    """
    try:
        # Get session's survey_ids
        session_query = "SELECT survey_ids FROM chat_sessions WHERE id = $1"
        result = await db.execute_fetchrow(session_query, [session_id])
        
        if not result or not result.get("survey_ids"):
            logger.debug(f"No survey_ids found for session {session_id}")
            return None
        
        survey_ids = result["survey_ids"] if isinstance(result["survey_ids"], list) else []
        
        if not survey_ids:
            logger.debug(f"Empty survey_ids list for session {session_id}")
            return None
        
        logger.info(f"Performing search for session {session_id} with surveys: {survey_ids}")
        
        # Generate proper embedding for the query
        try:
            from app.services.embedding_service import embedding_service
            query_embedding = await embedding_service.generate_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate embedding - using fallback approach")
                return None
                
        except ImportError as e:
            logger.error(f"Embedding service import failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
        
        # Perform fast search with actual embedding
        search_results = await fast_search.fast_search(
            query_embedding=query_embedding,
            survey_ids=survey_ids,
            threshold=0.25,
            max_results=500,  # Reduced for performance
            query_text=query
        )
        
        # Log search results for debugging
        results_count = len(search_results.get('responses', []))
        logger.info(f"Search completed: {results_count} results found for query: '{query[:50]}...'")
        
        if results_count == 0:
            logger.warning(f"Zero results found. Search metadata: {search_results.get('metadata', {})}")
        
        return search_results
        
    except Exception as e:
        logger.error(f"Search failed for session {session_id}: {e}", exc_info=True)
        return None


@router.get("/sessions/{session_id}/quick")
async def get_quick_session(
    session_id: str,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """Get chat session with recent messages - lightweight version"""
    try:
        # Get session info
        session = await db.get_chat_session(session_id, current_user.id)
        if not session:
            return JSONResponse(
                status_code=404,
                content=StreamlinedResponses.error_response(
                    "Chat session not found",
                    "session_not_found",
                    404
                )
            )
        
        # Get recent messages
        messages = await db.get_recent_messages(session_id, limit=20)
        
        # Clean response without Pydantic overhead
        return JSONResponse(
            content=StreamlinedResponses.session_with_messages_response(
                session=clean_db_response(session),
                messages=[clean_db_response(msg) for msg in messages]
            )
        )
        
    except Exception as e:
        logger.error(f"Error getting quick session: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=StreamlinedResponses.error_response(
                "Failed to retrieve session",
                "session_retrieval_error",
                500,
                str(e)
            )
        )


@router.get("/sessions")
async def get_user_chat_sessions(
    limit: int = Query(default=50, le=100),
    surveyId: Optional[str] = Query(default=None, alias="survey_id"),
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """Get all chat sessions for the current user, optionally filtered by survey"""
    try:
        if surveyId:
            # Filter by specific survey
            logger.info(f"Getting sessions for user {current_user.id} and survey {surveyId}")
            sessions = await db.get_chat_sessions_by_survey(
                user_id=current_user.id,
                survey_id=surveyId,
                limit=limit
            )
        else:
            # Get all user sessions
            logger.info(f"Getting all sessions for user {current_user.id}")
            sessions = await db.get_user_chat_sessions(
                user_id=current_user.id,
                limit=limit
            )
        
        # Convert to frontend-friendly format
        sessions_list = []
        for session in sessions:
            session_data = clean_db_response(session)
            # Convert UUID array to string array for frontend
            if session_data.get('survey_ids'):
                session_data['survey_ids'] = [str(sid) for sid in session_data['survey_ids']]
            else:
                session_data['survey_ids'] = []
            sessions_list.append(session_data)
        
        return JSONResponse(
            content={
                "sessions": sessions_list,
                "total": len(sessions_list)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting user sessions: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=StreamlinedResponses.error_response(
                "Failed to retrieve sessions",
                "sessions_retrieval_error",
                500,
                str(e)
            )
        )


@router.post("/sessions/quick")
async def create_quick_session(
    title: str = "New Chat",
    survey_ids: List[str] = None,
    category: str = None,
    personality_id: str = None,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """Create new chat session - lightweight version"""
    try:
        session_id = await db.create_chat_session(
            user_id=current_user.id,
            title=title,
            survey_ids=survey_ids or [],
            category=category,
            personality_id=personality_id
        )
        
        # Enhanced response for frontend compatibility
        session_data = StreamlinedResponses.session_response(
            session_id=session_id,
            title=title,
            user_id=str(current_user.id),
            survey_ids=survey_ids or [],
            category=category,
            personality_id=str(personality_id) if personality_id else None
        )
        
        return JSONResponse(
            content={
                "success": True,
                "session": session_data,
                "sessionId": session_id,  # Explicit session ID for frontend
                **session_data  # Spread session data at root level for compatibility
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating quick session: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=StreamlinedResponses.error_response(
                "Failed to create session",
                "session_creation_error",
                500,
                str(e)
            )
        )


class CreateChatSessionRequest(BaseModel):
    survey_ids: List[str]
    category: Optional[str] = None
    title: str = "New Chat"
    personality_id: Optional[str] = None
    selected_file_ids: List[str] = []


@router.post("/sessions")
async def create_chat_session(
    request: CreateChatSessionRequest,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """Create new chat session - main endpoint that matches TypeScript API"""
    try:
        # Log the incoming request for debugging
        logger.info(f"Creating chat session with: survey_ids={request.survey_ids}, "
                   f"category={request.category}, personality_id={request.personality_id}, "
                   f"selected_file_ids={request.selected_file_ids}")
        
        if not request.survey_ids:
            return JSONResponse(
                status_code=400,
                content={"error": "Survey IDs array is required"}
            )
        
        session_id = await db.create_chat_session(
            user_id=current_user.id,
            title=request.title,
            survey_ids=request.survey_ids,
            category=request.category,
            personality_id=request.personality_id,
            selected_file_ids=request.selected_file_ids
        )
        
        # Return session data in the format expected by frontend
        session_data = {
            "id": session_id,
            "user_id": str(current_user.id),
            "survey_ids": request.survey_ids,
            "category": request.category,
            "title": request.title,
            "personality_id": request.personality_id,
            "selected_file_ids": request.selected_file_ids,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Created chat session {session_id} successfully")
        
        return JSONResponse(
            status_code=201,
            content=session_data
        )
        
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to create chat session",
                "details": str(e)
            }
        )


@router.post("/sessions/create-optimized")
async def create_optimized_session(
    request: CreateSessionRequest,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """Create new optimized chat session - alias for quick session creation"""
    try:
        # Log the incoming request for debugging
        logger.info(f"Creating optimized session with: survey_ids={request.survey_ids}, "
                   f"category={request.category}, personality_id={request.personality_id}, "
                   f"selected_file_ids={request.selected_file_ids}")
        
        session_id = await db.create_chat_session(
            user_id=current_user.id,
            title=request.title,
            survey_ids=request.survey_ids or [],
            category=request.category,
            personality_id=request.personality_id,
            selected_file_ids=request.selected_file_ids or []
        )
        
        # Enhanced response for frontend compatibility
        session_data = StreamlinedResponses.session_response(
            session_id=session_id,
            title=request.title,
            user_id=str(current_user.id),
            survey_ids=request.survey_ids or [],
            category=request.category,
            personality_id=str(request.personality_id) if request.personality_id else None
        )
        
        return JSONResponse(
            content={
                "success": True,
                "session": session_data,
                "sessionId": session_id,  # Explicit session ID for frontend
                **session_data  # Spread session data at root level for compatibility
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating optimized session: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=StreamlinedResponses.error_response(
                "Failed to create session",
                "session_creation_error",
                500,
                str(e)
            )
        )


@router.get("/cache/stats")
async def get_cache_stats(
    current_user: SimpleUser = Depends(get_current_user),
    ai: SimpleAIService = Depends(get_simple_ai)
):
    """Get cache statistics for monitoring"""
    return JSONResponse(
        content=StreamlinedResponses.cache_stats_response(
            ai_cache_size=len(ai.response_cache),
            ai_cache_hits=getattr(ai.response_cache, 'hits', 0),
            ai_cache_misses=getattr(ai.response_cache, 'misses', 0),
            search_cache_size=len(fast_search.survey_cache),
            embedding_cache_size=len(fast_search.embedding_cache)
        )
    )


class SaveMessageRequest(BaseModel):
    session_id: str
    content: str
    sender: str
    data_snapshot: Optional[dict] = None
    confidence: Optional[float] = None
    personality_used: Optional[str] = None


@router.post("/sessions/{session_id}/messages")
async def save_chat_message(
    session_id: str,
    message_data: SaveMessageRequest,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Save a single message to a chat session.
    This endpoint is used by the frontend to save individual messages.
    """
    try:
        # Validate that the session belongs to the current user
        session = await db.execute_fetchrow(
            "SELECT user_id FROM chat_sessions WHERE id = $1",
            [session_id]
        )
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        if session['user_id'] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this chat session"
            )
        
        # Validate sender type
        if message_data.sender not in ['user', 'assistant', 'system']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid sender type"
            )
        
        # Save the message
        message_id = await db.save_message(
            session_id=session_id,
            role=message_data.sender,
            content=message_data.content,
            metadata={
                'data_snapshot': message_data.data_snapshot,
                'confidence': message_data.confidence,
                'personality_used': message_data.personality_used
            }
        )
        
        # Update session timestamp
        await db.execute_command(
            "UPDATE chat_sessions SET updated_at = $1 WHERE id = $2",
            [datetime.utcnow(), session_id]
        )
        
        # Return the saved message in the format expected by the frontend
        saved_message = {
            "id": message_id,
            "session_id": session_id,
            "content": message_data.content,
            "sender": message_data.sender,
            "timestamp": datetime.utcnow().isoformat(),
            "data_snapshot": message_data.data_snapshot,
            "confidence": message_data.confidence,
            "personality_used": message_data.personality_used
        }
        
        logger.info(f"Successfully saved message {message_id} to session {session_id} for user {current_user.id}")
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=saved_message
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error saving message to session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save message: {str(e)}"
        )


@router.get("/sessions/{session_id}/messages")
async def get_chat_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get messages for a chat session.
    This endpoint provides the message history for a session.
    """
    try:
        # Validate that the session belongs to the current user
        session = await db.execute_fetchrow(
            "SELECT user_id FROM chat_sessions WHERE id = $1",
            [session_id]
        )
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        if session['user_id'] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this chat session"
            )
        
        # Get messages from the session
        messages = await db.get_session_messages(session_id, limit)
        
        logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
        
        return JSONResponse(
            content=messages
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error getting messages for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get messages: {str(e)}"
        )


@router.post("/cache/clear")
async def clear_caches(
    current_user: SimpleUser = Depends(get_current_user),
    ai: SimpleAIService = Depends(get_simple_ai)
):
    """Clear all caches - useful for development"""
    ai.clear_cache()
    fast_search.survey_cache.clear()
    fast_search.embedding_cache.clear()
    
    return JSONResponse(
        content=StreamlinedResponses.success_response(
            "All caches cleared"
        )
    )


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """Delete a chat session and all its messages - matches TypeScript API"""
    try:
        logger.info(f"Deleting chat session {session_id} for user {current_user.id}")
        
        # Delete the session (includes ownership verification)
        deleted = await db.delete_chat_session(session_id, current_user.id)
        
        if deleted:
            logger.info(f"Successfully deleted chat session {session_id}")
            return JSONResponse(
                content=StreamlinedResponses.success_response(
                    "Chat session deleted successfully"
                )
            )
        else:
            logger.warning(f"Chat session {session_id} not found or access denied")
            return JSONResponse(
                status_code=404,
                content=StreamlinedResponses.error_response(
                    "Chat session not found",
                    "session_not_found",
                    404
                )
            )
        
    except Exception as e:
        logger.error(f"Error deleting chat session {session_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=StreamlinedResponses.error_response(
                "Failed to delete chat session",
                "session_deletion_error",
                500,
                str(e)
            )
        )


@router.delete("/sessions/{session_id}")
async def delete_session_alternative(
    session_id: str,
    current_user: SimpleUser = Depends(get_current_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """Alternative delete endpoint to match TypeScript API route pattern"""
    return await delete_chat_session(session_id, current_user, db)