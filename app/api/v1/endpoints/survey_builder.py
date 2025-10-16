"""
Survey Builder API endpoints for chat-based survey creation
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
import uuid
import os
import json
from datetime import datetime

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import get_current_regular_user, get_current_admin_user, SimpleUser
from app.models.schemas import SuccessResponse
from app.services.survey_builder_service import survey_builder_service
from app.services.survey_generation_service import survey_generation_service
from app.services.simple_ai_service import get_simple_ai, SimpleAIService
from app.utils.survey_status_updater import update_survey_status, auto_update_survey_statuses
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class SurveyBuildRequest:
    """Request model for survey building"""
    def __init__(self, conversation: List[Dict[str, Any]], survey_title: Optional[str] = None):
        self.conversation = conversation
        self.survey_title = survey_title


@router.post("/chat")
async def survey_builder_chat(
    session_id: str,
    message: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db),
    ai: SimpleAIService = Depends(get_simple_ai)
):
    """
    Chat endpoint for survey builder using Survey Builder Assistant personality
    
    Parameters:
    - **session_id**: ID of the chat session
    - **message**: User's message
    
    Returns AI response from Survey Builder Assistant
    """
    try:
        # Get the chat session
        session = await db.get_chat_session(session_id, current_user.id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Verify this is a survey_builder session
        if session.get("category") != "survey_builder":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This endpoint only works with survey_builder sessions"
            )
        
        # Get conversation history
        conversation_history = await db.get_recent_messages(session_id, limit=10)
        
        # Get survey builder module configuration
        survey_builder_config = await survey_builder_service.get_survey_builder_config(db)
        if not survey_builder_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Survey builder module not configured"
            )
        
        # Get LLM configuration from the module config
        llm_config = await ai.get_llm_configuration("survey_builder")
        if not llm_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Survey builder LLM configuration not found"
            )
        
        provider = llm_config.get("provider", "openai")
        model = llm_config.get("model", "gpt-4o-mini")
        temperature = survey_builder_config.temperature or 0.7
        max_tokens = survey_builder_config.max_tokens or 1000
        
        # Get Survey Builder personality system prompt - AI personality is required
        system_prompt = None
        
        # Check if session has personality_id
        if not session.get("personality_id"):
            logger.error("Survey builder session missing personality_id")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Survey Builder not properly configured: No AI personality assigned to session"
            )
        
        # Get AI personality prompt (this is required for survey builder)
        personality_data = await db.execute_fetchrow(
            "SELECT name, detailed_analysis_prompt FROM ai_personalities WHERE id = $1 AND is_active = true",
            [session["personality_id"]]
        )
        
        if not personality_data:
            logger.error(f"Survey Builder personality not found or inactive: {session['personality_id']}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Survey Builder not properly configured: AI personality not found or inactive"
            )
        
        if not personality_data["detailed_analysis_prompt"]:
            logger.error(f"Survey Builder personality '{personality_data['name']}' has no detailed_analysis_prompt")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Survey Builder not properly configured: AI personality '{personality_data['name']}' has no detailed prompt configured"
            )
        
        system_prompt = personality_data["detailed_analysis_prompt"]
        logger.info(f"Using Survey Builder personality prompt from: {personality_data['name']}")
        
        # Build AI messages without search context (survey builder doesn't need survey data)
        messages = ai.build_messages(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            current_question=message,
            search_context=None
        )
        
        # Generate AI response using survey builder module configuration
        ai_response = await ai.generate_response(
            messages=messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Process AI response - extract text if it's JSON formatted
        processed_response = ai_response
        try:
            # Try to parse as JSON in case AI returns structured response
            parsed_response = json.loads(ai_response)
            if isinstance(parsed_response, dict):
                # Extract conversational response if it's structured
                if "conversationalResponse" in parsed_response:
                    processed_response = parsed_response["conversationalResponse"]
                elif "response" in parsed_response:
                    processed_response = parsed_response["response"]
                # If it's just a plain object with text, keep original
        except (json.JSONDecodeError, TypeError):
            # Not JSON, use as-is (which is what we want for survey builder)
            pass
        
        logger.info(f"Survey builder AI response: {len(processed_response)} characters")
        
        # Save message pair to database
        user_msg_id, ai_msg_id = await db.save_message_pair(
            session_id=session_id,
            user_content=message,
            ai_content=processed_response,  # Use processed response (extracted text)
            ai_metadata={
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "module": "survey_builder",
                "personality_id": session.get("personality_id"),
                "original_response_length": len(ai_response),
                "processed_response_length": len(processed_response)
            }
        )
        
        # Update session title if first exchange
        if len(conversation_history) == 0:
            title = message[:50] + "..." if len(message) > 50 else message
            await db.update_session_title(session_id, title)
        
        logger.info(f"Survey builder chat response generated for session {session_id}")
        
        # Return simple response format for survey builder
        return {
            "response": processed_response,  # Return processed (extracted) response
            "user_message_id": user_msg_id,
            "ai_message_id": ai_msg_id,
            "session_id": session_id,
            "metadata": {
                "provider": provider,
                "model": model,
                "personality_used": session.get("personality_id") is not None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in survey builder chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate survey builder response"
        )


@router.post("/analyze-conversation")
async def analyze_survey_conversation(
    request_data: Dict[str, Any],
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Analyze a chat conversation using guided flow to determine next step
    
    Body:
    - **conversation**: List of chat messages
    - **survey_title**: Optional title for the survey
    
    Returns conversation state and next question or ready status for survey generation
    """
    try:
        conversation = request_data.get("conversation", [])
        
        if not conversation:
            # If no conversation, start with the first question
            return {
                "ready_to_generate": False,
                "next_question": "What is the main topic or purpose of your survey? Please describe what you want to study or measure.",
                "step": "topic",
                "completion_percentage": 0,
                "examples": ["customer satisfaction", "product feedback", "market research", "employee engagement"],
                "message": "Let's start building your survey step by step!"
            }
        
        # Get the next question based on conversation state
        next_question_data = survey_builder_service.get_next_question(conversation)
        
        if next_question_data["is_complete"]:
            # Conversation is complete, analyze for survey generation
            analysis = survey_generation_service.analyze_conversation_for_survey(conversation)
            
            logger.info(f"Analyzed complete conversation for user {current_user.id}: {len(conversation)} messages")
            
            return {
                "ready_to_generate": True,
                "analysis": analysis,
                "message": "Perfect! I have all the information needed to create your survey.",
                "completion_percentage": 100,
                "collected_info": next_question_data["collected_info"]
            }
        else:
            # Return next question to ask
            return {
                "ready_to_generate": False,
                "next_question": next_question_data["question"],
                "step": next_question_data.get("step", "unknown"),
                "completion_percentage": next_question_data["completion_percentage"],
                "examples": next_question_data.get("examples", []),
                "collected_info": next_question_data["collected_info"],
                "message": f"Question {int(next_question_data['completion_percentage'] / 15) + 1} of 7 - Let's gather the information I need."
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze conversation"
        )


@router.post("/conversation-state")
async def get_conversation_state(
    request_data: Dict[str, Any]
):
    """
    Get the current state of the conversation flow
    
    Body:
    - **conversation**: List of chat messages
    
    Returns the current conversation state, progress, and next required information
    """
    try:
        conversation = request_data.get("conversation", [])
        
        # Analyze conversation state
        state = survey_builder_service.analyze_conversation_state(conversation)
        
        return {
            "success": True,
            "state": state,
            "message": "Conversation state analyzed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation state: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze conversation state"
        )


@router.get("/conversation-steps")
async def get_conversation_steps(
):
    """
    Get the defined conversation flow steps
    
    Returns the list of conversation steps with questions and examples
    """
    try:
        return {
            "success": True,
            "steps": survey_builder_service.CONVERSATION_STEPS,
            "total_steps": len(survey_builder_service.CONVERSATION_STEPS),
            "message": "Conversation steps retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation steps: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation steps"
        )


@router.post("/generate-survey")
async def generate_survey_from_conversation(
    request_data: Dict[str, Any],
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Generate CSV/XLSX survey files from a chat conversation and create database records
    
    Body:
    - **conversation**: List of chat messages
    - **survey_title**: Optional title for the survey files
    
    Returns information about the generated survey files and database records
    """
    try:
        conversation = request_data.get("conversation", [])
        survey_title = request_data.get("survey_title", "Generated Survey")
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conversation cannot be empty"
            )
        
        # Validate that conversation is ready for generation
        is_complete = survey_builder_service.validate_survey_completion_intent(conversation)
        
        if not is_complete:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conversation does not indicate completion intent. Please continue the conversation."
            )
        
        # Analyze conversation and generate survey
        analysis = survey_generation_service.analyze_conversation_for_survey(conversation)
        
        # Generate survey file
        csv_path = survey_generation_service.generate_survey_files(
            analysis, survey_title
        )

        # Get file information
        csv_filename = os.path.basename(csv_path)

        # Create survey record in database
        survey_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Insert survey record
        survey_query = """
        INSERT INTO surveys (id, title, category, description, number_participants, 
                           total_files, processing_status, primary_language, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        await db.execute_command(survey_query, [
            survey_id,
            survey_title,
            "survey_builder_generated",
            f"Survey generated from conversation on {now.strftime('%Y-%m-%d %H:%M:%S')}",
            0,  # number_participants
            1,  # total_files (CSV only)
            "pending",  # Generated surveys start as pending until they get responses
            "en",
            now,
            now
        ])

        # Create survey file record
        csv_file_id = str(uuid.uuid4())

        # Insert CSV file record
        csv_file_query = """
        INSERT INTO survey_files (id, survey_id, filename, storage_path, file_size, 
                                created_at, updated_at, upload_date)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        await db.execute_command(csv_file_query, [
            csv_file_id,
            survey_id,
            csv_filename,
            csv_path,
            os.path.getsize(csv_path),
            now,
            now,
            now
        ])

        logger.info(f"Generated survey {survey_id} with file for user {current_user.id}: {csv_filename}")

        return {
            "success": True,
            "message": "Survey file generated successfully",
            "survey_id": survey_id,
            "files": {
                "csv": {
                    "id": csv_file_id,
                    "filename": csv_filename,
                    "path": csv_path,
                    "size": os.path.getsize(csv_path)
                }
            },
            "survey_info": {
                "id": survey_id,
                "title": survey_title,
                "questions_count": len(analysis["questions"]),
                "requirements": analysis["requirements"],
                "generated_at": analysis["metadata"]["generated_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating survey: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate survey files"
        )


@router.get("/generated-surveys")
async def get_generated_surveys(
    db: LightweightDBService = Depends(get_lightweight_db)):
    """
    Get all surveys generated through the survey builder
    
    Returns list of generated surveys with their files
    """
    try:
        # Get all survey builder generated surveys
        surveys_query = """
        SELECT s.id, s.title, s.category, s.description, s.number_participants,
               s.total_files, s.processing_status, s.primary_language, s.created_at,
               sf.id as file_id, sf.filename, sf.storage_path, sf.file_size
        FROM surveys s
        LEFT JOIN survey_files sf ON s.id = sf.survey_id
        WHERE s.category = 'survey_builder_generated'
        ORDER BY s.created_at DESC, sf.filename
        """
        
        results = await db.execute_query(surveys_query)
        
        # Group results by survey
        surveys = {}
        for row in results:
            survey_id = str(row["id"])
            if survey_id not in surveys:
                surveys[survey_id] = {
                    "id": survey_id,
                    "title": row["title"],
                    "category": row["category"],
                    "description": row["description"],
                    "number_participants": row["number_participants"],
                    "total_files": row["total_files"],
                    "processing_status": row["processing_status"],
                    "primary_language": row["primary_language"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "files": []
                }
            
            # Add file if it exists
            if row["file_id"]:
                surveys[survey_id]["files"].append({
                    "id": str(row["file_id"]),
                    "filename": row["filename"],
                    "storage_path": row["storage_path"],
                    "file_size": row["file_size"]
                })
        
        return {
            "success": True,
            "surveys": list(surveys.values())
        }
        
    except Exception as e:
        logger.error(f"Error getting generated surveys: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve generated surveys"
        )


@router.get("/download-survey-file/{file_id}")
async def download_survey_file(
    file_id: str,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Download a generated survey file by file ID
    
    - **file_id**: ID of the survey file to download
    
    Returns the requested survey file for download
    """
    try:
        # Get file information from database
        file_query = """
        SELECT sf.filename, sf.storage_path, s.category
        FROM survey_files sf
        JOIN surveys s ON sf.survey_id = s.id
        WHERE sf.id = $1 AND s.category = 'survey_builder_generated'
        """
        
        file_data = await db.execute_fetchrow(file_query, [file_id])
        
        if not file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey file not found"
            )
        
        file_path = file_data["storage_path"]
        filename = file_data["filename"]
        
        # Check if file exists on disk
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found on disk"
            )
        
        # Determine media type
        media_type = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if filename.endswith('.xlsx')
            else "text/csv"
        )
        
        logger.info(f"User {current_user.id} downloading survey file: {filename}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading survey file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download file"
        )


@router.get("/download/{filename}")
async def download_survey_file(
    filename: str,
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Download a generated survey file
    
    - **filename**: Name of the file to download
    
    Returns the requested survey file for download
    """
    try:
        # Validate filename
        if not filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Only CSV and XLSX files are supported."
            )
        
        # Check if file exists
        file_path = os.path.join(survey_generation_service.output_directory, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Determine media type
        media_type = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" 
            if filename.endswith('.xlsx') 
            else "text/csv"
        )
        
        logger.info(f"User {current_user.id} downloading survey file: {filename}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download file"
        )


@router.get("/files")
async def list_generated_files(
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get list of all generated survey files
    
    Returns list of available survey files with metadata
    """
    try:
        files = survey_generation_service.get_generated_files()
        
        logger.info(f"Listed {len(files)} generated files for user {current_user.id}")
        
        return {
            "files": files,
            "total_count": len(files)
        }
        
    except Exception as e:
        logger.error(f"Error listing generated files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list generated files"
        )


@router.delete("/files/{filename}")
async def delete_survey_file(
    filename: str,
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Delete a generated survey file
    
    - **filename**: Name of the file to delete
    
    Note: This endpoint is available to all authenticated users.
    In production, you might want to add additional access controls.
    """
    try:
        # Validate filename
        if not filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type"
            )
        
        file_path = os.path.join(survey_generation_service.output_directory, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        # Delete the file
        os.remove(file_path)
        
        logger.info(f"User {current_user.id} deleted survey file: {filename}")
        
        return SuccessResponse(
            success=True,
            message=f"File {filename} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file"
        )


@router.get("/view-file/{file_id}")
async def view_survey_file_content(
    file_id: str,
    db: LightweightDBService = Depends(get_lightweight_db),
):
    """
    Get file content for viewing (headers and rows) from a generated survey file
    
    Parameters:
    - **file_id**: ID of the survey file to view
    
    Returns file content as JSON with headers and rows for display
    """
    try:
        # Get file information from database
        file_query = """
        SELECT sf.*, s.id as survey_id, s.title as survey_title
        FROM survey_files sf
        INNER JOIN surveys s ON sf.survey_id = s.id
        WHERE sf.id = $1 AND s.category = 'survey_builder_generated'
        """
        
        file_data = await db.execute_fetchrow(file_query, [file_id])
        
        if not file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey file not found"
            )
        
        # Check if file exists on disk
        storage_path = file_data["storage_path"]
        if not storage_path or not os.path.exists(storage_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File data not found on disk"
            )
        
        try:
            # Read file content based on extension
            if storage_path.endswith('.xlsx') or storage_path.endswith('.xls'):
                import pandas as pd
                try:
                    df = pd.read_excel(storage_path, engine='openpyxl')
                except Exception:
                    try:
                        df = pd.read_excel(storage_path, engine='xlrd')
                    except Exception:
                        df = pd.read_excel(storage_path)
                
                # Replace NaN values with None for JSON compatibility
                df = df.where(pd.notnull(df), None)
                
                headers = df.columns.tolist()
                rows_data = df.values.tolist()
                
                return {
                    "filename": file_data["filename"],
                    "headers": headers,
                    "rows": rows_data,
                    "total_rows": len(df),
                    "total_columns": len(df.columns)
                }
                
            elif storage_path.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(storage_path)
                
                # Replace NaN values with None for JSON compatibility
                df = df.where(pd.notnull(df), None)
                
                headers = df.columns.tolist()
                rows_data = df.values.tolist()
                
                return {
                    "filename": file_data["filename"],
                    "headers": headers,
                    "rows": rows_data,
                    "total_rows": len(df),
                    "total_columns": len(df.columns)
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported file format"
                )
                
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="File reading libraries not available"
            )
        except Exception as e:
            logger.error(f"Error reading file content: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to read file content"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error viewing survey file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to view file"
        )


@router.post("/update-survey-status/{survey_id}")
async def update_survey_processing_status(
    survey_id: str,
    request_data: Dict[str, Any],
    db: LightweightDBService = Depends(get_lightweight_db),
):
    """
    Update the processing status of a survey
    
    Body:
    - **status**: "pending" or "completed"
    """
    try:
        status_value = request_data.get("status", "").lower()
        
        if status_value not in ["pending", "completed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Status must be 'pending' or 'completed'"
            )
        
        # Check if survey exists
        survey_check = await db.execute_fetchrow(
            "SELECT id FROM surveys WHERE id = $1",
            [survey_id]
        )
        
        if not survey_check:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        
        # Update the status
        success = await update_survey_status(survey_id, status_value)
        
        if success:
            return {
                "success": True,
                "message": f"Survey status updated to {status_value}",
                "survey_id": survey_id,
                "status": status_value
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update survey status"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating survey status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update survey status"
        )


@router.post("/auto-update-statuses")
async def auto_update_all_survey_statuses(
):
    """
    Automatically update all survey statuses based on response data
    """
    try:
        updates_made = await auto_update_survey_statuses()
        
        return {
            "success": True,
            "message": f"Auto-update completed: {updates_made} surveys updated",
            "updates_made": updates_made
        }
        
    except Exception as e:
        logger.error(f"Error in auto-update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to auto-update survey statuses"
        )