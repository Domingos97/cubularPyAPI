from typing import List, Optional, Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import os
import shutil
import uuid
import math
import json

from app.core.lightweight_dependencies import get_current_regular_user, get_current_admin_user, SimpleUser
from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.utils.logging import get_logger
from typing import List, Optional
from uuid import UUID

from app.models.schemas import (
    UserSurveyAccess as UserSurveyAccessSchema,
    UserSurveyFileAccess as UserSurveyFileAccessSchema,
    UserSurveyAccessCreate,
    UserSurveyFileAccessCreate,
    AccessType,
    BulkAccessGrant, Survey, SurveyCreate
)
from app.services.access_control_service import access_control_service

logger = get_logger(__name__)

router = APIRouter()


def sanitize_for_json(data: Any) -> Any:
    """
    Sanitize data for JSON serialization by handling NaN, UUID, and other problematic types
    """
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, uuid.UUID):
        return str(data)
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data


@router.post(
    "/",
    response_model=Survey,
    summary="Create a new survey"
)
async def create_survey(
    survey_data: SurveyCreate,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Create a new survey.
    Only authenticated users can create surveys.
    """
    try:
        # Create the survey using raw SQL
        import uuid
        from datetime import datetime
        
        survey_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        query = """
        INSERT INTO surveys (id, title, category, description, ai_suggestions, 
                           number_participants, total_files, processing_status, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        
        params = [
            survey_id,
            survey_data.title,
            survey_data.category,
            survey_data.description,
            survey_data.ai_suggestions or [],
            survey_data.number_participants or 0,
            0,  # total_files starts at 0
            "completed",  # Uploaded surveys have data and are considered complete
            now,
            now
        ]
        
        await db.execute_command(query, params)
        
        # Fetch the created survey
        created_survey = await db.execute_fetchrow(
            """
            SELECT id, title, category, description, ai_suggestions, 
                   number_participants, total_files, processing_status, created_at, updated_at
            FROM surveys WHERE id = $1
            """,
            [survey_id]
        )
        
        if not created_survey:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create survey"
            )
        
        # Convert to Survey schema format
        survey_response = Survey(
            id=str(created_survey["id"]),
            title=created_survey["title"],
            category=created_survey["category"] or "",
            description=created_survey["description"] or "",
            ai_suggestions=created_survey.get("ai_suggestions", []),
            number_participants=created_survey.get("number_participants", 0),
            total_files=created_survey.get("total_files", 0),
            created_at=created_survey["created_at"].isoformat() if created_survey.get("created_at") else "",
            updated_at=created_survey["updated_at"].isoformat() if created_survey.get("updated_at") else "",
            # Backward compatibility fields
            fileid=None,
            filename=None,
            createdat=created_survey["created_at"].isoformat() if created_survey.get("created_at") else "",
            storage_path=None,
            primary_language=None,
            files=[]
        )
        
        logger.info(f"Survey created successfully: {survey_id} by user {current_user.id}")
        return survey_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating survey: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create survey: {str(e)}"
        )


@router.post(
    "/{survey_id}/suggestions",
    summary="Generate AI suggestions for a survey"
)
async def generate_survey_suggestions(
    survey_id: str,
    request_data: dict,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Generate AI suggestions for survey analysis using AI personalities and module configuration.
    """
    try:
        # Check if survey exists and user has access
        survey_data = await db.execute_fetchrow(
            "SELECT id, title, description, category FROM surveys WHERE id = $1",
            [survey_id]
        )
        
        if not survey_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        
        # Extract data from request
        personality_id = request_data.get("personalityId")
        file_content = request_data.get("fileContent", {})
        
        # Get module configuration for survey suggestions generation
        from app.services.module_configuration_service import module_configuration_service
        
        # Use the lightweight database service to get module configuration
        module_config_query = """
        SELECT mc.model, mc.temperature, mc.max_tokens, mc.ai_personality_id,
               ls.provider
        FROM module_configurations mc
        LEFT JOIN llm_settings ls ON mc.llm_setting_id = ls.id
        WHERE mc.module_name = $1 AND mc.active = true
        """
        
        module_config_data = await db.execute_fetchrow(module_config_query, ["survey_suggestions_generation"])
        
        if not module_config_data:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Survey suggestions generation module not configured"
            )
        
        module_config = {
            "provider": module_config_data["provider"],
            "model": module_config_data["model"],
            "temperature": float(module_config_data.get("temperature", 0.7)) if module_config_data.get("temperature") is not None else 0.7,
            "max_tokens": module_config_data.get("max_tokens", 1000)
        }
        
        # Get AI personality suggestions prompt from module configuration
        suggestions_prompt = None
        ai_personality_id = module_config_data.get("ai_personality_id")
        
        if ai_personality_id:
            personality_data = await db.execute_fetchrow(
                "SELECT name, suggestions_prompt FROM ai_personalities WHERE id = $1 AND is_active = true",
                [ai_personality_id]
            )
            if personality_data and personality_data["suggestions_prompt"]:
                suggestions_prompt = personality_data["suggestions_prompt"]
        
        # If no AI personality or suggestions prompt configured, raise error
        if not suggestions_prompt:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No AI personality suggestions prompt configured for survey suggestions generation"
            )
        
        # Build file context for the prompt
        file_context = ""
        headers = file_content.get("headers", [])
        sample_rows = file_content.get("sampleRows", [])
        
        if headers:
            file_context += f"Available data columns: {', '.join(headers)}. "
            
        if sample_rows:
            file_context += f"Sample data includes: {sample_rows[:2]}. "
        
        if not file_context:
            file_context = "No specific file data available yet. "
        
        # Use the suggestions prompt from AI personality as-is
        # The AI personality should handle all formatting and instructions
        # We'll provide the survey context as user input
        survey_context = f"""Survey Details:
- Title: {survey_data.get("title", "No title provided")}
- Description: {survey_data.get("description", "No description provided")}
- Category: {survey_data.get("category", "General")}
- File Context: {file_context}"""
        
        # Generate AI suggestions using the configured AI service
        from app.services.simple_ai_service import simple_ai
        
        # Add language instruction to the suggestions prompt
        user_language = current_user.language or 'English'
        language_instruction = f"\n\nIMPORTANT NOTES: You need to always respond in user language: {user_language}"
        enhanced_suggestions_prompt = suggestions_prompt + language_instruction
        
        messages = [
            {
                "role": "system",
                "content": enhanced_suggestions_prompt
            },
            {
                "role": "user", 
                "content": survey_context
            }
        ]
        
        # Use module configuration for AI parameters
        ai_response = await simple_ai.generate_response(
            messages=messages,
            provider=module_config["provider"],
            model=module_config["model"],
            temperature=module_config.get("temperature", 0.7),
            max_tokens=module_config.get("max_tokens", 1000)
        )
        
        # Parse AI response into suggestions list
        suggestions = []
        if ai_response:
            try:
                # Try to parse as JSON first (for the new format)
                import json
                parsed_response = json.loads(ai_response.strip())
                if isinstance(parsed_response, list):
                    suggestions = [str(item) for item in parsed_response if item and len(str(item).strip()) > 10]
                else:
                    # Fallback to old parsing method
                    lines = ai_response.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and len(line) > 10:
                            # Remove numbering, bullets, or prefixes
                            line = line.lstrip('0123456789.-•*').strip()
                            if line:
                                suggestions.append(line)
            except json.JSONDecodeError:
                # Fallback to old parsing method if JSON parsing fails
                lines = ai_response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and len(line) > 10:
                        # Remove numbering, bullets, or prefixes
                        line = line.lstrip('0123456789.-•*').strip()
                        if line:
                            suggestions.append(line)
        
        # Fallback to default suggestions if AI didn't generate any
        if not suggestions:
            logger.warning("AI did not generate valid suggestions, using fallback")
            suggestions = [
                "How can these survey insights drive revenue growth and market expansion?",
                "What operational improvements can be implemented based on customer feedback?", 
                "How do these findings impact competitive positioning and strategic planning?"
            ]
        
        # Ensure we have exactly 3 suggestions (trim or pad as needed)
        if len(suggestions) > 3:
            suggestions = suggestions[:3]
        elif len(suggestions) < 3:
            # Pad with generic strategic questions if needed
            fallback_questions = [
                "What strategic opportunities does this data reveal?",
                "How can these insights inform business decision-making?",
                "What competitive advantages can be derived from these findings?"
            ]
            while len(suggestions) < 3:
                for fallback in fallback_questions:
                    if fallback not in suggestions:
                        suggestions.append(fallback)
                        break
                if len(suggestions) >= 3:
                    break
        
        # Update the survey with generated suggestions
        from datetime import datetime
        await db.execute_command(
            "UPDATE surveys SET ai_suggestions = $1, updated_at = $2 WHERE id = $3",
            [suggestions, datetime.utcnow(), survey_id]
        )
        
        logger.info(f"Generated {len(suggestions)} strategic AI suggestions for survey {survey_id} using {module_config['provider']} {module_config['model']}")
        
        return JSONResponse(
            content={
                "message": "Suggestions generated successfully",
                "suggestions": suggestions
            },
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating suggestions for survey {survey_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


@router.post(
    "/{survey_id}/files",
    summary="Upload a file to a survey"
)
async def upload_survey_file(
    survey_id: str,
    file: UploadFile = File(...),
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Upload a CSV or Excel file to a survey.
    Only administrators can upload files to surveys.
    """
    try:
        # Check if user is admin - only admins can upload files
        is_admin = current_user.role in ["admin", "super_admin"]
        
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can upload files to surveys"
            )
        
        # Check if survey exists
        survey_data = await db.execute_fetchrow(
            "SELECT id, title FROM surveys WHERE id = $1",
            [survey_id]
        )
        
        if not survey_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV and Excel files are supported"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Use the updated survey upload helper with embedding generation
        file_info = await upload_file_to_survey(
            db=db,
            survey_id=survey_id,
            file_content=file_content,
            filename=file.filename,
            user_id=current_user.id
        )
        
        logger.info(f"File uploaded successfully to survey {survey_id}: {file.filename}")
        
        return JSONResponse(
            content={
                "message": "File uploaded successfully",
                "file": file_info
            },
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file to survey {survey_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )


async def upload_file_to_survey(db: LightweightDBService, survey_id: str, file_content: bytes, filename: str, user_id: str):
    """
    Helper function to upload file to survey using lightweight DB
    Now creates proper directory structure and generates embeddings
    """
    import uuid
    from datetime import datetime
    from pathlib import Path
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Create survey and file-specific directory structure
    survey_dir = Path("survey_data") / survey_id
    file_dir = survey_dir / file_id  # Create file-specific subdirectory
    file_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file with standard name
    file_extension = Path(filename).suffix.lower()
    if file_extension == '.csv':
        storage_filename = 'original_file.csv'
    else:
        storage_filename = 'original_file.xlsx'
    
    storage_path = file_dir / storage_filename
    
    # Write file to disk
    with open(storage_path, 'wb') as f:
        f.write(file_content)
    
    # Get file size
    file_size = len(file_content)
    
    # Insert file record into database
    now = datetime.utcnow()
    
    query = """
    INSERT INTO survey_files (id, survey_id, filename, file_size, storage_path, created_at, updated_at)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    """
    
    params = [
        file_id,
        survey_id,
        filename,
        file_size,
        str(storage_path),
        now,
        now
    ]
    
    await db.execute_command(query, params)
    
    # Update survey file count and mark as completed (has data)
    await db.execute_command(
        """
        UPDATE surveys 
        SET total_files = (
            SELECT COUNT(*) FROM survey_files WHERE survey_id = $1
        ),
        processing_status = 'completed',
        updated_at = $2
        WHERE id = $1
        """,
        [survey_id, now]
    )
    
    # Process the file and generate embeddings
    try:
        logger.info(f"Starting file processing and embedding generation for survey {survey_id}, file {file_id}")
        
        # Import the file processor and survey indexing service
        from app.services.survey_service import FileProcessor
        from app.services.survey_indexing_service import survey_indexing_service
        
        # Process the uploaded file
        processed_data = await FileProcessor.process_survey_file(
            str(storage_path), 
            filename
        )
        
        # Generate embeddings and create pickle file
        indexing_success = await survey_indexing_service.index_survey_file(
            survey_id=survey_id,
            file_id=file_id,
            processed_data=processed_data,
            db=None  # No SQLAlchemy session available in lightweight mode
        )
        
        if indexing_success:
            logger.info(f"Successfully created embeddings and pickle file for survey {survey_id}, file {file_id}")
        else:
            logger.warning(f"Failed to create embeddings for survey {survey_id}, file {file_id}")
            
    except Exception as embedding_error:
        # Don't fail the entire upload if embedding generation fails
        logger.error(f"Embedding generation failed for survey {survey_id}, file {file_id}: {str(embedding_error)}")
        # Log the full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
    
    return {
        "id": file_id,
        "filename": filename,
        "file_size": file_size,
        "storage_path": str(storage_path),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat()
    }

@router.get(
    "/{survey_id}/access-check",
    summary="Check user access to survey and return accessible files"
)
async def check_survey_access(
    survey_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Check if user has access to survey and return accessible files with processing status.
    Used by frontend chat component to determine available files.
    """
    try:
        # Check if user is admin - admins have access to all surveys
        is_admin = current_user.role in ["admin", "super_admin"]
        
        if is_admin:
            # Admin users get access to any survey that exists
            survey_data = await db.execute_fetchrow(
                "SELECT id, title FROM surveys WHERE id = $1",
                [survey_id]
            )
        else:
            # Regular users need explicit access
            survey_data = await db.execute_fetchrow(
                """
                SELECT s.id, s.title
                FROM surveys s
                INNER JOIN user_survey_access usa ON s.id = usa.survey_id
                WHERE s.id = $1 AND usa.user_id = $2 AND usa.is_active = true
                """,
                [survey_id, current_user.id]
            )
        
        if not survey_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found or access denied"
            )
        
        # Get accessible files for this survey
        if is_admin:
            files_data = await db.execute_query(
                """
                SELECT id, filename, file_size, storage_path, survey_id
                FROM survey_files
                WHERE survey_id = $1
                ORDER BY filename
                """,
                [survey_id]
            )
        else:
            files_data = await db.execute_query(
                """
                SELECT sf.id, sf.filename, sf.file_size, sf.storage_path, sf.survey_id
                FROM survey_files sf
                INNER JOIN surveys s ON sf.survey_id = s.id
                INNER JOIN user_survey_access usa ON s.id = usa.survey_id
                WHERE sf.survey_id = $1 AND usa.user_id = $2 AND usa.is_active = true
                ORDER BY sf.filename
                """,
                [survey_id, current_user.id]
            )
        
        # Check processing status for each file
        accessible_files = []
        for file_data in files_data:
            file_id = str(file_data["id"])
            file_dir = os.path.join("survey_data", survey_id, file_id)
            
            # For admins: Apply OldPyAPI simple approach - allow access to all files
            # For regular users: Check processing status
            if is_admin:
                # Admins get access to all files, following OldPyAPI simple approach
                # Check if pickle exists but don't block access if it doesn't
                pickle_file = os.path.join(file_dir, "survey_data.pkl")
                is_processed = os.path.exists(pickle_file)
                
                accessible_files.append({
                    "id": file_id,
                    "fileId": file_id,  # Also include for compatibility
                    "filename": file_data["filename"],
                    "file_size": file_data["file_size"] or 0,
                    "isProcessed": is_processed,
                    "processingStatus": "completed" if is_processed else "processing_available"  # Admin can access regardless
                })
            else:
                # Regular users need processed files
                pickle_file = os.path.join(file_dir, "survey_data.pkl")
                is_processed = os.path.exists(pickle_file)
                
                # Only include processed files for regular users
                if is_processed:
                    accessible_files.append({
                        "id": file_id,
                        "fileId": file_id,
                        "filename": file_data["filename"],
                        "file_size": file_data["file_size"] or 0,
                        "isProcessed": True,
                        "processingStatus": "completed"
                    })
        
        return {
            "surveyId": survey_id,
            "hasAccess": True,
            "accessibleFiles": accessible_files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking access for survey {survey_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check survey access: {str(e)}"
        )


@router.get(
    "/{survey_id}",
    response_model=Survey,
    summary="Get a single survey by ID"
)
async def get_survey(
    survey_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get a specific survey by ID.
    Returns survey basic information including ai_suggestions.
    """
    try:
        # Check if user is admin - admins have access to all surveys
        is_admin = current_user.role in ["admin", "super_admin"]
        
        if is_admin:
            # Admin users get access to any survey that exists
            survey_data = await db.execute_fetchrow(
                """
                SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
                       s.number_participants, s.created_at, s.updated_at,
                       COUNT(sf.id) as total_files
                FROM surveys s
                LEFT JOIN survey_files sf ON s.id = sf.survey_id
                WHERE s.id = $1
                GROUP BY s.id, s.title, s.category, s.description, s.ai_suggestions, 
                         s.number_participants, s.created_at, s.updated_at
                """,
                [survey_id]
            )
        else:
            # Regular users need explicit access
            survey_data = await db.execute_fetchrow(
                """
                SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
                       s.number_participants, s.created_at, s.updated_at,
                       COUNT(sf.id) as total_files
                FROM surveys s
                LEFT JOIN survey_files sf ON s.id = sf.survey_id
                INNER JOIN user_survey_access usa ON s.id = usa.survey_id
                WHERE s.id = $1 AND usa.user_id = $2 AND usa.is_active = true
                GROUP BY s.id, s.title, s.category, s.description, s.ai_suggestions, 
                         s.number_participants, s.created_at, s.updated_at
                """,
                [survey_id, current_user.id]
            )
        
        if not survey_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found or access denied"
            )
        
        # Create survey object
        survey = Survey(
            id=str(survey_data["id"]),
            title=survey_data["title"],
            category=survey_data["category"] or "",
            description=survey_data["description"] or "",
            ai_suggestions=survey_data.get("ai_suggestions") or [],
            number_participants=survey_data.get("number_participants", 0),
            total_files=survey_data.get("total_files", 0),
            created_at=survey_data["created_at"].isoformat() if survey_data["created_at"] else "",
            updated_at=survey_data["updated_at"].isoformat() if survey_data["updated_at"] else ""
        )
        
        return survey
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching survey {survey_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch survey: {str(e)}"
        )


@router.get(
    "/{survey_id}/with-files",
    summary="Get survey with its files"
)
async def get_survey_with_files(
    survey_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get a specific survey along with its associated files.
    """
    try:
        # Check if user is admin - admins have access to all surveys
        is_admin = current_user.role in ["admin", "super_admin"]
        
        if is_admin:
            # Admin users get access to any survey that exists
            survey_data = await db.execute_fetchrow(
                """
                SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
                       s.number_participants, s.created_at, s.updated_at,
                       COUNT(sf.id) as total_files
                FROM surveys s
                LEFT JOIN survey_files sf ON s.id = sf.survey_id
                WHERE s.id = $1
                GROUP BY s.id, s.title, s.category, s.description, s.ai_suggestions, 
                         s.number_participants, s.created_at, s.updated_at
                """,
                [survey_id]
            )
        else:
            # Regular users need explicit access
            survey_data = await db.execute_fetchrow(
                """
                SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
                       s.number_participants, s.created_at, s.updated_at,
                       COUNT(sf.id) as total_files
                FROM surveys s
                LEFT JOIN survey_files sf ON s.id = sf.survey_id
                INNER JOIN user_survey_access usa ON s.id = usa.survey_id
                WHERE s.id = $1 AND usa.user_id = $2 AND usa.is_active = true
                GROUP BY s.id, s.title, s.category, s.description, s.ai_suggestions, 
                         s.number_participants, s.created_at, s.updated_at
                """,
                [survey_id, current_user.id]
            )
        
        if not survey_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found or access denied"
            )
        
        # Get the survey files
        files_data = await db.execute_query(
            """
            SELECT id, filename, file_size, storage_path, created_at, updated_at
            FROM survey_files
            WHERE survey_id = $1
            ORDER BY created_at DESC
            """,
            [survey_id]
        )
        
        files = []
        for file_data in files_data:
            files.append({
                "id": str(file_data["id"]),
                "filename": file_data["filename"],
                "file_size": file_data.get("file_size"),
                "storage_path": file_data.get("storage_path"),
                "created_at": file_data["created_at"].isoformat() if file_data.get("created_at") else "",
                "updated_at": file_data["updated_at"].isoformat() if file_data.get("updated_at") else ""
            })
        
        return {
            "id": str(survey_data["id"]),
            "title": survey_data["title"],
            "category": survey_data["category"] or "",
            "description": survey_data["description"] or "",
            "ai_suggestions": survey_data.get("ai_suggestions", False),
            "number_participants": survey_data.get("number_participants", 0),
            "total_files": survey_data.get("total_files", 0),
            "created_at": survey_data["created_at"].isoformat() if survey_data.get("created_at") else "",
            "updated_at": survey_data["updated_at"].isoformat() if survey_data.get("updated_at") else "",
            "files": files
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting survey with files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve survey with files"
        )


@router.get(
    "/{survey_id}/files/{file_id}/rows",
    summary="Get survey file data rows"
)
async def get_survey_file_rows(
    survey_id: str,
    file_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get the data rows from a survey file (Excel/CSV).
    Returns the file content as JSON with rows and columns.
    """
    try:
        # Check if user has access to the survey
        is_admin = current_user.role in ["admin", "super_admin"]
        
        if is_admin:
            # Admin users get access to any survey file that exists
            file_data = await db.execute_fetchrow(
                """
                SELECT sf.*, s.title as survey_title
                FROM survey_files sf
                INNER JOIN surveys s ON sf.survey_id = s.id
                WHERE sf.id = $1 AND sf.survey_id = $2
                """,
                [file_id, survey_id]
            )
        else:
            # Regular users need explicit access
            file_data = await db.execute_fetchrow(
                """
                SELECT sf.*, s.title as survey_title
                FROM survey_files sf
                INNER JOIN surveys s ON sf.survey_id = s.id
                INNER JOIN user_survey_access usa ON s.id = usa.survey_id
                WHERE sf.id = $1 AND sf.survey_id = $2 
                AND usa.user_id = $3 AND usa.is_active = true
                """,
                [file_id, survey_id, current_user.id]
            )
        
        if not file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey file not found or access denied"
            )
        
        # Get the file path
        storage_path = file_data["storage_path"]
        if not storage_path or not os.path.exists(storage_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File data not found on disk"
            )
        
        try:
            # Try to read the file data
            if storage_path.endswith('.xlsx') or storage_path.endswith('.xls'):
                # For Excel files, try to read with pandas if available
                try:
                    import pandas as pd
                    # Try different engines for Excel files
                    try:
                        # Try openpyxl first (for .xlsx files)
                        df = pd.read_excel(storage_path, engine='openpyxl')
                    except Exception:
                        try:
                            # Try xlrd for older .xls files
                            df = pd.read_excel(storage_path, engine='xlrd')
                        except Exception:
                            # Fallback to default engine
                            df = pd.read_excel(storage_path)
                    
                    # Replace NaN values with None/null for JSON compatibility
                    df = df.where(pd.notnull(df), None)
                    
                    # Convert to array format expected by frontend (array of arrays, not objects)
                    headers = df.columns.tolist()
                    rows_data = df.values.tolist()  # This gives us array of arrays
                    
                    # Sanitize data for JSON serialization
                    data = sanitize_for_json({
                        "filename": file_data["filename"],
                        "headers": headers,  # Changed from "columns" to "headers" to match frontend
                        "rows": rows_data,
                        "total_rows": len(df),
                        "total_columns": len(df.columns)
                    })
                    return data
                except ImportError:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Excel file reading not supported - pandas not available"
                    )
            elif storage_path.endswith('.csv'):
                # For CSV files
                try:
                    import pandas as pd
                    df = pd.read_csv(storage_path)
                    
                    # Replace NaN values with None/null for JSON compatibility
                    df = df.where(pd.notnull(df), None)
                    
                    # Convert to array format expected by frontend (array of arrays, not objects)
                    headers = df.columns.tolist()
                    rows_data = df.values.tolist()  # This gives us array of arrays
                    
                    # Sanitize data for JSON serialization
                    data = sanitize_for_json({
                        "filename": file_data["filename"],
                        "headers": headers,  # Changed from "columns" to "headers" to match frontend
                        "rows": rows_data,
                        "total_rows": len(df),
                        "total_columns": len(df.columns)
                    })
                    return data
                except ImportError:
                    # Fallback to basic CSV reading
                    import csv
                    headers = []
                    rows_data = []
                    with open(storage_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)  # Use csv.reader instead of DictReader for arrays
                        headers = next(reader, [])  # First row as headers
                        rows_data = list(reader)    # Remaining rows as arrays
                    
                    # Sanitize data for JSON serialization
                    data = sanitize_for_json({
                        "filename": file_data["filename"],
                        "headers": headers,  # Changed from "columns" to "headers"
                        "rows": rows_data,
                        "total_rows": len(rows_data),
                        "total_columns": len(headers)
                    })
                    return data
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported file format"
                )
                
        except Exception as file_error:
            logger.error(f"Error reading file {storage_path}: {str(file_error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error reading file data: {str(file_error)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting survey file rows: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve survey file data"
        )


@router.delete(
    "/{survey_id}",
    summary="Delete a survey and all associated data"
)
async def delete_survey(
    survey_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Delete a survey and all associated data including:
    - Survey record
    - All survey files and their physical files
    - All chat sessions associated with this survey (for all users)
    - All chat messages from those sessions
    """
    try:
        # Check if user has permission to delete this survey
        # Only admins can delete surveys
        is_admin = current_user.role in ["admin", "super_admin"]
        
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can delete surveys"
            )
        
        # Check if survey exists
        survey_data = await db.execute_fetchrow(
            "SELECT id, title FROM surveys WHERE id = $1",
            [survey_id]
        )
        
        if not survey_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        
        logger.info(f"Starting deletion of survey {survey_id} by admin user {current_user.id}")
        
        # Get all chat sessions associated with this survey (across all users)
        # survey_ids is stored as text[] in the database
        chat_sessions = await db.execute_query(
            """
            SELECT id FROM chat_sessions 
            WHERE $1 = ANY(survey_ids)
            """,
            [survey_id]
        )
        
        # Delete all chat messages from sessions associated with this survey
        if chat_sessions:
            session_ids = [session["id"] for session in chat_sessions]
            logger.info(f"Deleting {len(session_ids)} chat sessions associated with survey {survey_id}")
            
            for session_id in session_ids:
                # Delete messages for this session
                await db.execute_command(
                    "DELETE FROM chat_messages WHERE session_id = $1",
                    [session_id]
                )
                
                # Delete the session
                await db.execute_command(
                    "DELETE FROM chat_sessions WHERE id = $1",
                    [session_id]
                )
        
        # Get all survey files to delete physical files
        survey_files = await db.execute_query(
            "SELECT id, storage_path FROM survey_files WHERE survey_id = $1",
            [survey_id]
        )
        
        # Delete physical files
        if survey_files:
            logger.info(f"Deleting {len(survey_files)} files associated with survey {survey_id}")
            
            for file_record in survey_files:
                storage_path = file_record.get("storage_path")
                if storage_path and os.path.exists(storage_path):
                    try:
                        os.remove(storage_path)
                        logger.info(f"Deleted physical file: {storage_path}")
                        
                        # Also delete metadata file if it exists
                        metadata_path = os.path.join(
                            os.path.dirname(storage_path),
                            f"{file_record['id']}_metadata.json"
                        )
                        if os.path.exists(metadata_path):
                            os.remove(metadata_path)
                            logger.info(f"Deleted metadata file: {metadata_path}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to delete physical file {storage_path}: {e}")
        
        # Delete user survey file access records for files of this survey (do this before deleting files)
        await db.execute_command(
            """
            DELETE FROM user_survey_file_access 
            WHERE survey_file_id IN (
                SELECT id FROM survey_files WHERE survey_id = $1
            )
            """,
            [survey_id]
        )
        
        # Delete survey files from database
        await db.execute_command(
            "DELETE FROM survey_files WHERE survey_id = $1",
            [survey_id]
        )
        
        # Delete user survey access records
        await db.execute_command(
            "DELETE FROM user_survey_access WHERE survey_id = $1",
            [survey_id]
        )
        
        # Delete the survey directory if it exists
        survey_dir = os.path.join("survey_data", survey_id)
        if os.path.exists(survey_dir):
            try:
                shutil.rmtree(survey_dir)
                logger.info(f"Deleted survey directory: {survey_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete survey directory {survey_dir}: {e}")
        
        # Finally, delete the survey itself
        await db.execute_command(
            "DELETE FROM surveys WHERE id = $1",
            [survey_id]
        )
        
        logger.info(f"Successfully deleted survey {survey_id} and all associated data")
        
        return JSONResponse(
            content={"message": "Survey deleted successfully"},
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting survey {survey_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete survey: {str(e)}"
        )


@router.delete(
    "/files/{file_id}",
    summary="Delete a survey file"
)
async def delete_survey_file(
    file_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Delete a specific survey file.
    Only administrators can delete files.
    """
    try:
        # Check if user has permission to delete files
        # Only admins can delete files
        is_admin = current_user.role in ["admin", "super_admin"]
        
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can delete files"
            )
        
        # Check if file exists
        file_data = await db.execute_fetchrow(
            "SELECT id, filename, storage_path, survey_id FROM survey_files WHERE id = $1",
            [file_id]
        )
        
        if not file_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )
        
        logger.info(f"Starting deletion of file {file_id} ({file_data['filename']}) by admin user {current_user.id}")
        
        # Delete the entire file directory from survey_data (contains all files: original, pickle, embeddings, metadata)
        file_dir = os.path.join("survey_data", str(file_data["survey_id"]), file_id)
        if os.path.exists(file_dir):
            try:
                shutil.rmtree(file_dir)
                logger.info(f"Deleted file directory and all contents: {file_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete file directory {file_dir}: {e}")
        else:
            logger.info(f"File directory {file_dir} does not exist, skipping directory deletion")
        
        # Also delete the storage_path file if it exists outside the file directory (fallback)
        storage_path = file_data.get("storage_path")
        if storage_path and os.path.exists(storage_path):
            try:
                os.remove(storage_path)
                logger.info(f"Deleted fallback storage file: {storage_path}")
            except Exception as e:
                logger.warning(f"Failed to delete fallback storage file {storage_path}: {e}")
        
        
        # Delete user file access records
        await db.execute_command(
            "DELETE FROM user_survey_file_access WHERE survey_file_id = $1",
            [file_id]
        )
        
        # Delete file from database
        await db.execute_command(
            "DELETE FROM survey_files WHERE id = $1",
            [file_id]
        )
        
        logger.info(f"Successfully deleted file {file_id}")
        
        return JSONResponse(
            content={"message": "File deleted successfully"},
            status_code=200
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}"
        )


@router.post(
    "/semantic-chat",
    summary="Survey semantic chat - AI analysis with session support"
)
async def survey_semantic_chat(
    request_data: dict,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Perform semantic chat analysis on survey data with session support.
    Expected request body:
    {
        "question": "string",
        "surveyIds": ["uuid1", "uuid2"],
        "selectedFileIds": ["file_id1", "file_id2"],
        "createSession": true,
        "sessionId": "optional_session_id",
        "personalityId": "optional_personality_id"
    }
    """
    try:
        # Extract request parameters
        question = request_data.get("question", "")
        survey_ids = request_data.get("surveyIds", [])
        selected_file_ids = request_data.get("selectedFileIds", [])
        create_session = request_data.get("createSession", False)
        session_id = request_data.get("sessionId")
        personality_id = request_data.get("personalityId")

        # If session_id is provided, retrieve selected_file_ids from the database
        # This ensures we always use the files that were configured for the session
        if session_id:
            session_data = await db.get_chat_session(session_id, current_user.id)
            if session_data:
                # Use selected_file_ids from the session instead of request
                selected_file_ids = session_data.get("selected_file_ids", [])
                # Also use survey_ids from session if not provided in request
                if not survey_ids:
                    survey_ids = session_data.get("survey_ids", [])
                logger.info(f"Using session {session_id} file access: {len(selected_file_ids)} files, {len(survey_ids)} surveys")
            else:
                logger.warning(f"Session {session_id} not found for user {current_user.id}")

        logger.info(f"Semantic chat request: user={current_user.id}, question='{question[:50]}...', surveys={len(survey_ids)}, files={len(selected_file_ids)}")

        # Validate required parameters
        if not question:
            return JSONResponse(
                content={"error": "Question is required"},
                status_code=400
            )

        # Get module configuration for ai_chat_integration
        from app.services.module_config_cache import module_config_cache
        
        module_name = "ai_chat_integration"
        module_config = await module_config_cache.get_config(db, module_name)
        
        if not module_config:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI chat integration module not configured"
            )
        # Get AI personality and prompt
        effective_personality_id = personality_id or module_config.get("ai_personality_id")
        
        chat_prompt = None
        if effective_personality_id:
            personality_data = await db.execute_fetchrow(
                "SELECT name, detailed_analysis_prompt FROM ai_personalities WHERE id = $1 AND is_active = true",
                [effective_personality_id]
            )
            if personality_data and personality_data["detailed_analysis_prompt"]:
                chat_prompt = personality_data["detailed_analysis_prompt"]

        # Helper to compute a short session title from the user's initial question
        def _compute_session_title(text: str, language: str | None = None) -> str:
            """Compute a short session title from text with simple i18n support.

            Supports language codes: en, es, pt, sv (and common full names).
            Produces a slightly longer title (up to 6 meaningful words) and
            uses translated fallback titles when necessary.
            """
            import re

            # Normalize language to two-letter code
            lang_norm = (language or '').strip().lower()
            if not lang_norm:
                lang = 'en'
            else:
                # Accept values like 'en', 'en-US', 'english'
                if '-' in lang_norm:
                    lang_norm = lang_norm.split('-')[0]
                lang_map = {
                    'english': 'en', 'en': 'en',
                    'spanish': 'es', 'es': 'es',
                    'portuguese': 'pt', 'pt': 'pt',
                    'swedish': 'sv', 'sv': 'sv'
                }
                lang = lang_map.get(lang_norm, 'en')

            translations = {
                'en': 'New Chat',
                'es': 'Nueva conversación',
                'pt': 'Nova conversa',
                'sv': 'Nytt samtal'
            }

            stopwords_map = {
                'en': {'the','a','an','and','or','of','in','on','for','to','with','about','is','are','it','this','that','these','those','my','our','your','their','be','i'},
                'es': {'el','la','los','las','un','una','unos','unas','y','o','de','en','por','para','con','sobre','que','es','son','mi','nuestro','su','sus','se'},
                'pt': {'o','a','os','as','um','uma','e','ou','de','em','por','para','com','sobre','que','é','são','meu','nosso','seu','seus','se'},
                'sv': {'och','eller','av','i','på','för','med','att','är','det','den','de','min','vår','din','deras','en','ett'}
            }

            if not text:
                return translations.get(lang, 'New Chat')

            # Take the first non-empty line
            first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())

            # Replace common punctuation with spaces
            cleaned = re.sub(r"['\"`()\[\],.!?;:/\\-]+", ' ', first_line)
            raw_words = [w for w in re.split(r'\s+', cleaned) if w]

            stopwords = stopwords_map.get(lang, stopwords_map['en'])
            meaningful = [w for w in raw_words if w.lower() not in stopwords]

            # Increase number of words in the title for better context
            num_words = 6
            title_words = meaningful[:num_words] if len(meaningful) >= 1 else raw_words[:num_words]

            if not title_words:
                fallback = first_line[:60].strip()
                return fallback or translations.get(lang, 'New Chat')

            # Capitalize and join
            return ' '.join(w.capitalize() for w in title_words)

        # Create session if requested
        if create_session and not session_id:
            logger.info(f"🔵 Creating new session for user {current_user.id}")
            # If no specific files were provided, auto-collect processed files the user has access to
            if (not selected_file_ids or len(selected_file_ids) == 0) and survey_ids:
                try:
                    auto_files = []
                    for sid in survey_ids:
                        # For each survey, get processed files accessible to the user
                        if current_user.role in ["admin", "super_admin"]:
                            files = await db.execute_query(
                                "SELECT id FROM survey_files WHERE survey_id = $1",
                                [sid]
                            )
                        else:
                            files = await db.execute_query(
                                "SELECT sf.id FROM survey_files sf INNER JOIN surveys s ON sf.survey_id = s.id INNER JOIN user_survey_access usa ON s.id = usa.survey_id WHERE sf.survey_id = $1 AND usa.user_id = $2 AND usa.is_active = true",
                                [sid, current_user.id]
                            )
                        # Only include files that have been processed (pickle exists)
                        for f in files:
                            file_id = str(f.get('id'))
                            pickle_path = os.path.join('survey_data', sid, file_id, 'survey_data.pkl')
                            if os.path.exists(pickle_path):
                                auto_files.append(file_id)
                    # Deduplicate
                    selected_file_ids = list(dict.fromkeys(auto_files))
                    logger.info(f"Auto-collected {len(selected_file_ids)} processed files for session creation for user {current_user.id}")
                except Exception as e_auto:
                    logger.warning(f"Failed to auto-collect files for session creation: {e_auto}")

            session_id = await db.create_chat_session(
                user_id=current_user.id,
                title=_compute_session_title(question, current_user.language),
                survey_ids=survey_ids,
                category="survey-analysis",
                personality_id=effective_personality_id,
                selected_file_ids=selected_file_ids
            )
            logger.info(f"✅ Created new session: {session_id}")
        elif session_id:
            logger.info(f"🔵 Using existing session: {session_id}")
        else:
            logger.info(f"⚠️ No session will be created or used")

        # Perform search on survey data
        # Perform search on survey data with proper file access control
        search_context = ""
        search_metadata = {}
        
        if selected_file_ids or survey_ids:
            try:
                from app.services.fast_search_service import fast_search_service
                
                # Determine search scope based on access permissions
                if selected_file_ids:
                    # User has specific file access - search will be limited to these files
                    logger.info(f"File-specific search requested for files: {selected_file_ids}")
                    search_metadata_note = f"Search limited to {len(selected_file_ids)} specific files"
                    search_type = "file_specific"
                else:
                    # Search entire survey(s)
                    search_metadata_note = f"Search across {len(survey_ids)} survey(s)"
                    search_type = "survey_wide"
                
                if survey_ids:
                    search_results = await fast_search_service.search_surveys(
                        question=question,
                        survey_ids=survey_ids,
                        user_id=str(current_user.id),
                        limit=10,
                        file_ids=selected_file_ids if selected_file_ids else None
                    )
                    
                    if search_results and search_results.get("results"):
                        top_results = search_results["results"][:5]
                        search_context = "\n".join([
                            f"- {result.get('text', '')[:200]}..." 
                            for result in top_results if result.get('text')
                        ])
                        search_metadata = {
                            "search_results_count": len(search_results.get("results", [])),
                            "processing_time_ms": search_results.get("processing_time", 0),
                            "search_type": search_type,
                            "selected_file_ids": selected_file_ids,
                            "note": search_metadata_note
                        }
                        
                        # Log file access context for debugging
                        if selected_file_ids:
                            logger.info(f"Search performed with file restriction: {len(selected_file_ids)} files in {len(survey_ids)} surveys")
                        else:
                            logger.info(f"Search performed across full survey access: {len(survey_ids)} surveys")
                
            except Exception as search_error:
                logger.error(f"Search failed for question '{question[:50]}...': {str(search_error)}")
                search_context = ""
        # Build AI messages
        from app.services.simple_ai_service import simple_ai
        
        system_prompt = chat_prompt if chat_prompt else simple_ai.default_system_prompt
        
        # Add language instruction to the system prompt
        user_language = current_user.language or 'English'
        language_instruction = f"\n\nIMPORTANT NOTES: You need to always respond in user language: {user_language}"
        system_prompt += language_instruction
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if search_context:
            messages.append({
                "role": "system",
                "content": f"Survey data context:\n{search_context}"
            })
        
        # Add conversation history if session exists
        if session_id:
            try:
                conversation_history = await db.get_recent_messages(session_id, limit=6)
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            except Exception as e:
                logger.warning(f"Failed to load conversation history: {e}")
                pass  # Fail silently

        # Add current question
        messages.append({"role": "user", "content": question})

        # Generate AI response
        ai_response = await simple_ai.generate_response(
            messages=messages,
            provider=module_config["provider"],
            model=module_config["model"],
            temperature=module_config.get("temperature", 0.7),
            max_tokens=module_config.get("max_tokens", 1500)
        )

        if not ai_response or not ai_response.strip():
            ai_response = "I apologize, but I'm having trouble processing your request at the moment. Please try again."

        # Try to parse as JSON for structured responses, handling double-encoded JSON
        conversational_response = ai_response
        data_snapshot = None
        confidence = None

        def extract_fields_from_obj(obj):
            conv = obj.get("conversationalResponse", conversational_response)
            snap = obj.get("dataSnapshot", None)
            conf = obj.get("confidence", None)
            return conv, snap, conf

        logger.info(f"Raw AI response: {ai_response}")

        import re
        try:
            # Replace unescaped control characters (like newlines) with spaces for JSON parsing
            safe_ai_response = re.sub(r'[\x00-\x1F\x7F]', ' ', ai_response.strip())
            parsed_response = json.loads(safe_ai_response)
            logger.info(f"Parsed AI response (first parse): {parsed_response}")
            # If the result is a string, try to parse again
            if isinstance(parsed_response, str):
                try:
                    safe_parsed_response = re.sub(r'[\x00-\x1F\x7F]', ' ', parsed_response)
                    parsed_response_2 = json.loads(safe_parsed_response)
                    logger.info(f"Parsed AI response (second parse): {parsed_response_2}")
                    if isinstance(parsed_response_2, dict):
                        conversational_response, data_snapshot, confidence = extract_fields_from_obj(parsed_response_2)
                except Exception as parse2_err:
                    logger.warning(f"Failed second parse of AI response: {parse2_err}")
            elif isinstance(parsed_response, dict):
                conversational_response, data_snapshot, confidence = extract_fields_from_obj(parsed_response)
        except (json.JSONDecodeError, KeyError) as parse_err:
            logger.warning(f"Failed to parse AI response as JSON: {parse_err}")
            # Use raw response if not structured
            pass

        logger.info(f"Extracted conversationalResponse: {conversational_response}")
        logger.info(f"Extracted dataSnapshot: {data_snapshot}")
        logger.info(f"Extracted confidence: {confidence}")
        # Save messages to session (if session exists)
        if session_id:
            logger.info(f"🔵 Session exists: {session_id}, attempting to save messages")
            try:
                # Save both user and AI messages using save_message_pair for better performance
                user_msg_id, ai_msg_id = await db.save_message_pair(
                    session_id=session_id,
                    user_content=question,
                    ai_content=conversational_response,  # Only the conversational part
                    ai_metadata={
                        "data_snapshot": data_snapshot,      # This goes to data_snapshot column
                        "confidence": confidence,            # This goes to confidence column  
                        "personality_used": effective_personality_id,  # This goes to personality_used column
                        "provider": module_config["provider"],
                        "model": module_config["model"],
                        "search_metadata": search_metadata,
                        "full_response": ai_response  # Keep full response in metadata for debugging
                    }
                )
                logger.info(f"✅ Successfully saved messages to session {session_id}")
            except Exception as e:
                logger.error(f"❌ Failed to save messages to session {session_id}: {e}")
        else:
            logger.warning(f"⚠️ No session_id provided - messages will not be saved")

        return JSONResponse(
            content={
                "response": conversational_response,
                "dataSnapshot": data_snapshot,
                "confidence": confidence,
                "sessionId": session_id,
                "success": True,
                "metadata": {
                    "provider": module_config["provider"],
                    "model": module_config["model"],
                    "search_metadata": search_metadata
                }
            },
            status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic chat: {str(e)}")
        return JSONResponse(
            content={
                "error": "Semantic chat processing failed", 
                "success": False,
                "details": str(e)
            },
            status_code=500
        )


@router.post(
    "/semantic-chat-stream",
    summary="Survey semantic chat - AI analysis with streaming response"
)
async def survey_semantic_chat_stream(
    request_data: dict,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Perform semantic chat analysis on survey data with streaming response for improved performance.
    Same request format as semantic-chat but returns Server-Sent Events (SSE) stream.
    """
    
    async def generate_stream():
        try:
            # Extract request parameters (same as non-streaming version)
            question = request_data.get("question", "")
            survey_ids = request_data.get("surveyIds", [])
            selected_file_ids = request_data.get("selectedFileIds", [])
            create_session = request_data.get("createSession", False)
            session_id = request_data.get("sessionId")
            personality_id = request_data.get("personalityId")

            logger.info(f"Streaming semantic chat request: user={current_user.id}, question='{question[:50]}...', surveys={len(survey_ids)}")

            # Validate required parameters
            if not question:
                yield f"data: {{'error': 'Question is required'}}\n\n"
                return

            # Get module configuration (reuse same logic)
            from app.services.module_config_cache import module_config_cache
            
            module_name = "ai_chat_integration"
            module_config = await module_config_cache.get_config(db, module_name)
            
            if not module_config:
                yield f"data: {{'error': 'Module configuration not found for {module_name}'}}\n\n"
                return

            # Send initial status
            yield f"data: {{'type': 'status', 'message': 'Initializing AI chat...'}}\n\n"

            # Get AI personality and session setup (same logic as non-streaming)
            effective_personality_id = personality_id or module_config.get("ai_personality_id")
            
            chat_prompt = None
            if effective_personality_id:
                if (module_config.get("ai_personality_id") == effective_personality_id and 
                    module_config.get("detailed_analysis_prompt")):
                    chat_prompt = module_config["detailed_analysis_prompt"]
                else:
                    from app.services.query_optimizer import query_optimizer
                    chat_prompt = await query_optimizer.get_personality_prompt(db, effective_personality_id)

            # Create session if needed
            if create_session and not session_id:
                # Compute a short, meaningful title from the user's question (language-aware)
                def _compute_session_title(text: str, language: str | None = None) -> str:
                    import re

                    # Normalize language
                    lang_norm = (language or '').strip().lower()
                    if not lang_norm:
                        lang = 'en'
                    else:
                        if '-' in lang_norm:
                            lang_norm = lang_norm.split('-')[0]
                        lang_map = {
                            'english': 'en', 'en': 'en',
                            'spanish': 'es', 'es': 'es',
                            'portuguese': 'pt', 'pt': 'pt',
                            'swedish': 'sv', 'sv': 'sv'
                        }
                        lang = lang_map.get(lang_norm, 'en')

                    translations = {
                        'en': 'New Chat',
                        'es': 'Nueva conversación',
                        'pt': 'Nova conversa',
                        'sv': 'Nytt samtal'
                    }

                    stopwords_map = {
                        'en': {'the','a','an','and','or','of','in','on','for','to','with','about','is','are','it','this','that','these','those','my','our','your','their','be','i'},
                        'es': {'el','la','los','las','un','una','unos','unas','y','o','de','en','por','para','con','sobre','que','es','son','mi','nuestro','su','sus','se'},
                        'pt': {'o','a','os','as','um','uma','e','ou','de','em','por','para','com','sobre','que','é','são','meu','nosso','seu','seus','se'},
                        'sv': {'och','eller','av','i','på','för','med','att','är','det','den','de','min','vår','din','deras','en','ett'}
                    }

                    if not text:
                        return translations.get(lang, 'New Chat')

                    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text.strip())
                    cleaned = re.sub(r"['\"`()\[\],.!?;:/\\-]+", ' ', first_line)
                    raw_words = [w for w in re.split(r'\s+', cleaned) if w]

                    stopwords = stopwords_map.get(lang, stopwords_map['en'])
                    meaningful = [w for w in raw_words if w.lower() not in stopwords]

                    num_words = 6
                    title_words = meaningful[:num_words] if len(meaningful) >= 1 else raw_words[:num_words]

                    if not title_words:
                        fallback = first_line[:60].strip()
                        return fallback or translations.get(lang, 'New Chat')

                    return ' '.join(w.capitalize() for w in title_words)

                session_id = await db.create_chat_session(
                    user_id=current_user.id,
                    title=_compute_session_title(question, current_user.language),
                    survey_ids=survey_ids,
                    category="survey-analysis",
                    personality_id=effective_personality_id,
                    selected_file_ids=selected_file_ids
                )
                yield f"data: {{'type': 'session', 'sessionId': '{session_id}'}}\n\n"

            # Perform search (same logic as non-streaming)
            yield f"data: {{'type': 'status', 'message': 'Searching survey data...'}}\n\n"
            
            search_context = ""
            search_metadata = {}
            
            if selected_file_ids or survey_ids:
                try:
                    from app.services.fast_search_service import FastSearchService
                    search_service = FastSearchService()
                    
                    search_results = await search_service.search_surveys(
                        question=question,
                        survey_ids=survey_ids,
                        user_id=str(current_user.id),
                        limit=10
                    )
                    
                    if search_results and search_results.get("results"):
                        top_results = search_results["results"][:5]
                        search_context = "\n".join([
                            f"- {result.get('text', '')[:300]}..." 
                            for result in top_results if result.get('text')
                        ])
                        search_metadata = {
                            "search_results_count": len(search_results.get("results", [])),
                            "processing_time_ms": search_results.get("processing_time", 0),
                            "search_type": "file_specific" if selected_file_ids else "full_survey"
                        }
                        yield f"data: {{'type': 'search', 'resultsFound': {len(top_results)}}}\n\n"
                    else:
                        yield f"data: {{'type': 'search', 'resultsFound': 0}}\n\n"
                        
                except Exception as search_error:
                    logger.error(f"Search failed for streaming semantic chat: {str(search_error)}")
                    yield f"data: {{'type': 'warning', 'message': 'Search failed, proceeding without context'}}\n\n"

            # Build AI messages (same logic as non-streaming)
            from app.services.simple_ai_service import simple_ai
            
            system_prompt = chat_prompt if chat_prompt else simple_ai.default_system_prompt
            
            # Add language instruction to the system prompt
            user_language = current_user.language or 'English'
            language_instruction = f"\n\nIMPORTANT NOTES: You need to always respond in user language: {user_language}"
            system_prompt += language_instruction
            
            messages = [{"role": "system", "content": system_prompt}]
            
            if search_context:
                messages.append({
                    "role": "system",
                    "content": f"Relevant survey data context:\n{search_context}"
                })
            
            # Add conversation history if session exists
            if session_id:
                try:
                    from app.services.query_optimizer import query_optimizer
                    previous_messages = await query_optimizer.get_recent_chat_messages(db, session_id, 20)
                    
                    if previous_messages:
                        for msg in reversed(previous_messages):
                            role = "user" if msg["sender_type"] == "user" else "assistant"
                            messages.append({"role": role, "content": msg["content"]})
                except Exception as history_error:
                    logger.warning(f"Failed to load conversation history: {history_error}")
            
            messages.append({"role": "user", "content": question})

            # Generate streaming AI response
            yield f"data: {{'type': 'status', 'message': 'Generating AI response...'}}\n\n"
            
            full_response = ""
            try:
                # Use the new streaming functionality
                async for chunk in simple_ai.generate_streaming_response(
                    messages=messages,
                    provider=module_config["provider"],
                    model=module_config["model"],
                    temperature=module_config.get("temperature", 0.7),
                    max_tokens=module_config.get("max_tokens", 2000),
                    api_key=module_config.get("api_key")
                ):
                    if chunk:
                        full_response += chunk
                        # Escape quotes and newlines for JSON
                        escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        yield f"data: {{'type': 'content', 'chunk': \"{escaped_chunk}\"}}\n\n"
                
                # Save messages to session if session exists
                if session_id and full_response:
                    try:
                        await db.save_message_pair(
                            session_id=session_id,
                            user_content=question,
                            ai_content=full_response,
                            ai_metadata={
                                "provider": module_config["provider"],
                                "model": module_config["model"],
                                "temperature": module_config["temperature"],
                                "personality_id": effective_personality_id,
                                "search_results_count": len(search_context.split('\n')) if search_context else 0,
                                "streaming": True
                            }
                        )
                    except Exception as save_error:
                        logger.error(f"Failed to save streaming messages to session: {str(save_error)}")

                # Send completion with metadata
                yield f"data: {{'type': 'complete', 'sessionId': '{session_id}', 'metadata': {{'searchResults': {search_metadata.get('search_results_count', 0)}, 'provider': '{module_config['provider']}', 'model': '{module_config['model']}'}}}}\n\n"
                
            except Exception as ai_error:
                logger.error(f"AI streaming service failed: {str(ai_error)}")
                yield f"data: {{'type': 'error', 'message': 'AI service failed: {str(ai_error)}'}}\n\n"
                
        except Exception as e:
            logger.error(f"Error in streaming semantic chat: {str(e)}")
            yield f"data: {{'type': 'error', 'message': 'Streaming chat failed: {str(e)}'}}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )


@router.post("/surveys/{survey_id}/access", response_model=UserSurveyAccessSchema)
async def grant_survey_access(
    survey_id: UUID,
    access_data: UserSurveyAccessCreate,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Grant or update user access to a survey."""
    # Check if current user has admin access to the survey
    has_permission = await access_control_service.check_survey_permission(
        db, current_user.id, survey_id, AccessType.ADMIN
    )
    
    if not has_permission and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to grant survey access"
        )
    
    try:
        access = await access_control_service.grant_survey_access(
            db, access_data.user_id, survey_id, access_data.access_type, current_user.id
        )
        return access
    except Exception as e:
        logger.error(f"Error granting survey access to survey {survey_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to grant survey access: {str(e)}"
        )

@router.delete("/surveys/{survey_id}/access/{user_id}")
async def revoke_survey_access(
    survey_id: UUID,
    user_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Revoke user access to a survey."""
    # Check if current user has admin access to the survey
    has_permission = await access_control_service.check_survey_permission(
        db, current_user.id, survey_id, AccessType.ADMIN
    )
    
    if not has_permission and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to revoke survey access"
        )
    
    try:
        success = await access_control_service.revoke_survey_access(db, user_id, survey_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access record not found"
            )
        
        return {"message": "Survey access revoked successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking survey access for user {user_id} from survey {survey_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke survey access"
        )

@router.get("/surveys/{survey_id}/access/{user_id}", response_model=UserSurveyAccessSchema)
async def get_user_survey_access(
    survey_id: UUID,
    user_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Get user's access level to a specific survey."""
    # Users can check their own access, admins can check anyone's
    if current_user.id != user_id and current_user.role != "admin":
        # Check if current user has read access to the survey
        has_permission = await access_control_service.check_survey_permission(
            db, current_user.id, survey_id, AccessType.READ
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to view survey access"
            )
    
    access = await access_control_service.get_user_survey_access(db, user_id, survey_id)
    
    if not access:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access record not found"
        )
    
    return access

@router.get("/surveys/{survey_id}/users", response_model=List[UserSurveyAccessSchema])
async def get_survey_users(
    survey_id: UUID,
    access_type: Optional[AccessType] = None,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Get all users with access to a survey."""
    # Check if current user has read access to the survey
    has_permission = await access_control_service.check_survey_permission(
        db, current_user.id, survey_id, AccessType.READ
    )
    
    if not has_permission and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view survey users"
        )
    
    users = await access_control_service.get_survey_users(db, survey_id, access_type)
    return users

@router.post("/surveys/{survey_id}/bulk-access", response_model=List[UserSurveyAccessSchema])
async def bulk_grant_survey_access(
    survey_id: UUID,
    bulk_access: BulkAccessGrant,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Grant access to multiple users for a survey."""
    # Check if current user has admin access to the survey
    has_permission = await access_control_service.check_survey_permission(
        db, current_user.id, survey_id, AccessType.ADMIN
    )
    
    if not has_permission and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to grant survey access"
        )
    
    try:
        user_access_list = [
            {"user_id": item.user_id, "access_type": item.access_type}
            for item in bulk_access.access_grants
        ]
        
        results = await access_control_service.bulk_grant_survey_access(
            db, survey_id, user_access_list
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to bulk grant survey access: {str(e)}"
        )

@router.get("/check-survey-permission/{survey_id}")
async def check_survey_permission(
    survey_id: UUID,
    required_access: AccessType,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Check if current user has required permission for a survey."""
    has_permission = await access_control_service.check_survey_permission(
        db, current_user.id, survey_id, required_access
    )
    
    return {
        "user_id": current_user.id,
        "survey_id": survey_id,
        "required_access": required_access,
        "has_permission": has_permission
    }




@router.post("/files/{file_id}/access", response_model=UserSurveyFileAccessSchema)
async def grant_file_access(
    file_id: UUID,
    access_data: UserSurveyFileAccessCreate,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Grant or update user access to a file."""
    # Check if current user has admin access to the file
    has_permission = await access_control_service.check_file_permission(
        db, current_user.id, file_id, AccessType.ADMIN
    )
    
    if not has_permission and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to grant file access"
        )
    
    try:
        access = await access_control_service.grant_file_access(
            db, access_data.user_id, file_id, access_data.access_type
        )
        return access
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to grant file access: {str(e)}"
        )


@router.delete("/files/{file_id}/access/{user_id}")
async def revoke_file_access(
    file_id: UUID,
    user_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Revoke user access to a file."""
    # Check if current user has admin access to the file
    has_permission = await access_control_service.check_file_permission(
        db, current_user.id, file_id, AccessType.ADMIN
    )
    
    if not has_permission and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to revoke file access"
        )
    
    success = await access_control_service.revoke_file_access(db, user_id, file_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access record not found"
        )
    
    return {"message": "File access revoked successfully"}


@router.get("/files/{file_id}/access/{user_id}", response_model=UserSurveyFileAccessSchema)
async def get_user_file_access(
    file_id: UUID,
    user_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Get user's access level to a specific file."""
    # Users can check their own access, admins can check anyone's
    if current_user.id != user_id and current_user.role != "admin":
        # Check if current user has read access to the file
        has_permission = await access_control_service.check_file_permission(
            db, current_user.id, file_id, AccessType.READ
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to view file access"
            )
    
    access = await access_control_service.get_user_file_access(db, user_id, file_id)
    
    if not access:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Access record not found"
        )
    
    return access



@router.get("/files/{file_id}/users", response_model=List[UserSurveyFileAccessSchema])
async def get_file_users(
    file_id: UUID,
    access_type: Optional[AccessType] = None,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Get all users with access to a file."""
    # Check if current user has read access to the file
    has_permission = await access_control_service.check_file_permission(
        db, current_user.id, file_id, AccessType.READ
    )
    
    if not has_permission and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view file users"
        )
    
    users = await access_control_service.get_file_users(db, file_id, access_type)
    return users


@router.post("/files/{file_id}/bulk-access", response_model=List[UserSurveyFileAccessSchema])
async def bulk_grant_file_access(
    file_id: UUID,
    bulk_access: BulkAccessGrant,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Grant access to multiple users for a file."""
    # Check if current user has admin access to the file
    has_permission = await access_control_service.check_file_permission(
        db, current_user.id, file_id, AccessType.ADMIN
    )
    
    if not has_permission and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to grant file access"
        )
    
    try:
        user_access_list = [
            {"user_id": item.user_id, "access_type": item.access_type}
            for item in bulk_access.access_grants
        ]
        
        results = await access_control_service.bulk_grant_file_access(
            db, file_id, user_access_list
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to bulk grant file access: {str(e)}"
        )


@router.get("/check-file-permission/{file_id}")
async def check_file_permission(
    file_id: UUID,
    required_access: AccessType,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Check if current user has required permission for a file."""
    has_permission = await access_control_service.check_file_permission(
        db, current_user.id, file_id, required_access
    )
    
    return {
        "user_id": current_user.id,
        "file_id": file_id,
        "required_access": required_access,
        "has_permission": has_permission
    }
