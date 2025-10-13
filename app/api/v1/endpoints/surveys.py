from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import os
from datetime import datetime

from app.core.dependencies import get_current_user, get_db
from app.models.models import User
from app.models.schemas import (
    Survey, SurveyCreate, SurveyUpdate,
    SurveyFile, SurveyWithFilesResponse, SurveyStatsSimple,
    SurveyStatistics,
    GenerateSuggestionsRequest, SurveyIdSuggestionsRequest, 
    SimpleSuggestionsRequest, SuggestionsResponse,
    SemanticChatRequest, SemanticChatResponse
)
from app.services.survey_service import survey_service
from app.services import ai_service
from app.services.fast_search_service import FastSearchService
from app.services.simple_ai_service import SimpleAIService
from app.services.streamlined_responses import StreamlinedResponses
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Initialize services
fast_search_service = FastSearchService()
simple_ai_service = SimpleAIService()
streamlined_responses = StreamlinedResponses()

# Vector search service removed - using fast_search_service with direct file access
# def get_vector_search_service():
#     from app.services.vector_search_service import vector_search_service
#     return vector_search_service


@router.post(
    "/",
    response_model=Survey,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new survey"
)
async def create_survey(
    survey_data: SurveyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new survey.
    
    - **title**: Survey title (required)
    - **description**: Survey description (optional)
    """
    try:
        survey = await survey_service.create_survey(
            db=db,
            survey_data=survey_data,
            user_id=current_user.id
        )
        return Survey(
            id=str(survey.id),
            title=survey.title,
            category=survey.category,
            description=survey.description,
            ai_suggestions=survey.ai_suggestions,
            number_participants=survey.number_participants,
            total_files=survey.total_files,
            created_at=survey.created_at.isoformat() if survey.created_at else "",
            updated_at=survey.updated_at.isoformat() if survey.updated_at else "",
            fileid=None,
            filename=None,
            createdat=survey.created_at.isoformat() if survey.created_at else "",
            storage_path=None,
            primary_language=getattr(survey, 'primary_language', None),
            files=[]
        )
    except Exception as e:
        logger.error(f"Error creating survey: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create survey"
        )


@router.get(
    "/my-surveys",
    response_model=List[Survey],
    summary="Get surveys user has access to"
)
async def get_my_surveys(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get surveys that the current user has access to.
    This includes surveys they own and surveys they have been granted access to.
    """
    try:
        surveys = await survey_service.get_user_accessible_surveys(
            db=db,
            user_id=current_user.id,
            skip=skip,
            limit=limit
        )
        return [Survey(
            id=str(survey.id),
            title=survey.title,
            category=survey.category,
            description=survey.description,
            ai_suggestions=survey.ai_suggestions,
            number_participants=survey.number_participants,
            total_files=survey.total_files,
            created_at=survey.created_at.isoformat() if survey.created_at else "",
            updated_at=survey.updated_at.isoformat() if survey.updated_at else "",
            # Backward compatibility fields
            fileid=None,
            filename=None,
            createdat=survey.created_at.isoformat() if survey.created_at else "",
            storage_path=None,
            primary_language=getattr(survey, 'primary_language', None),
            files=[]
        ) for survey in surveys]
    except Exception as e:
        logger.error(f"Error getting user accessible surveys: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve accessible surveys"
        )


@router.get(
    "/",
    response_model=List[Survey],
    summary="Get user's surveys"
)
async def get_surveys(
    skip: int = 0,
    limit: int = 100,
    include_files: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all surveys for the current user.
    
    - **skip**: Number of surveys to skip (for pagination)
    - **limit**: Maximum number of surveys to return
    - **include_files**: Whether to include file information
    """
    try:
        surveys = await survey_service.get_user_surveys(
            db=db,
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            include_files=include_files
        )
        return [Survey(
            id=str(survey.id),
            title=survey.title,
            category=survey.category,
            description=survey.description,
            ai_suggestions=survey.ai_suggestions,
            number_participants=survey.number_participants,
            total_files=survey.total_files,
            created_at=survey.created_at.isoformat() if survey.created_at else "",
            updated_at=survey.updated_at.isoformat() if survey.updated_at else "",
            # Backward compatibility fields
            fileid=None,
            filename=None,
            createdat=survey.created_at.isoformat() if survey.created_at else "",
            storage_path=None,
            primary_language=getattr(survey, 'primary_language', None),
            files=[SurveyFile(
                id=str(f.id),
                survey_id=str(f.survey_id),
                filename=f.filename,
                storage_path=f.storage_path,
                file_size=f.file_size,
                file_hash=f.file_hash,
                upload_date=f.upload_date.isoformat() if f.upload_date else "",
                created_at=f.created_at.isoformat() if f.created_at else "",
                updated_at=f.updated_at.isoformat() if f.updated_at else ""
            ) for f in (survey.files or [])]
        ) for survey in surveys]
    except Exception as e:
        logger.error(f"Error getting surveys: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve surveys"
        )


@router.get(
    "/metadata",
    summary="Get survey list metadata for real-time updates"
)
async def get_survey_metadata(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get lightweight metadata for surveys accessible to the user.
    Used for real-time updates and quick survey list rendering.
    """
    try:
        metadata = await survey_service.get_survey_metadata(
            db=db,
            user_id=current_user.id
        )
        return JSONResponse(content=metadata)
    except Exception as e:
        logger.error(f"Error getting survey metadata: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve survey metadata"
        )


@router.get(
    "/{survey_id}",
    response_model=Survey,
    summary="Get a specific survey"
)
async def get_survey(
    survey_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific survey by ID.
    """
    survey = await survey_service.get_survey_with_files(
        db=db,
        survey_id=survey_id,
        user_id=current_user.id
    )
    if not survey:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Survey not found"
        )
    return Survey(
            id=str(survey.id),
            title=survey.title,
            category=survey.category,
            description=survey.description,
            ai_suggestions=survey.ai_suggestions,
            number_participants=survey.number_participants,
            total_files=survey.total_files,
            created_at=survey.created_at.isoformat() if survey.created_at else "",
            updated_at=survey.updated_at.isoformat() if survey.updated_at else "",
            fileid=None,
            filename=None,
            createdat=survey.created_at.isoformat() if survey.created_at else "",
            storage_path=None,
            primary_language=getattr(survey, 'primary_language', None),
            files=[]
        )


@router.put(
    "/{survey_id}",
    response_model=Survey,
    summary="Update a survey"
)
async def update_survey(
    survey_id: uuid.UUID,
    survey_data: SurveyUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a survey.
    """
    try:
        survey = await survey_service.update(
            db=db,
            id=survey_id,
            obj_in=survey_data,
            current_user_id=current_user.id
        )
        if not survey:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        return Survey(
            id=str(survey.id),
            title=survey.title,
            category=survey.category,
            description=survey.description,
            ai_suggestions=survey.ai_suggestions,
            number_participants=survey.number_participants,
            total_files=survey.total_files,
            created_at=survey.created_at.isoformat() if survey.created_at else "",
            updated_at=survey.updated_at.isoformat() if survey.updated_at else "",
            fileid=None,
            filename=None,
            createdat=survey.created_at.isoformat() if survey.created_at else "",
            storage_path=None,
            primary_language=getattr(survey, 'primary_language', None),
            files=[]
        )
    except Exception as e:
        logger.error(f"Error updating survey {survey_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update survey"
        )


@router.delete(
    "/{survey_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a survey"
)
async def delete_survey(
    survey_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a survey and all its files.
    """
    success = await survey_service.delete_survey(
        db=db,
        survey_id=survey_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Survey not found"
        )


@router.post(
    "/{survey_id}/files",
    response_model=SurveyFile,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file to a survey"
)
async def upload_file(
    survey_id: uuid.UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a file to a survey.
    
    Supported formats: CSV, Excel (.xls, .xlsx)
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Read file content
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File appears to be empty"
            )
        logger.info(f"Successfully read {len(content)} bytes from uploaded file: {file.filename}")
    except Exception as e:
        logger.error(f"Error reading uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read file content"
        )
    
    try:
        survey_file = await survey_service.upload_file(
            db=db,
            survey_id=survey_id,
            file_content=content,
            filename=file.filename,
            user_id=current_user.id
        )
        return SurveyFile(
                id=str(survey_file.id),
                survey_id=str(survey_file.survey_id),
                filename=survey_file.filename,
                storage_path=survey_file.storage_path,
                file_size=survey_file.file_size,
                file_hash=survey_file.file_hash,
                upload_date=survey_file.upload_date.isoformat() if survey_file.upload_date else "",
                created_at=survey_file.created_at.isoformat() if survey_file.created_at else "",
                updated_at=survey_file.updated_at.isoformat() if survey_file.updated_at else ""
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file"
        )


@router.get(
    "/{survey_id}/files",
    response_model=List[SurveyFile],
    summary="Get files for a survey"
)
async def get_survey_files(
    survey_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all files for a specific survey.
    """
    survey = await survey_service.get_survey_with_files(
        db=db,
        survey_id=survey_id,
        user_id=current_user.id
    )
    if not survey:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Survey not found"
        )
    
    return [SurveyFile(
                id=str(file.id),
                survey_id=str(file.survey_id),
                filename=file.filename,
                storage_path=file.storage_path,
                file_size=file.file_size,
                file_hash=file.file_hash,
                upload_date=file.upload_date.isoformat() if file.upload_date else "",
                created_at=file.created_at.isoformat() if file.created_at else "",
                updated_at=file.updated_at.isoformat() if file.updated_at else ""
            ) for file in survey.files]


@router.get(
    "/files/{file_id}/content",
    summary="Get processed file content"
)
async def get_file_content(
    file_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the processed content of a survey file.
    """
    content = await survey_service.get_file_content(
        db=db,
        file_id=file_id,
        user_id=current_user.id
    )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or not processed"
        )
    
    return JSONResponse(content=content)


@router.get(
    "/{survey_id}/statistics",
    response_model=SurveyStatistics,
    summary="Get survey statistics"
)
async def get_survey_statistics(
    survey_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive statistics for a survey.
    """
    stats = await survey_service.get_survey_statistics(
        db=db,
        survey_id=survey_id,
        user_id=current_user.id
    )
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Survey not found"
        )
    
    return SurveyStatistics(**stats)


@router.get(
    "/{survey_id}/list-item",
    summary="Get survey in list format"
)
async def get_survey_list_item(
    survey_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get survey in list format for real-time updates.
    """
    try:
        survey_item = await survey_service.get_survey_list_item(
            db=db,
            survey_id=survey_id,
            user_id=current_user.id
        )
        if not survey_item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        return JSONResponse(content=survey_item)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting survey list item: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve survey list item"
        )


@router.get(
    "/{survey_id}/with-files",
    response_model=SurveyWithFilesResponse,
    summary="Get survey with all files"
)
async def get_survey_with_files(
    survey_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get survey with all file information included.
    """
    survey = await survey_service.get_survey_with_files(
        db=db,
        survey_id=survey_id,
        user_id=current_user.id
    )
    if not survey:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Survey not found"
        )
    
    # Convert files to schema format and calculate stats
    files_list = []
    total_size = 0
    file_types = {}
    if survey.files:
        for file in survey.files:
            # Calculate file extension
            filename = file.filename
            ext = os.path.splitext(filename)[1].lower() if filename else ""
            file_types[ext] = file_types.get(ext, 0) + 1
            
            files_list.append(SurveyFile(
                id=str(file.id),
                survey_id=str(file.survey_id),
                filename=file.filename,
                storage_path=file.storage_path or "",
                file_size=file.file_size,
                file_hash=file.file_hash,
                upload_date=file.upload_date.isoformat() if file.upload_date else "",
                created_at=file.created_at.isoformat() if file.created_at else "",
                updated_at=file.updated_at.isoformat() if file.updated_at else ""
            ))
            total_size += file.file_size or 0
    
    # Create stats object
    stats = SurveyStatsSimple(
        totalFiles=len(files_list),
        totalSize=total_size,
        fileTypes=file_types,
        lastUpdated=survey.updated_at.isoformat() if survey.updated_at else None
    )
    
    return SurveyWithFilesResponse(
        id=str(survey.id),
        title=survey.title,
        category=survey.category,
        description=survey.description,
        ai_suggestions=survey.ai_suggestions,
        number_participants=survey.number_participants,
        total_files=survey.total_files,
        created_at=survey.created_at.isoformat() if survey.created_at else "",
        updated_at=survey.updated_at.isoformat() if survey.updated_at else "",
        fileid=None,
        filename=None,
        createdat=survey.created_at.isoformat() if survey.created_at else "",
        storage_path=None,
        primary_language=getattr(survey, 'primary_language', None),
        files=files_list,
        stats=stats
    )


@router.get(
    "/{survey_id}/access-check",
    summary="Check user access to survey"
)
async def check_survey_access(
    survey_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Check if current user has access to the specified survey and return accessible files.
    """
    try:
        # Check admin access first (more efficient for admin users)
        if current_user.role and current_user.role.role == "admin":
            has_access = True
        else:
            has_access = await survey_service.check_user_access(
                db=db,
                survey_id=survey_id,
                user_id=current_user.id
            )
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this survey"
            )
        
        # Get survey with files
        survey = await survey_service.get_survey_with_files(db, survey_id, current_user.id)
        if not survey:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        
        # Prepare accessible files list with processing status
        accessible_files = []
        for file in survey.files:
            # Fast search service works directly with pickle files, no processing status needed
            # is_processed = True  # Assume all files are ready for fast search
            
            file_info = {
                "id": str(file.id),
                "fileId": str(file.id),  # Keep both for compatibility
                "filename": file.filename,
                "file_size": file.file_size or 0,
                "accessType": "admin" if current_user.role and current_user.role.role == "admin" else "read",
                "isProcessed": True,  # Fast search works directly with files
                "processingStatus": "completed"  # Always completed with fast search
            }
            accessible_files.append(file_info)
        
        processed_count = sum(1 for f in accessible_files if f["isProcessed"])
        
        return {
            "hasAccess": True,
            "accessType": "admin" if current_user.role and current_user.role.role == "admin" else "read",
            "surveyId": str(survey_id),
            "accessibleFiles": accessible_files,
            "message": f"Survey access validated - {len(accessible_files)} files available ({processed_count} processed)"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking survey access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check survey access"
        )


@router.get(
    "/{survey_id}/my-file-access",
    summary="Get user's file access for survey"
)
async def get_my_file_access(
    survey_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get file access information for all files in a survey for the current user.
    """
    try:
        file_access = await survey_service.get_user_file_access(
            db=db,
            survey_id=survey_id,
            user_id=current_user.id
        )
        return JSONResponse(content=file_access)
    except Exception as e:
        logger.error(f"Error getting user file access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file access information"
        )


@router.post(
    "/upload",
    response_model=Survey,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and create survey from file"
)
async def upload_survey(
    file: UploadFile = File(...),
    title: str = None,
    description: str = None,
    category: str = None,
    number_participants: str = None,
    ai_suggestions: str = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a file and create a survey from it.
    
    Supported formats: CSV, Excel (.xls, .xlsx)
    """
    # Admin check would typically be handled by middleware in TypeScript
    # For now, we'll allow all authenticated users
    
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided"
        )
    
    # Parse AI suggestions if provided
    parsed_suggestions = None
    if ai_suggestions:
        try:
            import json
            parsed_suggestions = json.loads(ai_suggestions)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in ai_suggestions: {ai_suggestions}")
    
    # Parse number_participants if provided
    parsed_participants = None
    if number_participants:
        try:
            parsed_participants = int(number_participants)
        except ValueError:
            logger.warning(f"Invalid number_participants: {number_participants}")
    
    try:
        content = await file.read()
        
        print(f"DEBUG: File content size: {len(content)} bytes")
        print(f"DEBUG: File extension: {os.path.splitext(file.filename)[1]}")
        
        survey = await survey_service.create_survey_from_upload(
            db=db,
            file_content=content,
            filename=file.filename,
            title=title or file.filename,
            description=description,
            user_id=current_user.id,
            category=category,
            ai_suggestions=parsed_suggestions,
            number_participants=parsed_participants
        )
        
        return Survey(
            id=str(survey.id),
            title=survey.title,
            category=survey.category,
            description=survey.description,
            ai_suggestions=survey.ai_suggestions,
            number_participants=survey.number_participants,
            total_files=survey.total_files,
            created_at=survey.created_at.isoformat() if survey.created_at else "",
            updated_at=survey.updated_at.isoformat() if survey.updated_at else "",
            fileid=None,
            filename=None,
            createdat=survey.created_at.isoformat() if survey.created_at else "",
            storage_path=None,
            primary_language=getattr(survey, 'primary_language', None),
            files=[]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error uploading survey: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload survey"
        )


@router.post(
    "/generate-suggestions",
    response_model=SuggestionsResponse,
    summary="Generate AI suggestions without saving to survey"
)
async def generate_suggestions(
    request: GenerateSuggestionsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate AI-powered suggestions for survey analysis without saving to survey.
    Admin only endpoint (not used by frontend).
    """
    try:
        suggestions = await ai_service.generate_suggestions(
            db=db,
            description=request.description,
            category=request.category,
            personality_id=request.personalityId,
            file_content=request.fileContent,
            user_id=current_user.id
        )
        
        return SuggestionsResponse(
            suggestions=suggestions,
            category=request.category
        )
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e) if "API" in str(e) or "personality" in str(e) else "Failed to generate suggestions"
        )


@router.post(
    "/{survey_id}/suggestions",
    response_model=SuggestionsResponse,
    summary="Generate and save AI suggestions for a survey"
)
async def generate_survey_id_suggestions(
    survey_id: uuid.UUID,
    request: SurveyIdSuggestionsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate and save AI suggestions for a specific survey.
    Admin only endpoint.
    """
    try:
        # Get survey details
        survey = await survey_service.get_survey_with_files(
            db=db,
            survey_id=survey_id,
            user_id=current_user.id
        )
        if not survey:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        
        # Extract attributes while in async context to avoid greenlet issues
        survey_description = survey.description
        survey_category = survey.category
        
        # Description and category are optional - AI can work with file content alone
        suggestions = await ai_service.generate_and_save_suggestions(
            db=db,
            survey_id=survey_id,
            description=survey_description,
            category=survey_category,
            personality_id=request.personalityId,
            file_content=request.fileContent,
            user_id=current_user.id
        )
        
        return SuggestionsResponse(
            suggestions=suggestions,
            category=survey_category,
            surveyId=str(survey_id)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating survey suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e) if "API" in str(e) or "personality" in str(e) else "Failed to generate suggestions"
        )


@router.post(
    "/{survey_id}/suggestions/simple",
    response_model=SuggestionsResponse,
    summary="Generate enhanced AI suggestions with actual survey data"
)
async def generate_simple_enhanced_suggestions(
    survey_id: uuid.UUID,
    request: SimpleSuggestionsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate enhanced AI suggestions using actual survey data for analysis.
    Admin only endpoint.
    """
    try:
        # Get survey details
        survey = await survey_service.get_survey_with_files(
            db=db,
            survey_id=survey_id,
            user_id=current_user.id
        )
        if not survey:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Survey not found"
            )
        
        # Extract attributes while in async context to avoid greenlet issues
        survey_description = survey.description
        survey_category = survey.category
        
        # Description and category are optional - AI can work with file content alone
        suggestions = await ai_service.generate_simple_enhanced_suggestions(
            db=db,
            survey_id=survey_id,
            description=survey_description,
            category=survey_category,
            personality_id=request.personalityId,
            user_id=current_user.id
        )
        
        return SuggestionsResponse(
            suggestions=suggestions,
            category=survey_category,
            surveyId=str(survey_id),
            method="simple_enhanced_with_real_data"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating simple enhanced suggestions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e) if "API" in str(e) or "personality" in str(e) else "Failed to generate suggestions"
        )


@router.delete(
    "/files/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a survey file"
)
async def delete_survey_file(
    file_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a specific survey file.
    """
    success = await survey_service.delete_survey_file(
        db=db,
        file_id=file_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )


@router.get(
    "/{survey_id}/files/{file_id}/rows",
    summary="Get rows from a survey file"
)
async def get_survey_file_rows(
    survey_id: uuid.UUID,
    file_id: uuid.UUID,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get rows from a specific survey file with pagination.
    """
    try:
        rows = await survey_service.get_file_rows(
            db=db,
            survey_id=survey_id,
            file_id=file_id,
            user_id=current_user.id,
            skip=skip,
            limit=limit
        )
        if rows is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found or access denied"
            )
        
        return JSONResponse(content=rows)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting file rows: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve file rows"
        )


@router.post(
    "/{survey_id}/files/preload",
    summary="Preload survey files for optimization"
)
async def preload_survey_files(
    survey_id: uuid.UUID,
    file_ids: List[uuid.UUID],
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Preload survey files into cache for optimization.
    Processes and caches file data for faster subsequent access.
    """
    try:
        if not file_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file IDs provided"
            )
        
        # Check survey access
        if not await survey_service.check_user_access(db, survey_id, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this survey"
            )
        
        # Preload files
        result = await survey_service.preload_survey_files(
            db=db,
            survey_id=survey_id,
            file_ids=file_ids,
            user_id=current_user.id
        )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preloading survey files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to preload files"
        )


@router.post(
    "/semantic-chat",
    summary="Semantic chat with survey context",
    response_model=SemanticChatResponse
)
async def semantic_chat(
    request: SemanticChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Perform semantic chat with survey context using AI.
    Provides intelligent responses based on survey data and context.
    
    Expected request body:
    {
        "question": "Your question here",
        "surveyIds": ["uuid1", "uuid2"],
        "personalityId": "optional-personality-id",
        "sessionId": "optional-session-uuid",
        "createSession": true
    }
    """
    try:
        # Extract required fields from the request
        message = request.question
        survey_ids = [str(sid) for sid in request.surveyIds] if request.surveyIds else []
        personality_id = request.personalityId
        session_id = request.sessionId
        create_session = request.createSession
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question is required"
            )
        
        logger.info(f"Semantic chat request: {message} for surveys: {survey_ids}")
        
        # Use the working search functionality
        search_results = None
        if survey_ids:
            try:
                from app.services.embedding_service import embedding_service
                query_embedding = await embedding_service.generate_embedding(message)
                
                if query_embedding:
                    search_results = await fast_search_service.fast_search(
                        query_embedding=query_embedding,
                        survey_ids=survey_ids,
                        threshold=0.25,
                        max_results=500,
                        query_text=message
                    )
                    logger.info(f"Search completed: {search_results.get('metadata', {}).get('total_matches', 0)} matches found")
                else:
                    logger.warning("Failed to generate embedding for search")
            except Exception as e:
                logger.error(f"Search failed: {e}")
        
        # Build enhanced context like streamlined chat
        search_text = ""
        if search_results and search_results.get("responses"):
            responses = search_results["responses"]
            total_matches = search_results.get("metadata", {}).get("total_matches", len(responses))
            
            search_text = f"Processing complete survey dataset - found {total_matches} total data points for analysis\n"
            search_text += f"Most relevant responses (top {len(responses)} matches):\n\n"
            search_text += "Survey responses:\n"
            
            for i, response in enumerate(responses):
                similarity = response.get('value', 0) * 100
                search_text += f"{i + 1}. \"{response.get('text', '')}\" (relevance: {similarity:.1f}%)\n"
            
            # Add demographics and psychology if available
            if search_results.get("demographics"):
                search_text += f"\nDemographics: {search_results['demographics']}\n"
            if search_results.get("psychology"):
                search_text += f"\nPsychological insights: {search_results['psychology']}\n"
        
        # Build AI messages using the enhanced system
        messages = simple_ai_service.build_messages(
            system_prompt=simple_ai_service.default_system_prompt,
            conversation_history=[],
            current_question=message,
            search_context=search_text if search_text else None
        )
        
        # Generate AI response
        ai_response = await simple_ai_service.generate_response(
            messages=messages,
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse structured response using StreamlinedResponses
        structured_response = streamlined_responses.chat_completion_response(
            user_message_id=str(uuid.uuid4()),
            ai_response=ai_response,
            ai_message_id=str(uuid.uuid4()),
            processing_time=100.0,
            search_results=search_results,
            provider="openai",
            model="gpt-4o-mini"
        )
        
        # Format response for the frontend
        confidence_data = structured_response.get("confidence", {"score": 50})
        confidence_score = confidence_data.get("score", 50) if isinstance(confidence_data, dict) else confidence_data
        
        return SemanticChatResponse(
            sessionId=str(session_id) if session_id else str(uuid.uuid4()),
            question=message,
            answer=structured_response.get("conversationalResponse", ai_response),
            conversationalResponse=structured_response.get("conversationalResponse", ai_response),
            dataSnapshot=structured_response.get("dataSnapshot", {}),
            search_results=search_results,
            confidence=float(confidence_score)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process semantic chat request"
        )

