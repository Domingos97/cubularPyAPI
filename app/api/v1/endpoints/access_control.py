from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import get_current_regular_user, get_current_admin_user, SimpleUser
from app.utils.logging import get_logger
from app.models.schemas import (
    UserSurveyAccess as UserSurveyAccessSchema,
    UserSurveyFileAccess as UserSurveyFileAccessSchema,
    UserSurveyAccessCreate,
    UserSurveyFileAccessCreate,
    AccessType,
    BulkAccessGrant, Survey
)
from app.services.access_control_service import access_control_service

logger = get_logger(__name__)
router = APIRouter()


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


@router.get(
    "/my-surveys",
    response_model=List[Survey],
    summary="Get surveys user has access to"
)
async def get_my_surveys(
    skip: int = 0,
    limit: int = 100,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get surveys that the current user has access to.
    This includes surveys they own and surveys they have been granted access to.
    For non-admin users, only completed surveys are shown (pending surveys are hidden).
    """
    try:
        # Check if user is admin
        is_admin = current_user.role in ["admin", "super_admin"]
        
        logger.info(f"User {current_user.id} ({current_user.email}) requesting surveys, is_admin: {is_admin}")
        
        surveys_data = []
        
        if is_admin:
            # Admin users get access to all completed surveys (and pending ones if needed)
            query = """
            SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
                   s.number_participants, s.total_files, s.processing_status, s.created_at, s.updated_at
            FROM surveys s
            WHERE s.processing_status = 'completed' OR s.processing_status IS NULL
            ORDER BY s.created_at DESC
            OFFSET $1 LIMIT $2
            """
            surveys_data = await db.execute_query(query, [skip, limit])
            logger.info(f"Admin user: found {len(surveys_data)} surveys")
        else:
            # Regular users need explicit access via user_survey_access table
            logger.info(f"Regular user {current_user.id}, checking user_survey_access table")
            
            # Debug: Check what access records exist for this user
            access_check_query = """
            SELECT usa.id, usa.survey_id, usa.access_type, usa.is_active, s.title, s.processing_status
            FROM user_survey_access usa
            LEFT JOIN surveys s ON usa.survey_id = s.id
            WHERE usa.user_id = $1
            """
            access_records = await db.execute_query(access_check_query, [current_user.id])
            logger.info(f"Found {len(access_records)} access records for user {current_user.id}: {access_records}")
            
            # Also check if surveys exist at all
            all_surveys_query = "SELECT COUNT(*) as count FROM surveys"
            all_surveys_count = await db.execute_fetchrow(all_surveys_query)
            logger.info(f"Total surveys in database: {all_surveys_count['count'] if all_surveys_count else 0}")
            
            query = """
            SELECT s.id, s.title, s.category, s.description, s.ai_suggestions, 
                   s.number_participants, s.total_files, s.processing_status, s.created_at, s.updated_at
            FROM surveys s
            INNER JOIN user_survey_access usa ON s.id = usa.survey_id
            WHERE usa.user_id = $1 AND usa.is_active = true 
              AND (s.processing_status = 'completed' OR s.processing_status IS NULL)
            ORDER BY s.created_at DESC
            OFFSET $2 LIMIT $3
            """
            surveys_data = await db.execute_query(query, [current_user.id, skip, limit])
            logger.info(f"Regular user: found {len(surveys_data)} surveys after applying filters")
        
        surveys = []
        for survey_data in surveys_data:
            surveys.append(Survey(
                id=str(survey_data["id"]),
                title=survey_data["title"],
                category=survey_data["category"] or "",
                description=survey_data["description"] or "",
                ai_suggestions=survey_data.get("ai_suggestions", False),
                number_participants=survey_data.get("number_participants", 0),
                total_files=survey_data.get("total_files", 0),
                created_at=survey_data["created_at"].isoformat() if survey_data.get("created_at") else "",
                updated_at=survey_data["updated_at"].isoformat() if survey_data.get("updated_at") else "",
                # Backward compatibility fields
                fileid=None,
                filename=None,
                createdat=survey_data["created_at"].isoformat() if survey_data.get("created_at") else "",
                storage_path=None,
                primary_language=None,
                files=[]
            ))
        
        return surveys
        
    except Exception as e:
        logger.error(f"Error getting user accessible surveys: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve accessible surveys"
        )



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


@router.get("/users/{user_id}/files", response_model=List[UserSurveyFileAccessSchema])
async def get_user_files(
    user_id: UUID,
    access_type: Optional[AccessType] = None,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Get all files a user has access to."""
    # Users can check their own files, admins can check anyone's
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view user files"
        )
    
    files = await access_control_service.get_user_files(db, user_id, access_type)
    return files


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


# Additional endpoints to match TypeScript API patterns

@router.get("/users")
async def get_all_users_with_access(
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Get all users with their access permissions (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Get all users with their access permissions
        users_with_access = await access_control_service.get_all_users_with_access(db)
        return users_with_access
    except Exception as e:
        logger.error(f"Error getting all users with access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get users with access: {str(e)}"
        )


@router.get("/surveys-files")
async def get_surveys_and_files(
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """Get all surveys and files for access management (admin only) - matches TypeScript API"""
    try:
        # Get all surveys with basic info
        surveys_query = """
            SELECT s.id, s.title, s.category, s.description, 
                   s.processing_status, s.created_at, s.updated_at
            FROM surveys s
            ORDER BY s.created_at DESC
        """
        surveys = await db.execute_query(surveys_query)
        
        surveys_data = []
        total_files = 0
        
        for survey in surveys:
            # Get files for this survey
            files_query = """
                SELECT sf.id, sf.filename, sf.file_size, sf.storage_path,
                       sf.file_hash, sf.upload_date, sf.created_at, sf.updated_at
                FROM survey_files sf
                WHERE sf.survey_id = $1
                ORDER BY sf.created_at DESC
            """
            survey_files_rows = await db.execute_query(files_query, [survey['id']])
            
            survey_files = []
            for file_row in survey_files_rows:
                survey_files.append({
                    "id": str(file_row['id']),
                    "filename": file_row['filename'],
                    "file_size": file_row['file_size'],
                    "storage_path": file_row['storage_path'],
                    "file_hash": file_row['file_hash'],
                    "upload_date": file_row['upload_date'].isoformat() if file_row['upload_date'] else None,
                    "created_at": file_row['created_at'].isoformat() if file_row['created_at'] else None,
                    "updated_at": file_row['updated_at'].isoformat() if file_row['updated_at'] else None
                })
            
            total_files += len(survey_files)
            
            survey_data = {
                "id": str(survey['id']),
                "title": survey['title'],
                "category": survey['category'] or "general",
                "description": survey['description'],
                "processing_status": survey['processing_status'] or "pending",
                "created_at": survey['created_at'].isoformat() if survey['created_at'] else None,
                "updated_at": survey['updated_at'].isoformat() if survey['updated_at'] else None,
                "total_files": len(survey_files),
                "survey_files": survey_files
            }
            surveys_data.append(survey_data)
        
        # Return in the format expected by frontend
        return {
            "surveys": surveys_data,
            "total_surveys": len(surveys_data),
            "total_files": total_files
        }
        
    except Exception as e:
        logger.error(f"Error getting surveys and files for access management: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get surveys and files: {str(e)}"
        )


@router.post("/survey/grant")
async def grant_survey_access_simple(
    request_data: dict,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Grant survey access to a user (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Extract data from request body
        user_id = UUID(request_data["userId"])
        survey_id = UUID(request_data["surveyId"])
        access_type = request_data["accessType"]
        
        # Check if access already exists
        existing_access_query = """
        SELECT id FROM user_survey_access 
        WHERE user_id = $1 AND survey_id = $2
        """
        existing_access = await db.execute_fetchrow(existing_access_query, [str(user_id), str(survey_id)])
        
        if existing_access:
            # Update existing access
            update_query = """
            UPDATE user_survey_access 
            SET access_type = $1, granted_by = $2, granted_at = NOW(), is_active = true
            WHERE user_id = $3 AND survey_id = $4
            """
            await db.execute_command(update_query, [access_type, str(current_user.id), str(user_id), str(survey_id)])
        else:
            # Create new access
            import uuid
            access_id = str(uuid.uuid4())
            insert_query = """
            INSERT INTO user_survey_access (id, user_id, survey_id, access_type, granted_by, granted_at, is_active)
            VALUES ($1, $2, $3, $4, $5, NOW(), true)
            """
            await db.execute_command(insert_query, [access_id, str(user_id), str(survey_id), access_type, str(current_user.id)])
        
        return {"message": "Survey access granted successfully"}
    except Exception as e:
        logger.error(f"Error granting survey access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to grant survey access: {str(e)}"
        )


@router.post("/file/grant")
async def grant_file_access_simple(
    request_data: dict,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Grant file access to a user (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Extract data from request body
        user_id = UUID(request_data["userId"])
        file_id = UUID(request_data["surveyFileId"])  # Frontend sends surveyFileId
        access_type = request_data["accessType"]
        
        # Check if access already exists
        existing_access_query = """
        SELECT id FROM user_survey_file_access 
        WHERE user_id = $1 AND survey_file_id = $2
        """
        existing_access = await db.execute_fetchrow(existing_access_query, [str(user_id), str(file_id)])
        
        if existing_access:
            # Update existing access
            update_query = """
            UPDATE user_survey_file_access 
            SET access_type = $1, granted_by = $2, granted_at = NOW(), is_active = true
            WHERE user_id = $3 AND survey_file_id = $4
            """
            await db.execute_command(update_query, [access_type, str(current_user.id), str(user_id), str(file_id)])
        else:
            # Create new access
            import uuid
            access_id = str(uuid.uuid4())
            insert_query = """
            INSERT INTO user_survey_file_access (id, user_id, survey_file_id, access_type, granted_by, granted_at, is_active)
            VALUES ($1, $2, $3, $4, $5, NOW(), true)
            """
            await db.execute_command(insert_query, [access_id, str(user_id), str(file_id), access_type, str(current_user.id)])
        
        return {"message": "File access granted successfully"}
    except Exception as e:
        logger.error(f"Error granting file access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to grant file access: {str(e)}"
        )


@router.post("/survey/revoke")
async def revoke_survey_access_simple(
    request_data: dict,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Revoke survey access from a user (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Extract data from request body
        user_id = UUID(request_data["userId"])
        survey_id = UUID(request_data["surveyId"])
        
        # Check if access exists
        check_query = """
        SELECT id FROM user_survey_access 
        WHERE user_id = $1 AND survey_id = $2
        """
        existing_access = await db.execute_fetchrow(check_query, [str(user_id), str(survey_id)])
        
        if existing_access:
            # Revoke access by setting is_active to false
            revoke_query = """
            UPDATE user_survey_access 
            SET is_active = false
            WHERE user_id = $1 AND survey_id = $2
            """
            await db.execute_command(revoke_query, [str(user_id), str(survey_id)])
            return {"message": "Survey access revoked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking survey access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to revoke survey access: {str(e)}"
        )


@router.post("/file/revoke")
async def revoke_file_access_simple(
    request_data: dict,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """Revoke file access from a user (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Extract data from request body
        user_id = UUID(request_data["userId"])
        file_id = UUID(request_data["surveyFileId"])  # Frontend sends surveyFileId
        
        # Check if access exists
        check_query = """
        SELECT id FROM user_survey_file_access 
        WHERE user_id = $1 AND survey_file_id = $2
        """
        existing_access = await db.execute_fetchrow(check_query, [str(user_id), str(file_id)])
        
        if existing_access:
            # Revoke access by setting is_active to false
            revoke_query = """
            UPDATE user_survey_file_access 
            SET is_active = false
            WHERE user_id = $1 AND survey_file_id = $2
            """
            await db.execute_command(revoke_query, [str(user_id), str(file_id)])
            return {"message": "File access revoked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking file access: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to revoke file access: {str(e)}"
        )

"""

@router.get("/users/{user_id}/surveys", response_model=List[UserSurveyAccessSchema])
async def get_user_surveys(
    user_id: UUID,
    access_type: Optional[AccessType] = None,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: User = Depends(get_current_regular_user)
):
    #Get all surveys a user has access to.
    # Users can check their own surveys, admins can check anyone's
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view user surveys"
        )
    
    surveys = await access_control_service.get_user_surveys(db, user_id, access_type)
    return surveys
"""