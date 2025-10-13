from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.models import User
from app.utils.logging import get_logger
from app.models.schemas import (
    UserSurveyAccess as UserSurveyAccessSchema,
    UserSurveyFileAccess as UserSurveyFileAccessSchema,
    UserSurveyAccessCreate,
    UserSurveyAccessUpdate,
    UserSurveyFileAccessCreate,
    UserSurveyFileAccessUpdate,
    AccessType,
    BulkAccessGrant
)
from app.services.access_control_service import access_control_service

logger = get_logger(__name__)
router = APIRouter()


@router.post("/surveys/{survey_id}/access", response_model=UserSurveyAccessSchema)
async def grant_survey_access(
    survey_id: UUID,
    access_data: UserSurveyAccessCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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


@router.get("/users/{user_id}/surveys", response_model=List[UserSurveyAccessSchema])
async def get_user_surveys(
    user_id: UUID,
    access_type: Optional[AccessType] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all surveys a user has access to."""
    # Users can check their own surveys, admins can check anyone's
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view user surveys"
        )
    
    surveys = await access_control_service.get_user_surveys(db, user_id, access_type)
    return surveys


@router.get("/surveys/{survey_id}/users", response_model=List[UserSurveyAccessSchema])
async def get_survey_users(
    survey_id: UUID,
    access_type: Optional[AccessType] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all users with their access permissions (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role.role != "admin":
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


@router.get("/test-endpoint")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "test successful"}


@router.get("/debug-user/{user_id}")
async def debug_get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Debug endpoint to test user retrieval without authentication"""
    try:
        import uuid
        from sqlalchemy.orm import selectinload
        from app.services.auth_service import user_service
        
        user = await user_service.get_by_id(
            db, 
            uuid.UUID(user_id), 
            options=[selectinload(User.role)]
        )
        
        if not user:
            return {"error": "User not found", "user_id": user_id}
        
        return {
            "success": True,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "language": user.language,
                "email_confirmed": user.email_confirmed,
                "welcome_popup_dismissed": user.welcome_popup_dismissed,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "role": user.role.role if user.role else None,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "updated_at": user.updated_at.isoformat() if user.updated_at else None
            }
        }
        
    except Exception as e:
        return {"error": str(e), "user_id": user_id}

@router.get("/surveys-files")
async def get_surveys_and_files(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all surveys and files for access management (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Get all surveys and files for access management
        surveys_and_files = await access_control_service.get_surveys_and_files(db)
        return surveys_and_files
    except Exception as e:
        logger.error(f"Error getting surveys and files for access management: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get surveys and files: {str(e)}"
        )


@router.post("/survey/grant")
async def grant_survey_access_simple(
    request_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Grant survey access to a user (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Extract data from request body
        user_id = UUID(request_data["userId"])
        survey_id = UUID(request_data["surveyId"])
        access_type = request_data["accessType"]
        
        access = await access_control_service.grant_survey_access(
            db, user_id, survey_id, access_type, current_user.id
        )
        return {"message": "Survey access granted successfully", "access": access}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to grant survey access: {str(e)}"
        )


@router.post("/file/grant")
async def grant_file_access_simple(
    request_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Grant file access to a user (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Extract data from request body
        user_id = UUID(request_data["userId"])
        file_id = UUID(request_data["surveyFileId"])  # Frontend sends surveyFileId
        access_type = request_data["accessType"]
        
        access = await access_control_service.grant_file_access(
            db, user_id, file_id, access_type, current_user.id
        )
        return {"message": "File access granted successfully", "access": access}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to grant file access: {str(e)}"
        )


@router.post("/survey/revoke")
async def revoke_survey_access_simple(
    request_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Revoke survey access from a user (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Extract data from request body
        user_id = UUID(request_data["userId"])
        survey_id = UUID(request_data["surveyId"])
        
        success = await access_control_service.revoke_survey_access(db, user_id, survey_id)
        if success:
            return {"message": "Survey access revoked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to revoke survey access: {str(e)}"
        )


@router.post("/file/revoke")
async def revoke_file_access_simple(
    request_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Revoke file access from a user (admin only) - matches TypeScript API"""
    # Check admin permission
    if current_user.role.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        # Extract data from request body
        user_id = UUID(request_data["userId"])
        file_id = UUID(request_data["surveyFileId"])  # Frontend sends surveyFileId
        
        success = await access_control_service.revoke_file_access(db, user_id, file_id)
        if success:
            return {"message": "File access revoked successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Access not found"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to revoke file access: {str(e)}"
        )
