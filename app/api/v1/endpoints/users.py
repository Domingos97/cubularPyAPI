from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import get_current_regular_user, get_current_admin_user, common_parameters, CommonQueryParams, SimpleUser
from app.models.schemas import (
    User as UserSchema, 
    UserUpdate, 
    SuccessResponse,
    UserWithAccess
)
from app.utils.logging import get_logger
from app.utils.validation import ValidationHelpers
from fastapi import UploadFile, File
import os, uuid

logger = get_logger(__name__)

# Simple request model for welcome popup
class WelcomePopupRequest(BaseModel):
    dismissed: bool

router = APIRouter()

# Debug endpoint to test authentication
@router.get("/auth-test")
async def test_auth(current_user: SimpleUser = Depends(get_current_regular_user)):
    """Test endpoint to verify authentication is working"""
    return {
        "authenticated": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role
    }


@router.get("/me", response_model=UserSchema)
async def get_current_regular_user_profile(
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get current user profile
    """
    return UserSchema(
        id=str(current_user.id),
        email=current_user.email,
        username=current_user.username,
        created_at=current_user.created_at.isoformat() if current_user.created_at else "",
        updated_at=current_user.updated_at.isoformat() if current_user.updated_at else "",
        preferred_personality=str(current_user.preferred_personality) if current_user.preferred_personality else None,
        language_preference=current_user.language,
        role_id=None,  # SimpleUser doesn't have roleid, we'll use role string
        role=current_user.role,
            role_details=None,  # SimpleUser doesn't have role object
            personality_details=None,  # Will be populated if needed
            welcome_popup_dismissed=current_user.welcome_popup_dismissed,
            has_ai_personalities_access=getattr(current_user, 'has_ai_personalities_access', False)
    )


@router.put("/me", response_model=UserSchema)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Update current user profile
    
    - **username**: New username
    - **language**: Preferred language
    - **preferred_personality_id**: Preferred AI personality ID
    - **welcome_popup_dismissed**: Whether welcome popup is dismissed
    """
    try:
        updated_user = await db.update_user_profile(
            current_user.id, 
            user_update
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User profile updated: {updated_user['email']}")
        
        return UserSchema(
            id=str(updated_user["id"]),
            email=updated_user["email"],
            username=updated_user["username"],
            language_preference=updated_user.get("language"),
            welcome_popup_dismissed=updated_user.get("welcome_popup_dismissed"),
            role=updated_user.get("role_name"),
            preferred_personality=str(updated_user.get("preferred_personality")) if updated_user.get("preferred_personality") else None,
            created_at=str(updated_user.get("created_at")) if updated_user.get("created_at") else None,
            updated_at=str(updated_user.get("updated_at")) if updated_user.get("updated_at") else None
        )
        
    except Exception as e:
        logger.error(f"User profile update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.post('/me/avatar')
async def upload_current_user_avatar(
    file: UploadFile = File(...),
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Upload avatar for current user. Validates type/size, stores under /static/avatars and updates user record.
    """
    try:
        # Basic validations
        allowed_types = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail='Unsupported file type')

        contents = await file.read()
        max_size = 2 * 1024 * 1024  # 2MB
        if len(contents) > max_size:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail='File too large')

        # Remove previous avatar file if it exists and is stored in our avatars folder
        try:
            existing = await db.get_user_by_id(str(current_user.id))
            if existing and existing.get('avatar'):
                old_avatar = existing.get('avatar')
                # Only attempt deletion for files under our avatars static path
                if isinstance(old_avatar, str) and old_avatar.startswith('/static/avatars/'):
                    # Check if any other user references the same avatar path. If so, skip deletion.
                    try:
                        count_row = await db.execute_fetchrow("SELECT COUNT(*) AS cnt FROM users WHERE avatar = $1", [old_avatar])
                        count = int(count_row.get('cnt')) if count_row and count_row.get('cnt') is not None else 0
                    except Exception as e_count:
                        # If we can't determine references, skip deletion to be safe
                        logger.warning(f"Could not count avatar references for {old_avatar}: {e_count}")
                        count = 2

                    if count <= 1:
                        old_filename = old_avatar.split('/')[-1]
                        old_path = os.path.join('app', 'static', 'avatars', old_filename)
                        try:
                            if os.path.exists(old_path):
                                os.remove(old_path)
                                logger.info(f"Deleted old avatar file for user {current_user.id}: {old_path}")
                        except Exception as e_del:
                            # Log and continue - don't fail the upload because of file deletion issues
                            logger.warning(f"Failed to delete old avatar file {old_path} for user {current_user.id}: {e_del}")
                    else:
                        logger.info(f"Skipping deletion of avatar {old_avatar} because it's referenced by {count} users")
        except Exception as e_get:
            # If we fail to fetch the existing user for any reason, log and continue
            logger.warning(f"Could not fetch existing user to delete avatar for {current_user.id}: {e_get}")

        # Save new file
        avatars_dir = os.path.join('app', 'static', 'avatars')
        os.makedirs(avatars_dir, exist_ok=True)
        ext = (file.filename.rsplit('.', 1)[-1] if '.' in file.filename else 'png')
        filename = f"{uuid.uuid4()}.{ext}"
        save_path = os.path.join(avatars_dir, filename)
        with open(save_path, 'wb') as f:
            f.write(contents)

        public_path = f"/static/avatars/{filename}"

        # Update db with new avatar path
        # Use a tiny TempUpdate object so the lightweight DB service picks up avatar
        class TempUpdate: pass
        temp = TempUpdate()
        temp.avatar = public_path
        updated = await db.update_user_profile(str(current_user.id), temp)

        if not updated:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to update user profile with avatar')

        return { 'avatar': public_path }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Avatar upload failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Avatar upload failed')


@router.get("/me/stats")
async def get_current_regular_user_stats(
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get current user statistics
    """
    try:
        user_data = await db.get_user_with_stats(current_user.id)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user_data["stats"]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user statistics"
        )


@router.delete("/me", response_model=SuccessResponse)
async def delete_current_user_account(
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Delete current user account (soft delete)
    
    This will mark the account as inactive but preserve data for audit purposes.
    """
    try:
        # Instead of hard delete, we'll mark as inactive
        # In a real application, you might want to anonymize data
        
        from app.services.lightweight_auth_service import auth_service
        
        # Revoke all tokens
        await auth_service.revoke_all_user_tokens(db, current_user.id)
        
        # Mark user as inactive (you might want to add is_active field to User model)
        # For now, we'll just log the deletion request
        logger.info(f"User deletion requested: {current_user.email}")
        
        return SuccessResponse(
            message="Account deletion request processed. Your account has been deactivated."
        )
        
    except Exception as e:
        logger.error(f"User deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )


# Admin endpoints
@router.get("/", response_model=List[UserSchema])
async def get_all_users(
    admin_user: SimpleUser = Depends(get_current_admin_user),
    params: CommonQueryParams = Depends(common_parameters),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get all users (admin only)
    
    - **skip**: Number of users to skip
    - **limit**: Maximum number of users to return
    - **search**: Search term for username or email
    - **sort_by**: Field to sort by
    - **sort_order**: Sort order (asc/desc)
    """
    try:
                
        users = await db.get_multi_users(
            skip=params.skip,
            limit=params.limit
        )
        
        return [
            UserSchema(
                id=str(user["id"]),
                email=user["email"],
                username=user["username"],
                        created_at=user["created_at"].isoformat() if user.get("created_at") else "",
                        updated_at=user["updated_at"].isoformat() if user.get("updated_at") else "",
                        preferred_personality=str(user["preferred_personality"]) if user.get("preferred_personality") else None,
                        language_preference=user.get("language"),
                        role_id=str(user["roleid"]) if user.get("roleid") else None,
                        role=user.get("role_name"),
                        role_details={
                            "id": str(user["roleid"]),
                            "name": user.get("role_name")
                        } if user.get("roleid") else None,
                        personality_details=None,
                        welcome_popup_dismissed=user.get("welcome_popup_dismissed", False),
                        has_ai_personalities_access=user.get("has_ai_personalities_access", False)
            )
            for user in users
        ]
        
    except Exception as e:
        logger.error(f"Get all users error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get users"
        )


@router.get("/{user_id}", response_model=UserWithAccess)
async def get_user_by_id(
    user_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get user by ID with access permissions
    - Users can only view their own profile
    - Admins can view any user profile
    """
    # Authorization: Users can only view their own profile, admins can view any
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Users can only view their own profile"
        )
    try:
        import uuid
        from app.models.schemas import UserWithAccess, UserSurveyAccessWithDetails, UserSurveyFileAccessWithDetails, SurveyDetails, FileDetails
        
        user = await db.get_user_by_id_simple(
            user_id
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get user's survey access permissions
        user_survey_access_data = await db.get_user_survey_access(user["id"])
        user_file_access_data = await db.get_user_file_access(user["id"])
        
        # Get user's plans with details
        user_plans_data = await db.get_user_plans_with_details(str(user["id"]))
        
        # Transform survey access to match frontend format
        survey_access_list = []
        for access in user_survey_access_data:
            survey_access_list.append(UserSurveyAccessWithDetails(
                id=str(access["id"]),
                survey_id=str(access["survey_id"]),
                access_type=access["access_type"],
                granted_at=access["granted_at"].isoformat() if access["granted_at"] else None,
                expires_at=access["expires_at"].isoformat() if access["expires_at"] else None,
                is_active=access["is_active"],
                surveys=SurveyDetails(
                    id=str(access["survey_id"]),
                    title=access["title"] or "Untitled Survey",
                    category=access["category"]
                )
            ))
        
        # Transform file access to match frontend format
        file_access_list = []
        for access in user_file_access_data:
            file_access_list.append(UserSurveyFileAccessWithDetails(
                id=str(access["id"]),
                survey_file_id=str(access["survey_file_id"]),
                access_type=access["access_type"],
                granted_at=access["granted_at"].isoformat() if access["granted_at"] else None,
                expires_at=access["expires_at"].isoformat() if access["expires_at"] else None,
                is_active=access["is_active"],
                survey_files=FileDetails(
                    id=str(access["survey_file_id"]),
                    filename=access["filename"],
                    surveys=SurveyDetails(
                        id=str(access["survey_id"]),
                        title=access["title"] or "Untitled Survey",
                        category=access["category"]
                    )
                )
            ))
        
        return UserWithAccess(
            id=str(user["id"]),
            email=user["email"],
            username=user["username"],
            language_preference=user.get("language"),
            preferred_personality=str(user.get("preferred_personality")) if user.get("preferred_personality") else None,
            welcome_popup_dismissed=user.get("welcome_popup_dismissed"),
            has_ai_personalities_access=user.get("has_ai_personalities_access", False),
            created_at=user["created_at"].isoformat() if user["created_at"] else None,
            updated_at=user["updated_at"].isoformat() if user["updated_at"] else None,
            role=user["role_name"] if user["role_name"] else None,
            user_survey_access=survey_access_list,
            user_survey_file_access=file_access_list,
            user_plans=user_plans_data
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user by ID error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.put("/welcome-popup-dismissed", response_model=SuccessResponse)
async def update_welcome_popup_dismissed(
    request: WelcomePopupRequest,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Update welcome popup dismissed status
    """
    try:
        logger.info(f"Welcome popup dismissal request from user {current_user.id} ({current_user.email}): dismissed={request.dismissed}")
        logger.info(f"User details - role: {current_user.role}, is_active: {current_user.is_active}")
        
        update_data = UserUpdate(welcome_popup_dismissed=request.dismissed)
        updated_user = await db.update_user_profile(
            current_user.id, 
            update_data
        )
        
        if not updated_user:
            logger.error(f"User {current_user.id} not found when updating welcome popup status")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"Successfully updated welcome popup dismissed status for user {current_user.id}: {request.dismissed}")
        return SuccessResponse(message="Welcome popup status updated successfully")
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 401, 403, 404)
        raise
    except Exception as e:
        logger.error(f"Update welcome popup error for user {getattr(current_user, 'id', 'unknown')}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update welcome popup status"
        )


@router.put("/{user_id}", response_model=UserSchema)
async def update_user_by_id(
    user_id: str,
    user_update: UserUpdate,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Update user by ID
    - Users can only update their own profile
    - Admins can update any user profile
    """
    # Authorization: Users can only update their own profile, admins can update any
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Users can only update their own profile"
        )
    try:
        # Only admins are allowed to change another user's role
        if getattr(user_update, 'role', None) is not None:
            if current_user.role != 'admin':
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only admin users can change roles"
                )
            # Optionally validate role string length / format
            if not isinstance(user_update.role, str) or len(user_update.role) > 64:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid role"
                )
        # Validate user_id is a proper UUID format
        import uuid
        try:
            uuid_obj = uuid.UUID(user_id)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid UUID format for user_id '{user_id}': {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID format"
            )
        
        updated_user = await db.update_user_profile(
            user_id, 
            user_update
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User updated: {updated_user['email']} by {current_user.email} (role: {current_user.role})")
        
        return UserSchema(
            id=str(updated_user["id"]),
            email=updated_user["email"],
            username=updated_user["username"],
            language_preference=updated_user.get("language"),
            welcome_popup_dismissed=updated_user.get("welcome_popup_dismissed"),
            role=updated_user.get("role_name"),
            preferred_personality=str(updated_user.get("preferred_personality")) if updated_user.get("preferred_personality") else None,
            created_at=str(updated_user.get("created_at")) if updated_user.get("created_at") else None,
            updated_at=str(updated_user.get("updated_at")) if updated_user.get("updated_at") else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user by ID error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/{user_id}", response_model=SuccessResponse)
async def delete_user_by_id(
    user_id: str,
    admin_user: SimpleUser = Depends(get_current_admin_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Delete user by ID (admin only)
    """
    try:
        import uuid
        
        success = await db.delete_user_by_id(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User deleted by admin: {user_id} by {admin_user.email}")
        
        return SuccessResponse(message="User deleted successfully")
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete user by ID error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.get("/{user_id}/stats")
async def get_user_stats_by_id(
    user_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get user statistics by ID
    - Users can only view their own stats
    - Admins can view any user's stats
    """
    # Authorization: Users can only view their own stats, admins can view any
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Users can only view their own statistics"
        )
    try:
        import uuid
        
        user_data = await db.get_user_with_stats(user_id)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "user": UserSchema(
                id=str(user_data["user"]["id"]),
                email=user_data["user"]["email"],
                username=user_data["user"]["username"],
                language_preference=user_data["user"]["language"],
                welcome_popup_dismissed=user_data["user"]["welcome_popup_dismissed"],
                role=user_data["user"]["role_name"],
                preferred_personality=str(user_data["user"].get("preferred_personality")) if user_data["user"].get("preferred_personality") else None,
                created_at=str(user_data["user"]["created_at"]) if user_data["user"]["created_at"] else None,
                updated_at=str(user_data["user"]["updated_at"]) if user_data["user"]["updated_at"] else None
            ),
            "stats": user_data["stats"]
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user stats by ID error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user statistics"
        )


# Additional endpoints to match TypeScript API
@router.get("/profile", response_model=UserSchema)
async def get_user_profile(
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get user profile (alias for /me)
    """
    return UserSchema(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        language_preference=current_user.language,
        welcome_popup_dismissed=current_user.welcome_popup_dismissed,
        role=current_user.role,
        preferred_personality=str(current_user.preferred_personality) if current_user.preferred_personality else None,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
        updated_at=current_user.updated_at.isoformat() if current_user.updated_at else None
    )


@router.put("/profile", response_model=UserSchema)
async def update_user_profile_alias(
    user_update: UserUpdate,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Update user profile (alias for /me)
    """
    return await update_current_user_profile(user_update, current_user, db)


@router.get("/me/preferred-personality")
async def get_user_preferred_personality(
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get user's preferred AI personality
    """
    return {
        "preferred_personality": str(current_user.preferred_personality) if current_user.preferred_personality else None
    }


@router.put("/me/preferred-personality", response_model=SuccessResponse)
async def update_user_preferred_personality(
    preferred_personality_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Update user's preferred AI personality
    """
    try:
        import uuid
        personality_uuid = uuid.UUID(preferred_personality_id) if preferred_personality_id != "null" else None
        
        update_data = UserUpdate(preferred_personality=personality_uuid)
        updated_user = await db.update_user_profile(
            current_user.id, 
            update_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return SuccessResponse(message="Preferred personality updated successfully")
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid personality ID format"
        )
    except Exception as e:
        logger.error(f"Update preferred personality error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferred personality"
        )


@router.get("/me/language")
async def get_user_language(
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get user's preferred language
    """
    return {
        "language": current_user.language
    }


@router.put("/me/language", response_model=SuccessResponse)
async def update_user_language(
    language: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Update user's preferred language
    """
    try:
        update_data = UserUpdate(language=language)
        updated_user = await db.update_user_profile(
            current_user.id, 
            update_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return SuccessResponse(message="Language updated successfully")
        
    except Exception as e:
        logger.error(f"Update language error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update language"
        )


@router.get("/{user_id}/settings")
async def get_user_settings(
    user_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get user settings (language, welcome popup status, etc.)
    """
    try:
        # Check if user is accessing their own settings or is admin
        import uuid
        user_uuid = uuid.UUID(user_id)
        
        if str(current_user.id) != str(user_uuid) and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get user by ID
        user = await db.get_user_by_id_simple(str(user_uuid))
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "language": user.get("language") or "en-US",
            "welcome_popup_dismissed": user.get("welcome_popup_dismissed") or False
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user settings error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user settings"
        )


@router.put("/{user_id}/settings", response_model=SuccessResponse)
async def update_user_settings(
    user_id: str,
    language: Optional[str] = None,
    welcome_popup_dismissed: Optional[bool] = None,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Update user settings (language, welcome popup status, etc.)
    """
    try:
        # Check if user is updating their own settings or is admin
        import uuid
        user_uuid = uuid.UUID(user_id)
        
        if str(current_user.id) != str(user_uuid) and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Validate that at least one setting is provided
        if language is None and welcome_popup_dismissed is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid settings provided"
            )
        
        # Build update data
        update_data = UserUpdate()
        if language is not None:
            update_data.language = language
        if welcome_popup_dismissed is not None:
            update_data.welcome_popup_dismissed = welcome_popup_dismissed
        
        # Update user settings
        updated_user = await db.update_user_profile(
            str(user_uuid), 
            update_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return SuccessResponse(message="User settings updated successfully")
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update user settings error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user settings"
        )


@router.post("/{user_id}/assign-plan", response_model=SuccessResponse)
async def assign_plan_to_user(
    user_id: str,
    plan_id: str,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Admin endpoint to assign a plan to a user.
    Requires admin privileges.
    """
    try:
        import uuid
        user_uuid = uuid.UUID(user_id)
        plan_uuid = uuid.UUID(plan_id)
        
        # Import plan service
        from app.services.plan_service import plan_service
        
        # Assign plan to user
        result = await plan_service.assign_plan_to_user(
            db=db,
            user_id=user_uuid,
            plan_id=plan_uuid,
            admin_user_id=current_user.id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to assign plan to user"
            )
        
        return SuccessResponse(message="Plan assigned successfully")
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID or plan ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Assign plan error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign plan"
        )


@router.post("/{user_id}/revoke-plan", response_model=SuccessResponse)
async def revoke_plan_from_user(
    user_id: str,
    user_plan_id: Optional[str] = None,
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Admin endpoint to revoke a user's plan.
    If user_plan_id is provided, revokes that specific plan.
    Otherwise, revokes the user's current active plan.
    Requires admin privileges.
    """
    try:
        import uuid
        user_uuid = uuid.UUID(user_id)
        user_plan_uuid = uuid.UUID(user_plan_id) if user_plan_id else None
        
        # Import plan service
        from app.services.plan_service import plan_service
        
        # Revoke plan from user
        result = await plan_service.revoke_plan_from_user(
            db=db,
            user_id=user_uuid,
            user_plan_id=user_plan_uuid,
            admin_user_id=current_user.id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active plan found for user"
            )
        
        return SuccessResponse(message="Plan revoked successfully")
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID or plan ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Revoke plan error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke plan"
        )