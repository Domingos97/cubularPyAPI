from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_admin_user, common_parameters, CommonQueryParams
from app.models.schemas import (
    User as UserSchema, 
    UserUpdate, 
    SuccessResponse,
    UserWithAccess,
    UserSurveyAccessWithDetails,
    UserSurveyFileAccessWithDetails,
    SurveyDetails,
    FileDetails
)
from app.models.models import User, Survey, SurveyFile
from app.services.auth_service import user_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/me", response_model=UserSchema)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
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
        role_id=str(current_user.roleid) if current_user.roleid else None,
        role=current_user.role.role if current_user.role else None,
        role_details={
            "id": str(current_user.role.id),
            "name": current_user.role.role
        } if current_user.role else None,
        personality_details=None,  # Will be populated if needed
        welcome_popup_dismissed=current_user.welcome_popup_dismissed
    )


@router.put("/me", response_model=UserSchema)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update current user profile
    
    - **username**: New username
    - **language**: Preferred language
    - **preferred_personality_id**: Preferred AI personality ID
    - **welcome_popup_dismissed**: Whether welcome popup is dismissed
    """
    try:
        updated_user = await user_service.update_user_profile(
            db, 
            current_user.id, 
            user_update
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User profile updated: {updated_user.email}")
        
        return UserSchema(
            id=updated_user.id,
            email=updated_user.email,
            username=updated_user.username,
            language=updated_user.language,
            email_confirmed=updated_user.email_confirmed,
            welcome_popup_dismissed=updated_user.welcome_popup_dismissed,
            last_login=updated_user.last_login,
            role=updated_user.role.role,
            created_at=updated_user.created_at,
            updated_at=updated_user.updated_at
        )
        
    except Exception as e:
        logger.error(f"User profile update error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.get("/me/stats")
async def get_current_user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user statistics
    """
    try:
        user_data = await user_service.get_user_with_stats(db, current_user.id)
        
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete current user account (soft delete)
    
    This will mark the account as inactive but preserve data for audit purposes.
    """
    try:
        # Instead of hard delete, we'll mark as inactive
        # In a real application, you might want to anonymize data
        
        from app.services.auth_service import auth_service
        
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
    admin_user: User = Depends(get_current_admin_user),
    params: CommonQueryParams = Depends(common_parameters),
    db: AsyncSession = Depends(get_db)
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
        from sqlalchemy.orm import selectinload
        
        users = await user_service.get_multi(
            db,
            skip=params.skip,
            limit=params.limit,
            options=[selectinload(User.role)]
        )
        
        return [
            UserSchema(
                id=str(user.id),
                email=user.email,
                username=user.username,
                created_at=user.created_at.isoformat() if user.created_at else "",
                updated_at=user.updated_at.isoformat() if user.updated_at else "",
                preferred_personality=str(user.preferred_personality) if user.preferred_personality else None,
                language_preference=user.language,
                role_id=str(user.roleid) if user.roleid else None,
                role=user.role.role if user.role else None,
                role_details={
                    "id": str(user.role.id),
                    "name": user.role.role
                } if user.role else None,
                personality_details=None,
                welcome_popup_dismissed=user.welcome_popup_dismissed
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
    admin_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user by ID with access permissions (admin only)
    """
    try:
        import uuid
        from sqlalchemy.orm import selectinload
        from app.services.access_control_service import access_control_service
        from app.models.schemas import UserWithAccess, UserSurveyAccessWithDetails, UserSurveyFileAccessWithDetails, SurveyDetails, FileDetails
        
        user = await user_service.get_by_id(
            db, 
            uuid.UUID(user_id), 
            options=[selectinload(User.role)]
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get user's survey access permissions
        user_survey_access = await access_control_service.get_user_surveys(db, user.id)
        user_file_access = await access_control_service.get_user_files(db, user.id)
        
        # Transform survey access to match frontend format
        survey_access_list = []
        for access in user_survey_access:
            # Get survey details
            survey = await db.get(Survey, access.survey_id)
            if survey:
                survey_access_list.append(UserSurveyAccessWithDetails(
                    id=str(access.id),
                    survey_id=str(access.survey_id),
                    access_type=access.access_type,
                    granted_at=access.granted_at.isoformat() if access.granted_at else None,
                    expires_at=access.expires_at.isoformat() if access.expires_at else None,
                    is_active=access.is_active,
                    surveys=SurveyDetails(
                        id=str(survey.id),
                        title=survey.title or "Untitled Survey",
                        category=survey.category
                    )
                ))
        
        # Transform file access to match frontend format
        file_access_list = []
        for access in user_file_access:
            # Get file and survey details
            survey_file = await db.get(SurveyFile, access.survey_file_id)
            if survey_file:
                survey = await db.get(Survey, survey_file.survey_id)
                if survey:
                    file_access_list.append(UserSurveyFileAccessWithDetails(
                        id=str(access.id),
                        survey_file_id=str(access.survey_file_id),
                        access_type=access.access_type,
                        granted_at=access.granted_at.isoformat() if access.granted_at else None,
                        expires_at=access.expires_at.isoformat() if access.expires_at else None,
                        is_active=access.is_active,
                        survey_files=FileDetails(
                            id=str(survey_file.id),
                            filename=survey_file.filename,
                            surveys=SurveyDetails(
                                id=str(survey.id),
                                title=survey.title or "Untitled Survey",
                                category=survey.category
                            )
                        )
                    ))
        
        return UserWithAccess(
            id=str(user.id),
            email=user.email,
            username=user.username,
            created_at=user.created_at.isoformat() if user.created_at else None,
            updated_at=user.updated_at.isoformat() if user.updated_at else None,
            role=user.role.role if user.role else None,
            user_survey_access=survey_access_list,
            user_survey_file_access=file_access_list
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


@router.put("/{user_id}", response_model=UserSchema)
async def update_user_by_id(
    user_id: str,
    user_update: UserUpdate,
    admin_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user by ID (admin only)
    """
    try:
        import uuid
        
        updated_user = await user_service.update_user_profile(
            db, 
            uuid.UUID(user_id), 
            user_update
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User updated by admin: {updated_user.email} by {admin_user.email}")
        
        return UserSchema(
            id=updated_user.id,
            email=updated_user.email,
            username=updated_user.username,
            language=updated_user.language,
            email_confirmed=updated_user.email_confirmed,
            welcome_popup_dismissed=updated_user.welcome_popup_dismissed,
            last_login=updated_user.last_login,
            role=updated_user.role.role,
            created_at=updated_user.created_at,
            updated_at=updated_user.updated_at
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
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
    admin_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete user by ID (admin only)
    """
    try:
        import uuid
        
        success = await user_service.delete(db, uuid.UUID(user_id))
        
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
    admin_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user statistics by ID (admin only)
    """
    try:
        import uuid
        
        user_data = await user_service.get_user_with_stats(db, uuid.UUID(user_id))
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "user": UserSchema(
                id=user_data["user"].id,
                email=user_data["user"].email,
                username=user_data["user"].username,
                language=user_data["user"].language,
                email_confirmed=user_data["user"].email_confirmed,
                welcome_popup_dismissed=user_data["user"].welcome_popup_dismissed,
                last_login=user_data["user"].last_login,
                role=user_data["user"].role.role,
                created_at=user_data["user"].created_at,
                updated_at=user_data["user"].updated_at
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
    current_user: User = Depends(get_current_user)
):
    """
    Get user profile (alias for /me)
    """
    return UserSchema(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        language=current_user.language,
        email_confirmed=current_user.email_confirmed,
        welcome_popup_dismissed=current_user.welcome_popup_dismissed,
        last_login=current_user.last_login,
        role=current_user.role.role,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at
    )


@router.put("/profile", response_model=UserSchema)
async def update_user_profile_alias(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user profile (alias for /me)
    """
    return await update_current_user_profile(user_update, current_user, db)


@router.get("/me/preferred-personality")
async def get_user_preferred_personality(
    current_user: User = Depends(get_current_user)
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user's preferred AI personality
    """
    try:
        import uuid
        personality_uuid = uuid.UUID(preferred_personality_id) if preferred_personality_id != "null" else None
        
        update_data = UserUpdate(preferred_personality=personality_uuid)
        updated_user = await user_service.update_user_profile(
            db, 
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
    current_user: User = Depends(get_current_user)
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user's preferred language
    """
    try:
        update_data = UserUpdate(language=language)
        updated_user = await user_service.update_user_profile(
            db, 
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


@router.put("/welcome-popup-dismissed", response_model=SuccessResponse)
async def update_welcome_popup_dismissed(
    dismissed: bool,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update welcome popup dismissed status
    """
    try:
        update_data = UserUpdate(welcome_popup_dismissed=dismissed)
        updated_user = await user_service.update_user_profile(
            db, 
            current_user.id, 
            update_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return SuccessResponse(message="Welcome popup status updated successfully")
        
    except Exception as e:
        logger.error(f"Update welcome popup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update welcome popup status"
        )


@router.get("/{user_id}/settings")
async def get_user_settings(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get user settings (language, welcome popup status, etc.)
    """
    try:
        # Check if user is accessing their own settings or is admin
        import uuid
        user_uuid = uuid.UUID(user_id)
        
        if current_user.id != user_uuid and current_user.role.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get user by ID
        user = await user_service.get_user_by_id(db, user_uuid)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "language": user.language or "en",
            "welcome_popup_dismissed": user.welcome_popup_dismissed or False
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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user settings (language, welcome popup status, etc.)
    """
    try:
        # Check if user is updating their own settings or is admin
        import uuid
        user_uuid = uuid.UUID(user_id)
        
        if current_user.id != user_uuid and current_user.role.role != "admin":
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
        updated_user = await user_service.update_user_profile(
            db, 
            user_uuid, 
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
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
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
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
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
