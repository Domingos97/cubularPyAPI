from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

from app.models.models import UserSurveyAccess, UserSurveyFileAccess, User, Survey, SurveyFile
from app.models.schemas import (
    UserSurveyAccessCreate,
    UserSurveyAccessUpdate,
    UserSurveyFileAccessCreate,
    UserSurveyFileAccessUpdate,
    AccessType
)


class AccessControlService:
    """Service for managing user access control to surveys and files."""
    
    @staticmethod
    async def grant_survey_access(
        db: AsyncSession, 
        user_id: UUID, 
        survey_id: UUID, 
        access_type: AccessType,
        granted_by: UUID
    ) -> UserSurveyAccess:
        """Grant or update user access to a survey."""
        # Check if access already exists
        stmt = select(UserSurveyAccess).where(
            UserSurveyAccess.user_id == user_id,
            UserSurveyAccess.survey_id == survey_id
        )
        result = await db.execute(stmt)
        existing_access = result.scalars().first()
        
        if existing_access:
            # Update existing access
            existing_access.access_type = access_type
            await db.commit()
            await db.refresh(existing_access)
            return existing_access
        else:
            # Create new access
            new_access = UserSurveyAccess(
                user_id=user_id,
                survey_id=survey_id,
                access_type=access_type,
                granted_by=granted_by
            )
            db.add(new_access)
            await db.commit()
            await db.refresh(new_access)
            return new_access
    
    @staticmethod
    async def revoke_survey_access(
        db: AsyncSession, 
        user_id: UUID, 
        survey_id: UUID
    ) -> bool:
        """Revoke user access to a survey."""
        stmt = delete(UserSurveyAccess).where(
            UserSurveyAccess.user_id == user_id,
            UserSurveyAccess.survey_id == survey_id
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0
    
    @staticmethod
    async def get_user_survey_access(
        db: AsyncSession, 
        user_id: UUID, 
        survey_id: UUID
    ) -> Optional[UserSurveyAccess]:
        """Get user's access level to a specific survey."""
        stmt = select(UserSurveyAccess).where(
            UserSurveyAccess.user_id == user_id,
            UserSurveyAccess.survey_id == survey_id
        )
        result = await db.execute(stmt)
        return result.scalars().first()
    
    @staticmethod
    async def get_user_surveys(
        db: AsyncSession, 
        user_id: UUID,
        access_type: Optional[AccessType] = None
    ) -> List[UserSurveyAccess]:
        """Get all surveys a user has access to."""
        stmt = select(UserSurveyAccess).options(
            selectinload(UserSurveyAccess.survey)
        ).where(UserSurveyAccess.user_id == user_id)
        
        if access_type:
            stmt = stmt.where(UserSurveyAccess.access_type == access_type)
            
        result = await db.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def get_survey_users(
        db: AsyncSession, 
        survey_id: UUID,
        access_type: Optional[AccessType] = None
    ) -> List[UserSurveyAccess]:
        """Get all users with access to a survey."""
        stmt = select(UserSurveyAccess).options(
            selectinload(UserSurveyAccess.user)
        ).where(UserSurveyAccess.survey_id == survey_id)
        
        if access_type:
            stmt = stmt.where(UserSurveyAccess.access_type == access_type)
            
        result = await db.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def grant_file_access(
        db: AsyncSession,
        user_id: UUID,
        file_id: UUID,
        access_type: AccessType,
        granted_by: UUID
    ) -> UserSurveyFileAccess:
        """Grant or update user access to a file."""
        # Check if access already exists
        stmt = select(UserSurveyFileAccess).where(
            UserSurveyFileAccess.user_id == user_id,
            UserSurveyFileAccess.survey_file_id == file_id
        )
        result = await db.execute(stmt)
        existing_access = result.scalars().first()
        
        if existing_access:
            # Update existing access
            existing_access.access_type = access_type
            await db.commit()
            await db.refresh(existing_access)
            return existing_access
        else:
            # Create new access
            new_access = UserSurveyFileAccess(
                user_id=user_id,
                survey_file_id=file_id,
                access_type=access_type,
                granted_by=granted_by
            )
            db.add(new_access)
            await db.commit()
            await db.refresh(new_access)
            return new_access
    
    @staticmethod
    async def revoke_file_access(
        db: AsyncSession,
        user_id: UUID,
        file_id: UUID
    ) -> bool:
        """Revoke user access to a file."""
        stmt = delete(UserSurveyFileAccess).where(
            UserSurveyFileAccess.user_id == user_id,
            UserSurveyFileAccess.survey_file_id == file_id
        )
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0
    
    @staticmethod
    async def get_user_file_access(
        db: AsyncSession,
        user_id: UUID,
        file_id: UUID
    ) -> Optional[UserSurveyFileAccess]:
        """Get user's access level to a specific file."""
        stmt = select(UserSurveyFileAccess).where(
            UserSurveyFileAccess.user_id == user_id,
            UserSurveyFileAccess.survey_file_id == file_id
        )
        result = await db.execute(stmt)
        return result.scalars().first()
    
    @staticmethod
    async def get_user_files(
        db: AsyncSession,
        user_id: UUID,
        access_type: Optional[AccessType] = None
    ) -> List[UserSurveyFileAccess]:
        """Get all files a user has access to."""
        stmt = select(UserSurveyFileAccess).options(
            selectinload(UserSurveyFileAccess.survey_file)
        ).where(UserSurveyFileAccess.user_id == user_id)
        
        if access_type:
            stmt = stmt.where(UserSurveyFileAccess.access_type == access_type)
            
        result = await db.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def get_file_users(
        db: AsyncSession,
        file_id: UUID,
        access_type: Optional[AccessType] = None
    ) -> List[UserSurveyFileAccess]:
        """Get all users with access to a file."""
        stmt = select(UserSurveyFileAccess).options(
            selectinload(UserSurveyFileAccess.user)
        ).where(UserSurveyFileAccess.survey_file_id == file_id)
        
        if access_type:
            stmt = stmt.where(UserSurveyFileAccess.access_type == access_type)
            
        result = await db.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def check_survey_permission(
        db: AsyncSession,
        user_id: UUID,
        survey_id: UUID,
        required_access: AccessType
    ) -> bool:
        """Check if user has required permission for a survey."""
        access = await AccessControlService.get_user_survey_access(db, user_id, survey_id)
        if not access:
            return False
        
        # Define access hierarchy: admin > edit > read
        access_levels = {
            AccessType.READ: 1,
            AccessType.EDIT: 2,
            AccessType.ADMIN: 3
        }
        
        user_level = access_levels.get(access.access_type, 0)
        required_level = access_levels.get(required_access, 0)
        
        return user_level >= required_level
    
    @staticmethod
    async def check_file_permission(
        db: AsyncSession,
        user_id: UUID,
        file_id: UUID,
        required_access: AccessType
    ) -> bool:
        """Check if user has required permission for a file."""
        access = await AccessControlService.get_user_file_access(db, user_id, file_id)
        if not access:
            return False
        
        # Define access hierarchy: admin > edit > read
        access_levels = {
            AccessType.READ: 1,
            AccessType.EDIT: 2,
            AccessType.ADMIN: 3
        }
        
        user_level = access_levels.get(access.access_type, 0)
        required_level = access_levels.get(required_access, 0)
        
        return user_level >= required_level
    
    @staticmethod
    async def bulk_grant_survey_access(
        db: AsyncSession,
        survey_id: UUID,
        user_access_list: List[dict]  # [{"user_id": UUID, "access_type": AccessType}]
    ) -> List[UserSurveyAccess]:
        """Grant access to multiple users for a survey."""
        results = []
        for user_access in user_access_list:
            access = await AccessControlService.grant_survey_access(
                db, 
                user_access["user_id"], 
                survey_id, 
                user_access["access_type"]
            )
            results.append(access)
        return results
    
    @staticmethod
    async def bulk_grant_file_access(
        db: AsyncSession,
        file_id: UUID,
        user_access_list: List[dict]  # [{"user_id": UUID, "access_type": AccessType}]
    ) -> List[UserSurveyFileAccess]:
        """Grant access to multiple users for a file."""
        results = []
        for user_access in user_access_list:
            access = await AccessControlService.grant_file_access(
                db,
                user_access["user_id"],
                file_id,
                user_access["access_type"]
            )
            results.append(access)
        return results

    @staticmethod
    async def get_all_users_with_access(db: AsyncSession) -> dict:
        """Get all users with their access permissions (admin only)"""
        try:
            # Get all users with their survey and file access
            stmt = select(User).options(
                selectinload(User.survey_access),
                selectinload(User.file_access)
            )
            result = await db.execute(stmt)
            users = result.scalars().all()
            
            users_data = []
            for user in users:
                user_data = {
                    "id": str(user.id),
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.role if user.role else "user",
                    "survey_access": [
                        {
                            "survey_id": str(access.survey_id),
                            "access_type": access.access_type.value
                        }
                        for access in user.survey_access
                    ],
                    "file_access": [
                        {
                            "file_id": str(access.survey_file_id),
                            "access_type": access.access_type.value
                        }
                        for access in user.file_access
                    ]
                }
                users_data.append(user_data)
            
            return {
                "users": users_data,
                "total_users": len(users_data)
            }
        except Exception as e:
            raise Exception(f"Failed to get users with access: {str(e)}")

    @staticmethod
    async def get_surveys_and_files(db: AsyncSession) -> dict:
        """Get all surveys and files for access management (admin only)"""
        try:
            from sqlalchemy.orm import selectinload
            from app.models.models import Survey
            
            # Get all surveys with their files from database only
            stmt = select(Survey).options(selectinload(Survey.files))
            result = await db.execute(stmt)
            surveys = result.scalars().all()
            
            surveys_data = []
            
            for survey in surveys:
                # Only include surveys that exist in the database
                survey_files = []
                for file in survey.files:
                    survey_files.append({
                        "id": str(file.id),
                        "filename": file.filename,
                        "file_size": file.file_size,
                        "storage_path": file.storage_path,
                        "file_hash": file.file_hash,
                        "upload_date": file.upload_date.isoformat() if file.upload_date else None,
                        "created_at": file.created_at.isoformat() if file.created_at else None,
                        "updated_at": file.updated_at.isoformat() if file.updated_at else None
                    })
                
                survey_data = {
                    "id": str(survey.id),
                    "title": survey.title,
                    "category": survey.category or "general",
                    "description": survey.description,
                    "created_at": survey.created_at.isoformat() if survey.created_at else None,
                    "updated_at": survey.updated_at.isoformat() if survey.updated_at else None,
                    "total_files": len(survey_files),
                    "survey_files": survey_files  # Use survey_files as expected by frontend
                }
                surveys_data.append(survey_data)
            
            # Return in the format expected by frontend
            return {
                "surveys": surveys_data,
                "total_surveys": len(surveys_data),
                "total_files": sum(len(s["survey_files"]) for s in surveys_data)
            }
            
        except Exception as e:
            raise Exception(f"Failed to get surveys and files: {str(e)}")

# Create service instance
access_control_service = AccessControlService()