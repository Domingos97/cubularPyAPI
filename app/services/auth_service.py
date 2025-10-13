from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_
import uuid

from app.models.models import User, Role, RefreshToken
from app.models.schemas import UserCreate, UserUpdate
from app.core.security import (
    verify_password, 
    hash_password, 
    create_access_token, 
    create_refresh_token,
    generate_email_confirmation_token
)
from app.core.config import settings
from app.services.base import BaseService


class AuthService:
    """Authentication service"""
    
    @staticmethod
    async def authenticate_user(
        db: AsyncSession, 
        email: str, 
        password: str
    ) -> Optional[User]:
        """Authenticate user with email and password"""
        # Get user with role information
        query = (
            select(User)
            .options(selectinload(User.role))
            .where(User.email == email)
        )
        result = await db.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        if not verify_password(password, user.password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        await db.commit()
        
        return user
    
    @staticmethod
    async def create_user(
        db: AsyncSession,
        user_data: UserCreate
    ) -> User:
        """Create a new user"""
        # Check if user already exists
        existing_user = await AuthService.get_user_by_email(db, user_data.email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Get default user role
        role_query = select(Role).where(Role.role == "user")
        role_result = await db.execute(role_query)
        user_role = role_result.scalar_one_or_none()
        
        if not user_role:
            # Create default user role if it doesn't exist
            user_role = Role(role="user")
            db.add(user_role)
            await db.commit()
            await db.refresh(user_role)
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Generate email confirmation token
        user_id = str(uuid.uuid4())
        confirmation_token = generate_email_confirmation_token(user_id, user_data.email)
        
        # Create user
        db_user = User(
            id=uuid.UUID(user_id),
            email=user_data.email,
            username=user_data.username,
            password=hashed_password,
            language=user_data.language,
            roleid=user_role.id,
            email_confirmation_token=confirmation_token,
            email_confirmed=False
        )
        
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        
        # Load the role relationship
        await db.refresh(db_user, ["role"])
        
        return db_user
    
    @staticmethod
    async def get_user_by_email(
        db: AsyncSession,
        email: str
    ) -> Optional[User]:
        """Get user by email"""
        query = (
            select(User)
            .options(selectinload(User.role))
            .where(User.email == email)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_id(
        db: AsyncSession,
        user_id: uuid.UUID
    ) -> Optional[User]:
        """Get user by ID"""
        query = (
            select(User)
            .options(selectinload(User.role))
            .where(User.id == user_id)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def create_tokens(
        db: AsyncSession,
        user: User
    ) -> Dict[str, Any]:
        """Create access and refresh tokens for user"""
        # Create access token
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = create_access_token(
            data={
                "sub": str(user.id),
                "email": user.email,
                "role": user.role.role,
                "language": user.language,
                "welcome_popup_dismissed": user.welcome_popup_dismissed
            },
            expires_delta=access_token_expires
        )
        
        # Create refresh token
        refresh_token_value = create_refresh_token()
        refresh_token_expires = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
        
        # Store refresh token in database
        refresh_token = RefreshToken(
            user_id=user.id,
            token=refresh_token_value,
            expires_at=refresh_token_expires,
            is_revoked=False
        )
        
        db.add(refresh_token)
        await db.commit()
        
        return {
            "accessToken": access_token,  # Frontend expects camelCase
            "refreshToken": refresh_token_value,  # Frontend expects camelCase
            "tokenType": "Bearer",  # Frontend expects camelCase
            "expiresIn": 15 * 60  # Frontend expects camelCase (15 minutes in seconds)
        }
    
    @staticmethod
    async def refresh_access_token(
        db: AsyncSession,
        refresh_token: str
    ) -> Optional[Dict[str, Any]]:
        """Refresh access token using refresh token"""
        # Find valid refresh token
        query = (
            select(RefreshToken)
            .options(selectinload(RefreshToken.user).selectinload(User.role))
            .where(
                and_(
                    RefreshToken.token == refresh_token,
                    RefreshToken.expires_at > datetime.utcnow(),
                    RefreshToken.is_revoked == False
                )
            )
        )
        result = await db.execute(query)
        refresh_token_obj = result.scalar_one_or_none()
        
        if not refresh_token_obj:
            return None
        
        # Create new access token
        user = refresh_token_obj.user
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = create_access_token(
            data={
                "sub": str(user.id),
                "email": user.email,
                "role": user.role.role,
                "language": user.language
            },
            expires_delta=access_token_expires
        )
        
        return {
            "accessToken": access_token,  # Frontend expects camelCase
            "tokenType": "Bearer",  # Frontend expects camelCase
            "expiresIn": 15 * 60  # Frontend expects camelCase (15 minutes in seconds)
        }
    
    @staticmethod
    async def revoke_refresh_token(
        db: AsyncSession,
        refresh_token: str
    ) -> bool:
        """Revoke a refresh token"""
        query = select(RefreshToken).where(RefreshToken.token == refresh_token)
        result = await db.execute(query)
        refresh_token_obj = result.scalar_one_or_none()
        
        if not refresh_token_obj:
            return False
        
        refresh_token_obj.is_revoked = True
        await db.commit()
        return True
    
    @staticmethod
    async def revoke_all_user_tokens(
        db: AsyncSession,
        user_id: uuid.UUID
    ) -> int:
        """Revoke all refresh tokens for a user"""
        query = select(RefreshToken).where(
            and_(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False
            )
        )
        result = await db.execute(query)
        tokens = result.scalars().all()
        
        count = 0
        for token in tokens:
            token.is_revoked = True
            count += 1
        
        await db.commit()
        return count
    
    @staticmethod
    async def confirm_email(
        db: AsyncSession,
        token: str
    ) -> Optional[User]:
        """Confirm user email with token"""
        from app.core.security import verify_email_confirmation_token
        
        payload = verify_email_confirmation_token(token)
        if not payload:
            return None
        
        user_id = payload.get("user_id")
        email = payload.get("email")
        
        if not user_id or not email:
            return None
        
        # Find user
        user = await AuthService.get_user_by_id(db, uuid.UUID(user_id))
        if not user or user.email != email:
            return None
        
        # Confirm email
        user.email_confirmed = True
        user.email_confirmation_token = None
        await db.commit()
        
        return user
    
    @staticmethod
    async def update_user_password(
        db: AsyncSession,
        user_id: uuid.UUID,
        new_password: str
    ) -> Optional[User]:
        """Update user password"""
        user = await AuthService.get_user_by_id(db, user_id)
        if not user:
            return None
        
        user.password = hash_password(new_password)
        await db.commit()
        
        # Revoke all existing refresh tokens
        await AuthService.revoke_all_user_tokens(db, user_id)
        
        return user
    
    @staticmethod
    async def resend_confirmation_email(
        db: AsyncSession,
        email: str
    ) -> Optional[User]:
        """Resend email confirmation"""
        user = await AuthService.get_user_by_email(db, email)
        if not user:
            return None
        
        # Don't resend if already confirmed
        if user.email_confirmed:
            return user
        
        # Generate new confirmation token
        confirmation_token = generate_email_confirmation_token(user.id, user.email)
        user.email_confirmation_token = confirmation_token
        
        await db.commit()
        
        # Here you would typically send the email
        # For now, we'll just return the user
        return user
    
    @staticmethod
    async def cleanup_expired_tokens(db: AsyncSession) -> int:
        """Clean up expired refresh tokens"""
        query = select(RefreshToken).where(
            RefreshToken.expires_at <= datetime.utcnow()
        )
        result = await db.execute(query)
        expired_tokens = result.scalars().all()
        
        count = 0
        for token in expired_tokens:
            await db.delete(token)
            count += 1
        
        await db.commit()
        return count


class UserService(BaseService[User, UserCreate, UserUpdate]):
    """User service"""
    
    def __init__(self):
        super().__init__(User)
    
    async def update_user_profile(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        user_update: UserUpdate
    ) -> Optional[User]:
        """Update user profile"""
        user = await self.get_by_id(db, user_id, options=[selectinload(User.role)])
        if not user:
            return None
        
        update_data = user_update.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(user, field):
                setattr(user, field, value)
        
        await db.commit()
        await db.refresh(user)
        return user
    
    async def get_user_with_stats(
        self,
        db: AsyncSession,
        user_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Get user with statistics"""
        user = await self.get_by_id(
            db, 
            user_id, 
            options=[
                selectinload(User.role),
                selectinload(User.surveys),
                selectinload(User.chat_sessions)
            ]
        )
        
        if not user:
            return None
        
        return {
            "user": user,
            "stats": {
                "total_surveys": len(user.surveys),
                "total_chat_sessions": len(user.chat_sessions),
                "email_confirmed": user.email_confirmed,
                "last_login": user.last_login
            }
        }


# Create service instances
auth_service = AuthService()
user_service = UserService()