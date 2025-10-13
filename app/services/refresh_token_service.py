from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_, delete, func
import uuid
import bcrypt
import secrets

from app.models.models import RefreshToken, User, Role
from app.models.schemas import RefreshTokenRequest
from app.core.security import create_access_token, verify_password, hash_password
from app.core.config import settings
from app.services.base import BaseService


class RefreshTokenService(BaseService):
    """Refresh token management service"""
    
    @staticmethod
    async def generate_refresh_token(db: AsyncSession, user_id: str) -> str:
        """Generate a new refresh token for a user"""
        try:
            # Generate a secure random token
            token = secrets.token_urlsafe(32) + str(datetime.utcnow().timestamp())
            token_hash = hash_password(token)  # Use the same hashing as passwords
            
            # Set expiration to 7 days from now
            expires_at = datetime.utcnow() + timedelta(days=7)
            
            # Create refresh token
            user_uuid = uuid.UUID(user_id)
            refresh_token = RefreshToken(
                user_id=user_uuid,
                token=token_hash,
                expires_at=expires_at,
                is_revoked=False
            )
            
            db.add(refresh_token)
            await db.commit()
            
            return token
        except Exception as e:
            await db.rollback()
            raise e
    
    @staticmethod
    async def validate_refresh_token(
        db: AsyncSession, 
        token: str
    ) -> Optional[Dict[str, Any]]:
        """Validate a refresh token and return user info if valid"""
        try:
            # Get all valid (non-revoked, non-expired) refresh tokens with user info
            query = (
                select(RefreshToken, User, Role)
                .join(User, RefreshToken.user_id == User.id)
                .outerjoin(Role, User.roleid == Role.id)
                .where(
                    and_(
                        RefreshToken.expires_at > datetime.utcnow(),
                        RefreshToken.is_revoked == False
                    )
                )
            )
            
            result = await db.execute(query)
            tokens_with_users = result.fetchall()
            
            # Check each token hash to find a match
            matched_data = None
            for refresh_token, user, role in tokens_with_users:
                if verify_password(token, refresh_token.token):
                    matched_data = (refresh_token, user, role)
                    break
            
            if not matched_data:
                return None
            
            refresh_token, user, role = matched_data
            
            # Double check expiration
            if refresh_token.expires_at <= datetime.utcnow():
                # Delete expired token
                await db.delete(refresh_token)
                await db.commit()
                return None
            
            return {
                "user_id": str(user.id),
                "email": user.email,
                "role": role.role if role else "user",
                "welcome_popup_dismissed": user.welcome_popup_dismissed or False,
                "refresh_token_id": str(refresh_token.id)
            }
            
        except Exception as e:
            raise e
    
    @staticmethod
    async def revoke_refresh_token(db: AsyncSession, token: str) -> bool:
        """Revoke a specific refresh token"""
        try:
            # Get all refresh tokens
            query = select(RefreshToken)
            result = await db.execute(query)
            refresh_tokens = result.scalars().all()
            
            # Find matching token hash
            token_to_revoke = None
            for refresh_token in refresh_tokens:
                if verify_password(token, refresh_token.token):
                    token_to_revoke = refresh_token
                    break
            
            if not token_to_revoke:
                return False
            
            # Delete the token
            await db.delete(token_to_revoke)
            await db.commit()
            
            return True
            
        except Exception as e:
            await db.rollback()
            raise e
    
    @staticmethod
    async def revoke_all_user_tokens(db: AsyncSession, user_id: str) -> bool:
        """Revoke all refresh tokens for a user (logout all devices)"""
        try:
            user_uuid = uuid.UUID(user_id)
            
            # Delete all refresh tokens for the user
            delete_stmt = delete(RefreshToken).where(RefreshToken.user_id == user_uuid)
            await db.execute(delete_stmt)
            await db.commit()
            
            return True
            
        except Exception as e:
            await db.rollback()
            raise e
    
    @staticmethod
    async def cleanup_expired_tokens(db: AsyncSession) -> int:
        """Clean up expired refresh tokens"""
        try:
            # Count expired tokens before deletion
            count_query = select(func.count(RefreshToken.id)).where(
                RefreshToken.expires_at < datetime.utcnow()
            )
            count_result = await db.execute(count_query)
            expired_count = count_result.scalar()
            
            # Delete expired tokens
            delete_stmt = delete(RefreshToken).where(
                RefreshToken.expires_at < datetime.utcnow()
            )
            await db.execute(delete_stmt)
            await db.commit()
            
            return expired_count or 0
            
        except Exception as e:
            await db.rollback()
            raise e
    
    @staticmethod
    async def refresh_access_token(
        db: AsyncSession, 
        refresh_token: str
    ) -> Optional[Dict[str, Any]]:
        """Generate new access token from refresh token"""
        try:
            # Validate the refresh token
            user_info = await RefreshTokenService.validate_refresh_token(db, refresh_token)
            
            if not user_info:
                return None
            
            # Generate new access token (15 minutes expiry)
            access_token_data = {
                "sub": user_info["user_id"],
                "email": user_info["email"],
                "role": user_info["role"],
                "welcome_popup_dismissed": user_info["welcome_popup_dismissed"]
            }
            
            access_token = create_access_token(
                data=access_token_data,
                expires_delta=timedelta(minutes=15)
            )
            
            # Generate new refresh token (rotate for security)
            new_refresh_token = await RefreshTokenService.generate_refresh_token(
                db, user_info["user_id"]
            )
            
            # Revoke the old refresh token
            await RefreshTokenService.revoke_refresh_token(db, refresh_token)
            
            return {
                "accessToken": access_token,  # Frontend expects camelCase
                "refreshToken": new_refresh_token,  # Frontend expects camelCase
                "tokenType": "Bearer",  # Frontend expects camelCase
                "expiresIn": 15 * 60  # Frontend expects camelCase (15 minutes in seconds)
            }
            
        except Exception as e:
            raise e
    
    @staticmethod
    async def get_user_refresh_tokens_count(db: AsyncSession, user_id: str) -> int:
        """Get count of active refresh tokens for a user"""
        try:
            user_uuid = uuid.UUID(user_id)
            
            query = select(func.count(RefreshToken.id)).where(
                and_(
                    RefreshToken.user_id == user_uuid,
                    RefreshToken.expires_at > datetime.utcnow(),
                    RefreshToken.is_revoked == False
                )
            )
            
            result = await db.execute(query)
            return result.scalar() or 0
            
        except Exception as e:
            raise e
    
    @staticmethod
    async def get_refresh_token_info(
        db: AsyncSession, 
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get refresh token information for a user (for admin purposes)"""
        try:
            user_uuid = uuid.UUID(user_id)
            
            query = (
                select(RefreshToken)
                .where(RefreshToken.user_id == user_uuid)
                .where(RefreshToken.expires_at > datetime.utcnow())
                .where(RefreshToken.is_revoked == False)
                .order_by(RefreshToken.created_at.desc())
            )
            
            result = await db.execute(query)
            refresh_tokens = result.scalars().all()
            
            return [
                {
                    "id": str(token.id),
                    "created_at": token.created_at,
                    "expires_at": token.expires_at,
                    "is_revoked": token.is_revoked
                }
                for token in refresh_tokens
            ]
            
        except Exception as e:
            raise e