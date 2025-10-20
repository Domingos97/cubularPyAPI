"""
Lightweight Auth Service - Direct Database Access
==============================================
Replaces SQLAlchemy ORM auth_service with direct SQL queries using lightweight_db_service.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import uuid

from app.services.lightweight_db_service import LightweightDBService
from app.models.schemas import UserCreate, UserUpdate
from app.core.security import (
    verify_password, 
    hash_password, 
    create_access_token, 
    create_refresh_token,
    generate_email_confirmation_token
)
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LightweightAuthService:
    """Lightweight authentication service using direct SQL queries"""
    
    @staticmethod
    async def authenticate_user(
        db: LightweightDBService, 
        email: str, 
        password: str
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user with email and password"""
        user = await db.get_user_by_email(email)
        
        if not user:
            return None
        
        if not verify_password(password, user["password"]):
            return None
        
        # Update last login
        await db.update_user_last_login(str(user["id"]))
        
        # Update user dict with latest login time
        user["last_login"] = datetime.utcnow()
        
        # Ensure id is a string for JSON serialization
        user["id"] = str(user["id"])
        
        return user
    
    @staticmethod
    async def create_user(
        db: LightweightDBService,
        user_data: UserCreate
    ) -> Dict[str, Any]:
        """Create a new user and send confirmation email"""
        # Check if user already exists
        existing_user = await db.get_user_by_email(user_data.email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Create user
        user = await db.create_user(
            email=user_data.email,
            username=user_data.username,
            hashed_password=hashed_password,
            language=user_data.language or "en-US"
        )
        
        if not user:
            raise ValueError("Failed to create user")
        
        logger.info(f"Created new user: {user['email']}")
        
        # Send confirmation email
        try:
            from app.services.email_service import EmailService
            from app.core.config import settings
            
            # Generate confirmation token
            confirmation_token = generate_email_confirmation_token(
                str(user["id"]), 
                user["email"]
            )
            
            # Create confirmation link - point to API endpoint which will handle confirmation and redirect
            api_base_url = "https://cubularpyfront-production.up.railway.app"  # API server URL
            confirmation_link = f"{api_base_url}/api/auth/confirm-email?token={confirmation_token}"
            
            # Send email
            await EmailService.send_confirmation_email(
                to=user["email"],
                username=user["username"],
                confirmation_link=confirmation_link
            )
            
            logger.info(f"Confirmation email sent to: {user['email']}")
            
        except Exception as e:
            logger.warning(f"Failed to send confirmation email to {user['email']}: {e}")
            # Don't fail registration if email sending fails
        
        return user
    
    @staticmethod
    async def create_tokens(
        db: LightweightDBService, 
        user: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create access and refresh tokens for user"""
        # Ensure user_id is a string (convert UUID if needed)
        user_id = str(user["id"]) if user["id"] else None
        
        # Create access token with role information
        access_token = create_access_token(
            data={
                "sub": user_id, 
                "email": user["email"],
                "role": user.get("role_name", "user"),  # Include role in JWT
                "language": user.get("language", "en-US"),
                "welcome_popup_dismissed": user.get("welcome_popup_dismissed", False),
                # Include AI personalities access flag so frontend can decide to show UI
                "has_ai_personalities_access": bool(user.get("has_ai_personalities_access", False))
            },
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        
        # Create refresh token
        refresh_token = create_refresh_token()
        
        # Store refresh token in database
        expires_at = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
        await db.create_refresh_token(user_id, refresh_token, expires_at)
        
        return {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "tokenType": "bearer"
        }
    
    @staticmethod
    async def refresh_access_token(
        db: LightweightDBService,
        refresh_token: str
    ) -> Optional[Dict[str, str]]:
        """Refresh access token using refresh token"""
        # Validate refresh token
        token_data = await db.get_refresh_token(refresh_token)
        
        if not token_data:
            return None
        
        # Check if token is expired
        if token_data["expires_at"] < datetime.utcnow():
            await db.revoke_refresh_token(refresh_token)
            return None
        
        # Create new access token with role information
        access_token = create_access_token(
            data={
                "sub": str(token_data["user_id"]), 
                "email": token_data["email"],
                "role": token_data.get("role_name", "user"),  # Include role in JWT
                "language": token_data.get("language", "en-US"),
                "welcome_popup_dismissed": token_data.get("welcome_popup_dismissed", False),
                # Propagate AI personalities access flag from the user record
                "has_ai_personalities_access": bool(token_data.get("has_ai_personalities_access", False))
            },
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        
        return {
            "accessToken": access_token,
            "tokenType": "bearer"
        }
    
    @staticmethod
    async def revoke_refresh_token(
        db: LightweightDBService,
        refresh_token: str
    ) -> bool:
        """Revoke a specific refresh token"""
        return await db.revoke_refresh_token(refresh_token)
    
    @staticmethod
    async def revoke_all_user_tokens(
        db: LightweightDBService,
        user_id: str
    ) -> int:
        """Revoke all refresh tokens for a user"""
        return await db.revoke_all_user_tokens(user_id)
    
    @staticmethod
    async def confirm_email(
        db: LightweightDBService,
        token: str
    ) -> Optional[Dict[str, Any]]:
        """Confirm user email with token"""
        try:
            from app.core.security import verify_email_confirmation_token
            
            # Decode and verify the confirmation token
            token_data = verify_email_confirmation_token(token)
            if not token_data:
                logger.warning("Invalid or expired confirmation token")
                return None
            
            user_id = token_data.get("user_id")
            if not user_id:
                logger.warning("No user_id in confirmation token")
                return None
            
            # Confirm the email in database
            success = await db.confirm_email(user_id)
            if success:
                return await db.get_user_by_id(user_id)
            return None
            
        except Exception as e:
            logger.error(f"Email confirmation failed: {e}")
            return None
    
    @staticmethod
    async def resend_confirmation_email(
        db: LightweightDBService,
        email: str
    ) -> Optional[Dict[str, Any]]:
        """Resend confirmation email"""
        user = await db.get_user_by_email(email)
        if not user:
            return None
        
        if user["email_confirmed"]:
            raise ValueError("Email already confirmed")
        
        # Send confirmation email
        try:
            from app.services.email_service import EmailService
            from app.core.config import settings
            
            # Generate confirmation token
            confirmation_token = generate_email_confirmation_token(
                str(user["id"]), 
                user["email"]
            )
            
            # Create confirmation link - point to API endpoint which will handle confirmation and redirect
            api_base_url = "https://cubularpyfront-production.up.railway.app"  # API server URL
            confirmation_link = f"{api_base_url}/api/auth/confirm-email?token={confirmation_token}"
            
            # Send email
            await EmailService.send_confirmation_email(
                to=user["email"],
                username=user["username"],
                confirmation_link=confirmation_link
            )
            
            logger.info(f"Confirmation email resent to: {user['email']}")
            
        except Exception as e:
            logger.warning(f"Failed to resend confirmation email to {user['email']}: {e}")
            raise ValueError("Failed to send confirmation email")
        
        return user
    
    @staticmethod
    async def get_user_by_id(
        db: LightweightDBService,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        return await db.get_user_by_id(user_id)
    
    @staticmethod
    async def get_user_by_email(
        db: LightweightDBService,
        email: str
    ) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        return await db.get_user_by_email(email)
    
    @staticmethod
    async def update_user_password(
        db: LightweightDBService,
        user_id: str,
        new_password: str,
        current_password: str = None
    ) -> Optional[Dict[str, Any]]:
        """Update user password"""
        user = await db.get_user_by_id(user_id)
        if not user:
            return None
        
        # Verify current password if provided
        if current_password and not verify_password(current_password, user["password"]):
            raise ValueError("Current password is incorrect")
        
        # Hash new password
        hashed_password = hash_password(new_password)
        
        # Update password
        success = await db.update_user_password(user_id, hashed_password)
        if success:
            return await db.get_user_by_id(user_id)
        
        return None


# Global instance
# Create service instance - main auth service
auth_service = LightweightAuthService()