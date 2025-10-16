from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import bcrypt
from app.core.config import settings
import secrets
import string

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        # Check for empty inputs
        if not plain_password or not hashed_password:
            return False
            
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


def hash_password(password: str) -> str:
    """Hash a password"""
    try:
        # Ensure password is not empty
        if not password or not password.strip():
            raise ValueError("Password cannot be empty")
            
        # Use bcrypt with salt rounds 10 (same as TypeScript API)
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=10))
        
        return hashed.decode('utf-8')
    except Exception as e:
        # Log the error and re-raise instead of returning empty string
        print(f"Password hashing error: {e}")
        raise ValueError(f"Password hashing failed: {e}")


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt


def create_refresh_token() -> str:
    """Create a secure refresh token"""
    # Generate a secure random token
    alphabet = string.ascii_letters + string.digits
    token = ''.join(secrets.choice(alphabet) for _ in range(32))
    return token


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode a JWT token"""
    try:
        from app.utils.logging import get_logger
        logger = get_logger(__name__)
        
        logger.info(f"Verifying token: {token[:20]}...")
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        logger.info(f"Token verification successful, user_id: {payload.get('sub')}")
        return payload
    except JWTError as e:
        from app.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.error(f"JWT verification failed: {e}")
        return None


def decode_token_payload(token: str) -> Optional[Dict[str, Any]]:
    """Decode token payload without verification (for expired tokens)"""
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False}
        )
        return payload
    except JWTError:
        return None


def generate_email_confirmation_token(user_id: str, email: str) -> str:
    """Generate email confirmation token"""
    data = {
        "user_id": user_id,
        "email": email,
        "type": "email_confirmation",
        "exp": datetime.utcnow() + timedelta(hours=24)  # 24 hour expiry
    }
    return jwt.encode(data, settings.secret_key, algorithm=settings.jwt_algorithm)


def verify_email_confirmation_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify email confirmation token"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "email_confirmation":
            return None
        return payload
    except JWTError:
        return None


def generate_password_reset_token(user_id: str, email: str) -> str:
    """Generate password reset token"""
    data = {
        "user_id": user_id,
        "email": email,
        "type": "password_reset",
        "exp": datetime.utcnow() + timedelta(hours=1)  # 1 hour expiry
    }
    return jwt.encode(data, settings.secret_key, algorithm=settings.jwt_algorithm)


def verify_password_reset_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify password reset token"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "password_reset":
            return None
        return payload
    except JWTError:
        return None


class SecurityUtils:
    """Security utility class"""
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a cryptographically secure random token"""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def is_strong_password(password: str) -> bool:
        """Check if password meets security requirements"""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        return has_upper and has_lower and has_digit
    
    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """Basic input sanitization"""
        if not input_str:
            return ""
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '*']
        sanitized = input_str
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()