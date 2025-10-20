import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from typing import Optional
import secrets

from app.utils.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

# Encryption configuration to match TypeScript
ALGORITHM = 'AES'
KEY_LENGTH = 32  # 256 bits
IV_LENGTH = 16   # 128 bits
SALT_LENGTH = 64 # 512 bits
TAG_LENGTH = 16  # 128 bits
ITERATIONS = 100000  # PBKDF2 iterations
AAD = b'api-key'  # Additional authenticated data


class EncryptionService:
    """Service for encrypting and decrypting sensitive data like API keys - compatible with TypeScript"""
    
    def __init__(self):
        self.backend = default_backend()
    
    def _derive_key(self, secret: str, salt: bytes) -> bytes:
        """Derive encryption key using PBKDF2 (matches TypeScript implementation)"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=KEY_LENGTH,
            salt=salt,
            iterations=ITERATIONS,
            backend=self.backend
        )
        return kdf.derive(secret.encode('utf-8'))
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string using AES-256-GCM (matches TypeScript format)"""
        try:
            if not plaintext:
                raise ValueError("Cannot encrypt empty string")
            
            # Get encryption secret from settings
            encryption_secret = settings.encryption_secret
            if not encryption_secret:
                raise ValueError("ENCRYPTION_SECRET environment variable is required")
            
            # Generate random salt and IV for each encryption (like TypeScript)
            salt = secrets.token_bytes(SALT_LENGTH)
            iv = secrets.token_bytes(IV_LENGTH)
            
            # Derive key using PBKDF2 with the random salt
            key = self._derive_key(encryption_secret, salt)
            
            # Create cipher using AES-256-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Set additional authenticated data
            encryptor.authenticate_additional_data(AAD)
            
            # Encrypt the data
            ciphertext = encryptor.update(plaintext.encode('utf-8')) + encryptor.finalize()
            
            # Get authentication tag
            tag = encryptor.tag
            
            # Combine salt, iv, tag, and ciphertext (matches TypeScript format)
            encrypted = salt + iv + tag + ciphertext
            
            # Return as base64 string
            return base64.b64encode(encrypted).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise RuntimeError("Failed to encrypt data")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded encrypted data (compatible with TypeScript format)"""
        try:
            if not encrypted_data:
                return ""
            
            # Get encryption secret from settings
            encryption_secret = settings.encryption_secret
            if not encryption_secret:
                raise ValueError("ENCRYPTION_SECRET environment variable is required")
            
            # Decode from base64
            encrypted = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Extract components (matches TypeScript format)
            salt = encrypted[:SALT_LENGTH]
            iv = encrypted[SALT_LENGTH:SALT_LENGTH + IV_LENGTH]
            tag = encrypted[SALT_LENGTH + IV_LENGTH:SALT_LENGTH + IV_LENGTH + TAG_LENGTH]
            ciphertext = encrypted[SALT_LENGTH + IV_LENGTH + TAG_LENGTH:]
            
            # Derive the same key using the extracted salt
            key = self._derive_key(encryption_secret, salt)
            
            # Create decipher using AES-256-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            # Set additional authenticated data (must match encryption)
            decryptor.authenticate_additional_data(AAD)
            
            # Decrypt the data
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise RuntimeError("Failed to decrypt data")
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Specifically encrypt API keys"""
        if not api_key:
            return ""
        
        try:
            return self.encrypt(api_key)
        except Exception as e:
            logger.error(f"Failed to encrypt API key: {str(e)}")
            raise
    
    def decrypt_api_key(self, api_key: str) -> str:
        """Specifically decrypt API keys"""
        if not api_key:
            return ""
        
        try:
            return self.decrypt(api_key)
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {str(e)}")
            raise
    
    def test_encryption(self) -> bool:
        """Test that encryption/decryption is working correctly"""
        try:
            test_string = 'test-api-key-sk-1234567890abcdef'
            encrypted = self.encrypt(test_string)
            decrypted = self.decrypt(encrypted)
            return test_string == decrypted
        except Exception as e:
            logger.error(f"Encryption test failed: {str(e)}")
            return False

    def generate_encryption_secret(self) -> str:
        """Generate a secure random encryption secret (matches TypeScript format)"""
        return secrets.token_bytes(64).hex()

    def generate_encryption_salt(self) -> str:
        """Generate a secure random salt for key derivation"""
        return secrets.token_bytes(32).hex()
    
    def is_encrypted(self, data: str) -> bool:
        """Check if data appears to be encrypted (basic check)"""
        try:
            if not data:
                return False
            
            # Try to decode as base64 - encrypted data should be base64 encoded
            base64.urlsafe_b64decode(data.encode())
            return True
        except:
            return False
    
    def get_encryption_key(self) -> str:
        """Get the current encryption key (for backup/migration purposes)"""
        return self._key


class APIKeyManager:
    """Higher-level API key management with encryption"""
    
    def __init__(self):
        self.encryption_service = EncryptionService()
    
    def store_api_key(self, api_key: str) -> str:
        """Store an API key (encrypt it)"""
        if not api_key:
            return ""
        
        try:
            # Check if already encrypted
            if self.encryption_service.is_encrypted(api_key):
                return api_key
            
            return self.encryption_service.encrypt_api_key(api_key)
            
        except Exception as e:
            logger.error(f"Failed to store API key: {str(e)}")
            raise
    
    def retrieve_api_key(self, api_key: str) -> str:
        """Retrieve and decrypt an API key"""
        if not api_key:
            return ""
        
        try:
            # Check if it's actually encrypted
            if not self.encryption_service.is_encrypted(api_key):
                # If not encrypted, return as-is (for backward compatibility)
                return api_key
            
            return self.encryption_service.decrypt_api_key(api_key)
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key: {str(e)}")
            raise
    
    def mask_api_key(self, api_key: str) -> str:
        """Mask an API key for display purposes"""
        if not api_key:
            return ""
        
        if len(api_key) <= 8:
            return "*" * len(api_key)
        
        # Show first 4 and last 4 characters
        return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
    
    def validate_api_key_format(self, api_key: str, provider: str) -> bool:
        """Basic validation of API key format based on provider"""
        if not api_key:
            return False
        
        provider = provider.lower()
        
        # Basic format validation (extend as needed)
        if provider == "openai":
            return api_key.startswith("sk-") and len(api_key) > 20
        elif provider == "anthropic":
            return api_key.startswith("sk-ant-") and len(api_key) > 20
        elif provider == "google":
            return len(api_key) > 20  # Google keys vary in format
        elif provider == "azure":
            return len(api_key) > 20  # Azure keys vary in format
        elif provider == "openrouter":
            return api_key.startswith("sk-or-") and len(api_key) > 20
        else:
            # Generic validation
            return len(api_key) > 10


# Create singleton instances
encryption_service = EncryptionService()
api_key_manager = APIKeyManager()