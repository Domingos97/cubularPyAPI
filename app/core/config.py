from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field
from typing import List, Optional, Union
import os


class Settings(BaseSettings):
    """Application settings for local development environment"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from environment
    )
    
    # Application
    app_name: str = "CubularPyAPI"
    app_version: str = "1.0.0"
    debug: bool = Field(default=True, alias="DEBUG")  # Default to development mode
    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")  # More verbose logging for local dev
    port: int = Field(default=8000, alias="PORT")  # Consistent port configuration
    
    # Database - optimized for asyncpg direct connections
    database_host: str = Field(default="localhost", alias="DATABASE_HOST")
    database_port: int = Field(default=5432, alias="DATABASE_PORT")
    database_name: str = Field(default="projectxy", alias="DATABASE_NAME")
    database_user: str = Field(default="postgres", alias="DATABASE_USER")
    database_password: str = Field(..., alias="DATABASE_PASSWORD")  # REQUIRED - no default
    
    # Lightweight database pool settings
    database_pool_size: int = Field(default=5, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=5, alias="DATABASE_MAX_OVERFLOW")
    
    @property
    def database_url(self) -> str:
        """Construct database URL from individual components"""
        return f"postgresql+asyncpg://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"
    
    # Security - BOTH REQUIRED
    secret_key: str = Field(
        ...,  # REQUIRED - no default value
        alias="JWT_SECRET",
        description="Secret key for JWT - MUST be provided via environment variable"
    )
    encryption_secret: str = Field(
        ...,  # REQUIRED - no default value
        alias="ENCRYPTION_SECRET",
        description="Secret key for API key encryption/decryption - MUST be provided via environment variable"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=1440, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, alias="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # CORS
    allowed_origins: Union[str, List[str]] = Field(default="", alias="ALLOWED_ORIGINS")
    
    @field_validator("allowed_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if v is None or v == "":
            return []
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return []
    
    # AI Services: models and keys now come from DB only; deprecated fields removed
    
    # Email/SMTP Configuration
    smtp_host: Optional[str] = Field(default=None, alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, alias="SMTP_USER")
    smtp_pass: Optional[str] = Field(default=None, alias="SMTP_PASS")
    smtp_from: Optional[str] = Field(default=None, alias="SMTP_FROM")
    smtp_tls: bool = Field(default=True, alias="SMTP_TLS")
    
    # Legacy email settings (for backward compatibility)
    smtp_username: Optional[str] = Field(default=None, alias="SMTP_USERNAME")
    smtp_password: Optional[str] = Field(default=None, alias="SMTP_PASSWORD")
    from_email: str = Field(default="noreply@projectxy.com", alias="FROM_EMAIL")
    
    @property
    def SMTP_HOST(self) -> Optional[str]:
        return self.smtp_host
    
    @property
    def SMTP_PORT(self) -> int:
        return self.smtp_port
    
    @property
    def SMTP_USER(self) -> Optional[str]:
        return self.smtp_user or self.smtp_username
    
    @property
    def SMTP_PASS(self) -> Optional[str]:
        return self.smtp_pass or self.smtp_password
    
    @property
    def SMTP_FROM(self) -> Optional[str]:
        return self.smtp_from or self.from_email
    
    @property
    def SMTP_TLS(self) -> bool:
        return self.smtp_tls
    
    # File Upload
    max_file_size: int = Field(default=10485760, alias="MAX_FILE_SIZE")  # 10MB
    upload_dir: str = Field(default="survey_data", alias="UPLOAD_DIR")
    
    # Vector Search (only for manual operations via API)
    default_similarity_threshold: float = 0.25
    max_search_results: int = 10000


# Create settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.upload_dir, exist_ok=True)