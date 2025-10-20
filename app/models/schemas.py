# Pydantic schemas for API request/response models (DTOs)
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import json

# Enums for API
class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class NotificationType(str, Enum):
    SURVEY_REQUEST = "survey_request"
    FEATURE_REQUEST = "feature_request"
    SUPPORT_REQUEST = "support_request"
    FEEDBACK = "feedback"
    OTHER = "other"

class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warn"
    ERROR = "error"
    CRITICAL = "critical"

# Base schema for all Pydantic models
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True
        use_enum_values = True

# User schemas - Match TypeScript exactly
class User(BaseSchema):
    id: str
    email: str
    username: Optional[str] = None
    avatar: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    preferred_personality: Optional[str] = None
    language_preference: Optional[str] = None
    role_id: Optional[str] = None
    role: Optional[str] = None
    role_details: Optional[Dict[str, str]] = None
    personality_details: Optional[Dict[str, Any]] = None
    welcome_popup_dismissed: Optional[bool] = None
    has_ai_personalities_access: Optional[bool] = None

# Extended user schema with access permissions for frontend
class UserWithAccess(User):
    user_survey_access: Optional[List['UserSurveyAccessWithDetails']] = None
    user_survey_file_access: Optional[List['UserSurveyFileAccessWithDetails']] = None
    user_plans: Optional[List['UserPlanWithDetails']] = None

class UserBase(BaseSchema):
    email: EmailStr
    username: str
    language: str = "en"

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserUpdate(BaseSchema):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    language: Optional[str] = None
    preferred_personality: Optional[str] = None
    avatar: Optional[str] = None
    welcome_popup_dismissed: Optional[bool] = None
    has_ai_personalities_access: Optional[bool] = None

# Keep UserResponse as alias for backward compatibility
UserResponse = User

class UserInDB(User):
    password: str

# Authentication schemas
class LoginRequest(BaseSchema):
    email: EmailStr
    password: str

class Token(BaseSchema):
    accessToken: str
    refreshToken: str
    tokenType: str = "Bearer"
    expiresIn: int
    
    class Config:
        from_attributes = True
        use_enum_values = True

class LoginResponse(BaseSchema):
    accessToken: str
    refreshToken: str
    tokenType: str = "Bearer"
    expiresIn: int  # Expiration time in seconds
    user: Optional[dict] = None  # Flexible user data
    
    class Config:
        from_attributes = True
        use_enum_values = True

class TokenPayload(BaseSchema):
    sub: Optional[str] = None
    exp: Optional[int] = None

class RefreshTokenRequest(BaseSchema):
    refresh_token: str = Field(alias="refreshToken")
    
    class Config:
        populate_by_name = True  # Allow both snake_case and camelCase

class ResendConfirmationRequest(BaseSchema):
    email: EmailStr

# AI Personality schemas - Match TypeScript exactly
class AIPersonality(BaseSchema):
    id: str
    name: str
    description: str
    detailed_analysis_prompt: str
    suggestions_prompt: str
    model_override: Optional[str] = None
    temperature_override: Optional[float] = None
    is_default: bool
    is_active: bool
    created_at: str
    updated_at: str

class AIPersonalityCreate(BaseSchema):
    name: str
    description: str
    detailed_analysis_prompt: str
    suggestions_prompt: str
    system_prompt: Optional[str] = None
    model_override: Optional[str] = None
    temperature_override: Optional[float] = None

class AIPersonalityUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None
    system_prompt: Optional[str] = None
    detailed_analysis_prompt: Optional[str] = None
    suggestions_prompt: Optional[str] = None
    model_override: Optional[str] = None
    temperature_override: Optional[float] = None

# Keep AIPersonalityResponse as alias for backward compatibility
AIPersonalityResponse = AIPersonality

# Survey schemas - Match TypeScript exactly
class SurveyFile(BaseSchema):
    id: str
    survey_id: str
    filename: str
    storage_path: str
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    upload_date: str
    created_at: str
    updated_at: str

class Survey(BaseSchema):
    id: str
    fileid: Optional[str] = None  # For backward compatibility
    title: Optional[str] = None
    filename: Optional[str] = None  # For backward compatibility
    createdat: Optional[str] = None  # For backward compatibility
    created_at: Optional[str] = None
    storage_path: Optional[str] = None  # For backward compatibility
    primary_language: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    ai_suggestions: Optional[List[str]] = None
    number_participants: Optional[int] = None
    total_files: Optional[int] = None
    files: Optional[List[SurveyFile]] = None

class SurveyBase(BaseSchema):
    title: str
    description: Optional[str] = None

class SurveyCreate(SurveyBase):
    category: Optional[str] = None
    primary_language: Optional[str] = None
    number_participants: Optional[int] = None
    ai_suggestions: Optional[List[str]] = None

class SurveyUpdate(BaseSchema):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    primary_language: Optional[str] = None
    number_participants: Optional[int] = None
    ai_suggestions: Optional[List[str]] = None
    is_active: Optional[bool] = None

# Keep aliases for backward compatibility
SurveyResponse = Survey
SurveyFileResponse = SurveyFile

# Chat schemas
class ChatSessionBase(BaseSchema):
    title: Optional[str] = "New Chat"

class ChatSessionCreate(ChatSessionBase):
    survey_ids: Optional[List[uuid.UUID]] = None

class ChatSessionCreateOptimized(BaseSchema):
    """Schema for optimized chat session creation to match TypeScript API"""
    surveyId: uuid.UUID
    fileIds: List[str]
    title: Optional[str] = None
    personalityId: Optional[uuid.UUID] = None

class ChatSessionUpdate(BaseSchema):
    title: Optional[str] = None
    selected_file_ids: Optional[List[str]] = None

class ChatMessageBase(BaseSchema):
    content: str
    role: MessageRole = MessageRole.USER

class ChatMessageCreate(ChatMessageBase):
    analytics_metadata: Optional[Dict[str, Any]] = None

class ChatMessageResponse(ChatMessageBase):
    id: uuid.UUID
    session_id: uuid.UUID
    sender: str  # 'user' or 'assistant'
    timestamp: datetime
    data_snapshot: Optional[Dict[str, Any]] = None
    confidence: Optional[Dict[str, Any]] = None
    personality_used: Optional[uuid.UUID] = None
    analytics_metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None
    created_at: datetime

class ChatSessionResponse(ChatSessionBase):
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    survey_ids: Optional[List[uuid.UUID]] = []
    category: Optional[str] = None
    personality_id: Optional[uuid.UUID] = None
    selected_file_ids: List[str] = []  # Changed from UUID to str to match database
    messages: List[ChatMessageResponse] = []

class ChatSessionCreateResponse(BaseSchema):
    """Response for chat session creation endpoints"""
    message: str
    session: Dict[str, Any]

# Search schemas
class SearchRequest(BaseSchema):
    query: str
    survey_ids: List[uuid.UUID]
    threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    max_results: int = Field(default=1000, ge=1, le=10000)

class SearchResult(BaseSchema):
    survey_id: uuid.UUID
    response_index: int
    similarity: float
    text: str
    analytics_metadata: Dict[str, Any] = {}

class SearchResponse(BaseSchema):
    results: List[SearchResult]
    total_found: int
    search_params: Dict[str, Any]
    query_embedding: Optional[List[float]] = None

# Chat completion schemas
class ChatCompletionRequest(BaseSchema):
    session_id: uuid.UUID
    message: str
    include_search: bool = True
    search_params: Optional[SearchRequest] = None

class ChatCompletionResponse(BaseSchema):
    message: ChatMessageResponse
    search_results: Optional[SearchResponse] = None
    ai_response: ChatMessageResponse

# Notification schemas
class NotificationBase(BaseSchema):
    type: str = Field(default="info")
    title: str = Field(..., max_length=200)
    message: str = Field(..., max_length=1000)
    priority: int = Field(default=1, ge=1, le=10)

class NotificationCreate(NotificationBase):
    user_id: uuid.UUID

class NotificationUpdate(BaseSchema):
    is_read: Optional[bool] = None
    status: Optional[str] = None
    admin_response: Optional[str] = None
    read_at: Optional[datetime] = None

class Notification(NotificationBase):
    id: uuid.UUID
    user_id: uuid.UUID
    is_read: bool = False
    status: str = "pending"
    admin_response: Optional[str] = None
    responded_by: Optional[uuid.UUID] = None
    responded_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# File upload schemas
class FileUploadResponse(BaseSchema):
    filename: str
    file_size: int
    upload_status: str
    processing_status: str
    message: str

# Log schemas
class LogCreate(BaseSchema):
    level: LogLevel
    action: str = Field(..., max_length=500)
    user_id: Optional[uuid.UUID] = None
    resource: Optional[str] = Field(None, max_length=255)
    resource_id: Optional[uuid.UUID] = None
    details: Optional[Dict[str, Any]] = None  # Changed from analytics_metadata
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: Optional[datetime] = None
    method: Optional[str] = Field(None, max_length=50)
    endpoint: Optional[str] = Field(None, max_length=255)
    status_code: Optional[int] = None
    session_id: Optional[uuid.UUID] = None
    request_body: Optional[Dict[str, Any]] = None
    response_body: Optional[Dict[str, Any]] = None
    response_time: Optional[int] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    api_key_used: Optional[str] = Field(None, max_length=255)
    provider: Optional[str] = Field(None, max_length=50)
    model: Optional[str] = Field(None, max_length=100)
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    priority: Optional[str] = Field(None, max_length=50)
    # Keep analytics_metadata for backward compatibility
    analytics_metadata: Optional[Dict[str, Any]] = Field(None, alias="details")

class LogResponse(LogCreate):
    id: uuid.UUID
    created_at: datetime

# Health check schema
class HealthCheck(BaseSchema):
    status: str
    version: str
    timestamp: datetime
    database: str = "connected"
    services: Dict[str, str] = {}

# Error schemas
class ErrorResponse(BaseSchema):
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ValidationErrorResponse(BaseSchema):
    detail: List[Dict[str, Any]]
    error_code: str = "validation_error"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Success schemas
class SuccessResponse(BaseSchema):
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Semantic chat schemas
class SemanticChatRequest(BaseSchema):
    question: str
    surveyIds: List[uuid.UUID]
    personalityId: Optional[str] = None
    sessionId: Optional[uuid.UUID] = None
    createSession: bool = True
    selectedFileIds: Optional[List[str]] = []  # Frontend compatibility

class SemanticChatResponse(BaseSchema):
    sessionId: uuid.UUID
    question: str
    answer: str
    conversationalResponse: Optional[str] = None  # Frontend compatibility  
    dataSnapshot: Optional[Dict[str, Any]] = None  # Frontend compatibility
    search_results: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# Enhanced Semantic Chat schemas (matching TypeScript API)
class EnhancedSemanticChatRequest(BaseSchema):
    question: str
    surveyIds: List[str]  # String UUIDs to match TypeScript
    personalityId: Optional[str] = None
    sessionId: Optional[str] = None  # String UUID to match TypeScript
    createSession: bool = True
    selectedFileIds: Optional[List[str]] = []

class SearchResultItem(BaseSchema):
    text: str
    value: float
    survey_id: Optional[str] = None
    index: Optional[int] = None

class EnhancedSemanticChatResponse(BaseSchema):
    sessionId: Optional[str] = None
    question: str
    conversationalResponse: str
    dataSnapshot: Dict[str, Any] = {}
    searchResults: List[SearchResultItem] = []
    metadata: Dict[str, Any] = {}
    success: bool = True
    error: Optional[str] = None

# Survey Suggestion schemas
class GenerateSuggestionsRequest(BaseSchema):
    description: Optional[str] = None
    category: Optional[str] = None
    personalityId: Optional[str] = None
    fileContent: Optional[Dict[str, Any]] = None

class SurveyIdSuggestionsRequest(BaseSchema):
    personalityId: Optional[str] = None
    fileContent: Optional[Dict[str, Any]] = None

class SimpleSuggestionsRequest(BaseSchema):
    personalityId: Optional[str] = None

class SuggestionsResponse(BaseSchema):
    suggestions: List[str]
    category: Optional[str] = None
    surveyId: Optional[str] = None
    method: Optional[str] = None

# Module Configuration schemas
class ModuleConfigurationBase(BaseSchema):
    module_name: str
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None

class ModuleConfigurationCreate(ModuleConfigurationBase):
    llm_setting_id: uuid.UUID
    ai_personality_id: Optional[uuid.UUID] = None

class ModuleConfigurationUpdate(BaseSchema):
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    active: Optional[bool] = None
    ai_personality_id: Optional[uuid.UUID] = None

# Nested response models for module configuration
class LLMSettingNestedResponse(BaseSchema):
    id: str
    provider: str
    api_key_configured: bool

class AIPersonalityNestedResponse(BaseSchema):
    name: str
    detailed_analysis_prompt: str

class ModuleConfigurationResponse(ModuleConfigurationBase):
    id: uuid.UUID
    llm_setting_id: uuid.UUID
    active: bool
    created_by: Optional[uuid.UUID]
    ai_personality_id: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime
    # Nested objects for frontend compatibility
    llm_settings: Optional[LLMSettingNestedResponse] = None
    ai_personality: Optional[AIPersonalityNestedResponse] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            uuid.UUID: str,
            datetime: lambda v: v.isoformat()
        }

# Language schemas
class SupportedLanguageBase(BaseSchema):
    code: str = Field(..., description="ISO 639-1 language code (e.g., 'en', 'es')")
    name: str = Field(..., description="English name of the language")
    native_name: str = Field(..., description="Native name of the language")
    enabled: bool = True
    is_rtl: bool = False
    sort_order: int = 0

class SupportedLanguageCreate(SupportedLanguageBase):
    pass

class SupportedLanguageUpdate(BaseSchema):
    name: Optional[str] = None
    native_name: Optional[str] = None
    enabled: Optional[bool] = None
    is_rtl: Optional[bool] = None
    sort_order: Optional[int] = None

class SupportedLanguageResponse(SupportedLanguageBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

# Language detection and config schemas
class LanguageDetectionResult(BaseSchema):
    language: str = Field(..., description="Detected language code")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")

class LanguageConfigResponse(BaseSchema):
    default_language: str
    available_languages: List[str]
    languages: List[SupportedLanguageResponse]

class UserLanguagePreference(BaseSchema):
    language_code: str = Field(..., description="User's preferred language code")

# LLM Setting schemas
class LLMSettingBase(BaseSchema):
    provider: str

class LLMSettingCreate(LLMSettingBase):
    api_key: Optional[str] = None  # Made optional for upsert operations
    active: Optional[bool] = None  # Added to support frontend

class LLMSettingUpdate(BaseSchema):
    api_key: Optional[str] = None
    active: Optional[bool] = None

class LLMSettingResponse(LLMSettingBase):
    id: uuid.UUID
    active: bool
    api_key: Optional[str] = None  # Masked API key for frontend compatibility
    created_by: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime

# Vector Search schemas REMOVED - Using fast_search_service with direct file access
# class VectorSearchRequest(BaseSchema):
#     query: str
#     survey_ids: Optional[List[str]] = None
#     top_k: int = Field(default=10, ge=1, le=100)
#     similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)

# class VectorSearchResponse(BaseSchema):
#     survey_id: str
#     file_id: str
#     text: str
#     similarity_score: float
#     analytics_metadata: Dict[str, Any]

class SurveyInsightsRequest(BaseSchema):
    analysis_type: str = "themes"

class ThemeInsight(BaseSchema):
    theme_id: int
    size: int
    percentage: float
    representative_text: str
    sample_texts: List[str]

class InsightStatistics(BaseSchema):
    avg_response_length: float
    median_response_length: float
    min_response_length: int
    max_response_length: int

class SurveyInsightsResponse(BaseSchema):
    total_responses: int
    analysis_type: str
    survey_id: str
    themes: Optional[List[ThemeInsight]] = None
    statistics: InsightStatistics

# Survey Statistics schemas
class SurveyFileStats(BaseSchema):
    id: uuid.UUID
    filename: str
    upload_status: str
    processing_status: str
    total_responses: Optional[int] = None
    processed_responses: Optional[int] = None
    error_message: Optional[str] = None

class SurveyStatistics(BaseSchema):
    survey_id: uuid.UUID
    title: str
    created_at: datetime
    total_files: int
    total_responses: int
    processed_responses: int
    files: List[SurveyFileStats]

# Access Control schemas
class AccessType(str, Enum):
    READ = "read"
    EDIT = "edit"
    ADMIN = "admin"

class UserSurveyAccessBase(BaseSchema):
    user_id: uuid.UUID
    survey_id: uuid.UUID
    access_type: AccessType

class UserSurveyAccessCreate(BaseSchema):
    user_id: uuid.UUID
    access_type: AccessType

class UserSurveyAccessUpdate(BaseSchema):
    access_type: Optional[AccessType] = None

class UserSurveyAccess(UserSurveyAccessBase):
    id: uuid.UUID
    granted_by: uuid.UUID
    granted_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

class UserSurveyFileAccessBase(BaseSchema):
    user_id: uuid.UUID
    survey_file_id: uuid.UUID
    access_type: AccessType

class UserSurveyFileAccessCreate(BaseSchema):
    user_id: uuid.UUID
    access_type: AccessType

class UserSurveyFileAccessUpdate(BaseSchema):
    access_type: Optional[AccessType] = None

class UserSurveyFileAccess(UserSurveyFileAccessBase):
    id: uuid.UUID
    granted_by: uuid.UUID
    granted_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True

# Extended access schemas with related data for frontend
class SurveyDetails(BaseSchema):
    id: str
    title: str
    category: Optional[str] = None

class SurveyStatsSimple(BaseSchema):
    totalFiles: int
    totalSize: int
    fileTypes: Dict[str, int]
    lastUpdated: Optional[str] = None

class SurveyWithFilesResponse(BaseSchema):
    id: str
    fileid: Optional[str] = None  # For backward compatibility
    title: Optional[str] = None
    filename: Optional[str] = None  # For backward compatibility
    createdat: Optional[str] = None  # For backward compatibility
    created_at: Optional[str] = None
    storage_path: Optional[str] = None  # For backward compatibility
    primary_language: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    ai_suggestions: Optional[List[str]] = None
    number_participants: Optional[int] = None
    total_files: Optional[int] = None
    files: Optional[List[SurveyFile]] = None
    stats: SurveyStatsSimple

class FileDetails(BaseSchema):
    id: str
    filename: str
    surveys: SurveyDetails

class UserSurveyAccessWithDetails(BaseSchema):
    id: str
    survey_id: str
    access_type: str
    granted_at: str
    expires_at: Optional[str] = None
    is_active: bool
    surveys: SurveyDetails

class UserSurveyFileAccessWithDetails(BaseSchema):
    id: str
    survey_file_id: str
    access_type: str
    granted_at: str
    expires_at: Optional[str] = None
    is_active: bool
    survey_files: FileDetails

class AccessGrant(BaseSchema):
    user_id: uuid.UUID
    access_type: AccessType

class BulkAccessGrant(BaseSchema):
    access_grants: List[AccessGrant]

# Plan schemas
class PlanBase(BaseSchema):
    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency: str = "USD"
    billing: Optional[str] = None
    features: Optional[List[str]] = None  # List of feature strings
    max_surveys: Optional[int] = None
    max_responses: Optional[int] = None
    priority_support: bool = False
    api_access: bool = False

class PlanCreate(PlanBase):
    features: Optional[List[str]] = None  # List of feature strings

class PlanUpdate(BaseSchema):
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    billing: Optional[str] = None
    features: Optional[List[str]] = None
    max_surveys: Optional[int] = None
    max_responses: Optional[int] = None
    priority_support: Optional[bool] = None
    api_access: Optional[bool] = None
    is_active: Optional[bool] = None

class Plan(PlanBase):
    id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    @validator('features', pre=True)
    def parse_features(cls, v):
        """Parse features from JSONB."""
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except:
                return []
        return []

class UserPlanBase(BaseSchema):
    user_id: uuid.UUID
    plan_id: uuid.UUID

class UserPlanCreate(UserPlanBase):
    status: str = "active"
    end_date: Optional[datetime] = None
    trial_ends_at: Optional[datetime] = None
    auto_renew: bool = True
    payment_method_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None

class UserPlan(UserPlanBase):
    id: uuid.UUID
    status: str
    start_date: datetime
    end_date: Optional[datetime]
    trial_ends_at: Optional[datetime]
    auto_renew: bool
    payment_method_id: Optional[str]
    stripe_subscription_id: Optional[str]
    created_at: datetime
    updated_at: datetime

class UserPlanWithDetails(UserPlan):
    """UserPlan with associated plan details"""
    plans: Plan

# Enhanced chat schemas REMOVED - No longer needed with streamlined_chat approach

# Request/Response schemas for survey suggestion generation  
class GenerateSuggestionsRequest(BaseSchema):
    survey_id: uuid.UUID
    description: str = Field(..., description="Survey description for context")
    category: str = Field("analysis", description="Suggestion category")
    personality_id: Optional[uuid.UUID] = None
    file_content: Optional[Dict[str, Any]] = Field(None, description="Optional file content for context")

class GenerateSuggestionsResponse(BaseSchema):
    suggestions: List[str]  # Simplified to match SuggestionsResponse pattern
    generation_time: float
    total_suggestions: int

# Role schemas
class RoleBase(BaseSchema):
    role: str = Field(..., description="Role name", max_length=255)

class RoleCreate(RoleBase):
    pass

class RoleUpdate(BaseSchema):
    role: Optional[str] = Field(None, description="Role name", max_length=255)

class RoleResponse(RoleBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

class RoleStatistics(BaseSchema):
    total_roles: int
    total_users: int
    role_usage: List[Dict[str, Any]]

class UserRoleAssignment(BaseSchema):
    user_id: uuid.UUID
    roleid: uuid.UUID

class CheckUserRoleResponse(BaseSchema):
    has_role: bool

# Refresh Token schemas
class RefreshTokenBase(BaseSchema):
    user_id: uuid.UUID
    token: str
    expires_at: datetime
    is_active: bool = True

class RefreshTokenCreate(BaseSchema):
    user_id: uuid.UUID
    expires_at: datetime

class RefreshTokenResponse(BaseSchema):
    id: uuid.UUID
    user_id: uuid.UUID
    expires_at: datetime
    is_active: bool
    created_at: datetime
    last_used: Optional[datetime]

class RefreshTokenStats(BaseSchema):
    total_tokens: int
    active_tokens: int
    expired_tokens: int
    tokens_by_user: List[Dict[str, Any]]

# Email schemas
class EmailTemplateBase(BaseSchema):
    name: str = Field(..., description="Template name", max_length=255)
    subject: str = Field(..., description="Email subject", max_length=500)
    body: str = Field(..., description="Email body content")
    template_type: str = Field(..., description="Type of email template", max_length=100)

class EmailTemplateCreate(EmailTemplateBase):
    pass

class EmailTemplateUpdate(BaseSchema):
    name: Optional[str] = Field(None, max_length=255)
    subject: Optional[str] = Field(None, max_length=500)
    body: Optional[str] = None
    template_type: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None

class EmailTemplateResponse(EmailTemplateBase):
    id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime

class EmailLogBase(BaseSchema):
    recipient: str = Field(..., description="Email recipient")
    subject: str = Field(..., description="Email subject")
    template_used: Optional[str] = Field(None, description="Template name used")
    status: str = Field(..., description="Email status (sent, failed, pending)")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class EmailLogCreate(EmailLogBase):
    user_id: Optional[uuid.UUID] = None

class EmailLogResponse(EmailLogBase):
    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    sent_at: Optional[datetime]
    created_at: datetime

class SendEmailRequest(BaseSchema):
    recipient: str = Field(..., description="Email recipient")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    template_name: Optional[str] = Field(None, description="Template to use")
    template_variables: Optional[Dict[str, Any]] = Field(None, description="Variables for template")

class SendEmailResponse(BaseSchema):
    success: bool
    message: str
    email_log_id: Optional[uuid.UUID] = None

class EmailStats(BaseSchema):
    total_sent: int
    total_failed: int
    total_pending: int
    recent_emails: List[EmailLogResponse]


class TranslationBulkRequest(BaseSchema):
    prompts: List[Dict[str, str]] = Field(..., description="List of prompts to translate")
    target_languages: List[str] = Field(..., description="Target language codes")
    personality_id: Optional[uuid.UUID] = None


# Simple Suggestions schemas
class SimpleSuggestionBase(BaseSchema):
    survey_id: uuid.UUID = Field(..., description="Survey this suggestion is for")
    suggestion_text: str = Field(..., description="The suggestion text")
    category: str = Field(..., description="Suggestion category")
    confidence_score: float = Field(..., description="AI confidence (0.0-1.0)", ge=0.0, le=1.0)
    language_code: str = Field(default="en", description="Language of the suggestion")

class SimpleSuggestionCreate(SimpleSuggestionBase):
    ai_personality_id: Optional[uuid.UUID] = None

# Update forward references for UserWithAccess
UserWithAccess.model_rebuild()

class SimpleSuggestionUpdate(BaseSchema):
    suggestion_text: Optional[str] = None
    category: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    language_code: Optional[str] = None
    is_active: Optional[bool] = None

class SimpleSuggestionResponse(SimpleSuggestionBase):
    id: uuid.UUID
    ai_personality_id: Optional[uuid.UUID]
    generated_by: Optional[uuid.UUID]
    is_active: bool
    created_at: datetime
    updated_at: datetime

class GenerateSimpleSuggestionsRequest(BaseSchema):
    survey_id: uuid.UUID
    count: int = Field(default=5, description="Number of suggestions to generate", ge=1, le=20)
    category: str = Field(default="general", description="Category for suggestions")
    language_code: str = Field(default="en", description="Language for suggestions")
    ai_personality_id: Optional[uuid.UUID] = None

class GenerateSimpleSuggestionsResponse(BaseSchema):
    suggestions: List[SimpleSuggestionResponse]
    generation_time: float
    total_generated: int

class SimpleSuggestionStats(BaseSchema):
    total_suggestions: int
    active_suggestions: int
    suggestions_by_category: Dict[str, int]
    suggestions_by_language: Dict[str, int]
    recent_suggestions: List[SimpleSuggestionResponse]