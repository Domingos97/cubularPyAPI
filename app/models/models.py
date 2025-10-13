# SQLAlchemy database models (actual database tables)
from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, ForeignKey, Index, func, DECIMAL
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid
from datetime import datetime

Base = declarative_base()

class BaseModel(Base):
    """Base model with common fields for all tables"""
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Role(BaseModel):
    """User roles table"""
    __tablename__ = "roles"
    
    role = Column(String(255), nullable=False, unique=True, index=True)
    
    # Relationships
    users = relationship("User", back_populates="role")


class AIPersonality(BaseModel):
    """AI personalities table"""
    __tablename__ = "ai_personalities"
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    detailed_analysis_prompt = Column(Text)
    suggestions_prompt = Column(Text)
    created_by = Column(UUID(as_uuid=True))
    
    # Relationships
    users = relationship("User", back_populates="preferred_personality_rel")
    module_configurations = relationship("ModuleConfiguration", back_populates="ai_personality")
    chat_sessions = relationship("ChatSession", back_populates="personality")


class User(BaseModel):
    """Users table"""
    __tablename__ = "users"
    
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=False)
    password = Column(String(255))  # Will store hashed passwords
    language = Column(String(10), default="en")
    roleid = Column(UUID(as_uuid=True), ForeignKey("roles.id"), nullable=False)
    preferred_personality = Column(UUID(as_uuid=True), ForeignKey("ai_personalities.id"))
    welcome_popup_dismissed = Column(Boolean, default=False)
    email_confirmed = Column(Boolean, default=False)
    email_confirmation_token = Column(String(500))
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    role = relationship("Role", back_populates="users")
    preferred_personality_rel = relationship("AIPersonality", back_populates="users")
    chat_sessions = relationship("ChatSession", back_populates="user")
    refresh_tokens = relationship("RefreshToken", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    user_plans = relationship("UserPlan", back_populates="user")


class RefreshToken(BaseModel):
    """Refresh tokens table"""
    __tablename__ = "refresh_tokens"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token = Column(String(500), nullable=False, unique=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_revoked = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="refresh_tokens")


class LLMSetting(BaseModel):
    """LLM settings table"""
    __tablename__ = "llm_settings"
    
    provider = Column(String(50), nullable=False)
    active = Column(Boolean, default=False)
    api_key = Column(Text)  # Encrypted
    created_by = Column(UUID(as_uuid=True))
    
    # Relationships
    module_configurations = relationship("ModuleConfiguration", back_populates="llm_setting")


class ModuleConfiguration(BaseModel):
    """Module configurations table"""
    __tablename__ = "module_configurations"
    
    module_name = Column(String(100), nullable=False)
    llm_setting_id = Column(UUID(as_uuid=True), ForeignKey("llm_settings.id"))
    model = Column(String(100), nullable=False)
    temperature = Column(DECIMAL(4, 2))
    max_tokens = Column(Integer)
    max_completion_tokens = Column(Integer)
    active = Column(Boolean, default=True)
    created_by = Column(UUID(as_uuid=True))
    ai_personality_id = Column(UUID(as_uuid=True), ForeignKey("ai_personalities.id"))
    
    # Relationships
    llm_setting = relationship("LLMSetting", back_populates="module_configurations")
    ai_personality = relationship("AIPersonality", back_populates="module_configurations")


class Survey(BaseModel):
    """Surveys table"""
    __tablename__ = "surveys"
    
    title = Column(String(255))
    category = Column(String(100))
    description = Column(Text)
    ai_suggestions = Column(ARRAY(Text))
    number_participants = Column(Integer)
    total_files = Column(Integer, default=0)
    processed_data = Column(JSONB)
    processing_status = Column(Text, default="pending")
    primary_language = Column(String(10), default="en")
    
    # Relationships
    files = relationship("SurveyFile", back_populates="survey", cascade="all, delete-orphan")


class SurveyFile(BaseModel):
    """Survey files table"""
    __tablename__ = "survey_files"
    
    survey_id = Column(UUID(as_uuid=True), ForeignKey("surveys.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    storage_path = Column(String(255), nullable=False)
    file_size = Column(Integer)  # Using Integer instead of BIGINT for SQLAlchemy compatibility
    file_hash = Column(String(255))
    upload_date = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    survey = relationship("Survey", back_populates="files")
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    survey = relationship("Survey", back_populates="files")


class ChatSession(BaseModel):
    """Chat sessions table"""
    __tablename__ = "chat_sessions"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    survey_ids = Column(ARRAY(UUID(as_uuid=True)))  # Array of survey IDs
    category = Column(String(100))
    title = Column(String(500))
    personality_id = Column(UUID(as_uuid=True), ForeignKey("ai_personalities.id"))
    selected_file_ids = Column(ARRAY(Text), default=lambda: [])
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    personality = relationship("AIPersonality", back_populates="chat_sessions")


class ChatMessage(Base):
    """Chat messages table"""
    __tablename__ = "chat_messages"
    
    # Override BaseModel fields to match database schema
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Note: No updated_at column in chat_messages table
    
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    content = Column(Text, nullable=False)
    sender = Column(String(20), nullable=False)  # 'user' or 'assistant'
    timestamp = Column(DateTime(timezone=True), default=func.now())
    data_snapshot = Column(JSONB)
    confidence = Column(JSONB)
    personality_used = Column(UUID(as_uuid=True))
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class Notification(BaseModel):
    """Notifications table"""
    __tablename__ = "notifications"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String(50), default="info")
    is_read = Column(Boolean, default=False)
    status = Column(String(50), default="pending")
    priority = Column(Integer, default=1)
    admin_response = Column(Text)
    responded_by = Column(UUID(as_uuid=True))
    responded_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="notifications")


class Plan(BaseModel):
    """Plans table"""
    __tablename__ = "plans"
    
    name = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(200))
    description = Column(Text)
    price = Column(DECIMAL(10, 2))
    currency = Column(String(3), default="USD")
    billing = Column(String(20))  # monthly, yearly
    features = Column(JSONB)  # JSONB array
    max_surveys = Column(Integer)
    max_responses = Column(Integer)
    max_users = Column(Integer)
    max_storage_gb = Column(Integer)
    ai_analysis = Column(Boolean, default=False)
    priority_support = Column(Boolean, default=False)
    custom_branding = Column(Boolean, default=False)
    api_access = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    is_popular = Column(Boolean, default=False)
    
    # Relationships
    user_plans = relationship("UserPlan", back_populates="plan")


class UserPlan(BaseModel):
    """User plans table"""
    __tablename__ = "user_plans"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    plan_id = Column(UUID(as_uuid=True), ForeignKey("plans.id"), nullable=False)
    status = Column(String(50), nullable=False)
    start_date = Column(DateTime(timezone=True), default=func.now())
    end_date = Column(DateTime(timezone=True))
    trial_ends_at = Column(DateTime(timezone=True))
    auto_renew = Column(Boolean, default=True)
    payment_method_id = Column(String(255))
    stripe_subscription_id = Column(String(255))
    
    # Relationships
    user = relationship("User", back_populates="user_plans")
    plan = relationship("Plan", back_populates="user_plans")


class Log(Base):
    """Logs table - matches actual database schema exactly"""
    __tablename__ = "logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(500), nullable=False)
    resource = Column(String(255))
    resource_id = Column(UUID(as_uuid=True))
    details = Column(JSONB)
    ip_address = Column(String(45))  # INET in schema, but String works fine
    user_agent = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    level = Column(String(50), nullable=False, server_default='info', index=True)  # Changed to 50
    method = Column(String(50))  # Changed to 50
    endpoint = Column(String(255))
    status_code = Column(Integer)
    session_id = Column(UUID(as_uuid=True))
    request_body = Column(JSONB)
    response_body = Column(JSONB)
    response_time = Column(Integer)
    error_message = Column(Text)
    stack_trace = Column(Text)
    api_key_used = Column(String(255))
    provider = Column(String(50))
    model = Column(String(100))
    tokens_used = Column(Integer)
    cost = Column(DECIMAL(10,6))
    priority = Column(String(50), server_default='normal')  # Changed to 50
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Note: No updated_at column as it doesn't exist in the actual database


class UserSurveyAccess(BaseModel):
    """User survey access table"""
    __tablename__ = "user_survey_access"
    
    # Override base model timestamps since this table only has granted_at
    created_at = None
    updated_at = None
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    survey_id = Column(UUID(as_uuid=True), ForeignKey("surveys.id"), nullable=False) 
    access_type = Column(String(10), nullable=False)  # read, write, admin
    granted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    granted_at = Column(DateTime(timezone=True), default=func.now())
    expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    survey = relationship("Survey", foreign_keys=[survey_id])
    granted_by_user = relationship("User", foreign_keys=[granted_by])


class UserSurveyFileAccess(BaseModel):
    """User survey file access table"""
    __tablename__ = "user_survey_file_access"
    
    # Override base model timestamps since this table only has granted_at
    created_at = None
    updated_at = None
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    survey_file_id = Column(UUID(as_uuid=True), ForeignKey("survey_files.id"), nullable=False)
    access_type = Column(String(10), nullable=False)  # read, write, admin
    granted_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    granted_at = Column(DateTime(timezone=True), default=func.now())
    expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    survey_file = relationship("SurveyFile", foreign_keys=[survey_file_id])
    granted_by_user = relationship("User", foreign_keys=[granted_by])


# VectorSearchIndex REMOVED - No longer needed with fast_search_service direct file approach

class SupportedLanguage(BaseModel):
    """Supported languages table"""
    __tablename__ = "supported_languages"
    
    code = Column(String(10), nullable=False, unique=True, index=True)  # ISO 639-1 code (e.g., 'en', 'es')
    name = Column(String(100), nullable=False)  # English name (e.g., 'English', 'Spanish')
    native_name = Column(String(100), nullable=False)  # Native name (e.g., 'English', 'Español')
    enabled = Column(Boolean, default=True)
    is_rtl = Column(Boolean, default=False)  # Right-to-left language
    sort_order = Column(Integer, default=0)
    
    # No relationships needed for this simple lookup table


class PromptTranslation(BaseModel):
    """Prompt translations table"""
    __tablename__ = "prompt_translations"
    
    prompt_key = Column(String(255), nullable=False, index=True)  # Unique identifier for the prompt
    language_code = Column(String(10), ForeignKey("supported_languages.code"), nullable=False)
    translated_text = Column(Text, nullable=False)
    context = Column(String(500))  # Context for translators
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # New fields for personality-specific prompt translations
    personality_id = Column(UUID(as_uuid=True), ForeignKey("ai_personalities.id"))
    prompt_type = Column(String(50), default="system")  # 'system', 'detailed_analysis', 'suggestions'
    
    # Relationships
    language = relationship("SupportedLanguage")
    created_by_user = relationship("User")
    personality = relationship("AIPersonality")


class SurveySuggestion(BaseModel):
    """Survey suggestions generated by AI"""
    __tablename__ = "survey_suggestions"
    
    survey_id = Column(UUID(as_uuid=True), ForeignKey("surveys.id"), nullable=False)
    suggestion_text = Column(Text, nullable=False)
    category = Column(String(100))  # e.g., 'analysis', 'improvement', 'interpretation'
    confidence_score = Column(DECIMAL(4, 3), default=0.5)  # AI confidence score 0.0-1.0
    ai_personality_id = Column(UUID(as_uuid=True), ForeignKey("ai_personalities.id"))
    generated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    is_active = Column(Boolean, default=True)
    language_code = Column(String(10), ForeignKey("supported_languages.code"), default="en")
    
    # Relationships
    survey = relationship("Survey")
    ai_personality = relationship("AIPersonality")
    generated_by_user = relationship("User", foreign_keys=[generated_by])
    language = relationship("SupportedLanguage")


class SurveyAnalytics(BaseModel):
    """Survey analytics and metrics"""
    __tablename__ = "survey_analytics"
    
    survey_id = Column(UUID(as_uuid=True), ForeignKey("surveys.id"), nullable=False)
    total_files = Column(Integer, default=0)
    total_file_size = Column(Integer, default=0)  # Size in bytes
    total_suggestions = Column(Integer, default=0)
    total_chat_sessions = Column(Integer, default=0)
    total_chat_messages = Column(Integer, default=0)
    last_activity = Column(DateTime(timezone=True))
    analysis_status = Column(String(50), default="pending")  # pending, processing, completed, error
    analytics_metadata = Column(Text)  # JSON metadata for additional metrics
    
    # Relationships
    survey = relationship("Survey")


class EnhancedChatSession(BaseModel):
    """Enhanced chat sessions with RAG capabilities"""
    __tablename__ = "enhanced_chat_sessions"
    
    session_name = Column(String(255))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    survey_ids = Column(ARRAY(UUID), nullable=False)  # Array of survey IDs for RAG context
    personality_id = Column(UUID(as_uuid=True), ForeignKey("ai_personalities.id"))
    is_active = Column(Boolean, default=True)
    language_code = Column(String(10), ForeignKey("supported_languages.code"), default="en")
    context_metadata = Column(Text)  # JSON for session context
    
    # Relationships
    user = relationship("User")
    personality = relationship("AIPersonality")
    language = relationship("SupportedLanguage")
    messages = relationship("EnhancedChatMessage", back_populates="session", cascade="all, delete-orphan")


class EnhancedChatMessage(BaseModel):
    """Enhanced chat messages with RAG context"""
    __tablename__ = "enhanced_chat_messages"
    
    session_id = Column(UUID(as_uuid=True), ForeignKey("enhanced_chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    search_results = Column(Text)  # JSON of RAG search results used
    confidence_score = Column(DECIMAL(4, 3))  # AI response confidence
    processing_time = Column(Integer)  # Processing time in milliseconds
    tokens_used = Column(Integer)  # Tokens consumed for this message
    language_detected = Column(String(10))  # Auto-detected language
    
    # Relationships
    session = relationship("EnhancedChatSession", back_populates="messages")


# Create indexes for better performance
Index('idx_users_email', User.email)
Index('idx_users_roleid', User.roleid)
Index('idx_refresh_tokens_user_id', RefreshToken.user_id)
Index('idx_refresh_tokens_expires_at', RefreshToken.expires_at)
Index('idx_survey_files_survey_id', SurveyFile.survey_id)
Index('idx_chat_sessions_user_id', ChatSession.user_id)
Index('idx_chat_messages_session_id', ChatMessage.session_id)
Index('idx_notifications_user_id', Notification.user_id)
Index('idx_notifications_is_read', Notification.is_read)
Index('idx_logs_level', Log.level)
Index('idx_logs_user_id', Log.user_id)
Index('idx_logs_created_at', Log.created_at)
Index('idx_user_survey_access_user_id', UserSurveyAccess.user_id)
Index('idx_user_survey_access_survey_id', UserSurveyAccess.survey_id)
Index('idx_user_survey_file_access_user_id', UserSurveyFileAccess.user_id)
Index('idx_user_survey_file_access_file_id', UserSurveyFileAccess.survey_file_id)
# VectorSearchIndex indexes REMOVED - No longer needed with fast_search_service
# Index('idx_vector_search_index_survey_id', VectorSearchIndex.survey_id)
# Index('idx_vector_search_index_file_id', VectorSearchIndex.file_id)
Index('idx_supported_languages_code', SupportedLanguage.code)
Index('idx_supported_languages_enabled', SupportedLanguage.enabled)
Index('idx_prompt_translations_key', PromptTranslation.prompt_key)
Index('idx_prompt_translations_language', PromptTranslation.language_code)
Index('idx_survey_suggestions_survey_id', SurveySuggestion.survey_id)
Index('idx_survey_suggestions_category', SurveySuggestion.category)
Index('idx_survey_suggestions_active', SurveySuggestion.is_active)
Index('idx_survey_analytics_survey_id', SurveyAnalytics.survey_id)
Index('idx_survey_analytics_status', SurveyAnalytics.analysis_status)
Index('idx_enhanced_chat_sessions_user_id', EnhancedChatSession.user_id)
Index('idx_enhanced_chat_sessions_active', EnhancedChatSession.is_active)
Index('idx_enhanced_chat_messages_session_id', EnhancedChatMessage.session_id)
Index('idx_enhanced_chat_messages_role', EnhancedChatMessage.role)