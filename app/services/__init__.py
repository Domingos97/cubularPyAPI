"""
Services module - contains all business logic services
"""

# Core services
from .lightweight_auth_service import auth_service
# from .simple_ai_service import simple_ai as ai_service  # TODO: Convert to lightweight_db
from .survey_service import survey_service
from .embedding_service import embedding_service
from .survey_indexing_service import survey_indexing_service

# Enhanced services (ported from TypeScript)
from .background_queue_service import background_queue, add_survey_analysis_job, add_file_processing_job
from .fast_search_service import fast_search_service

# Other services
from .email_service import email_service
from .logging_service import logging_service
from .notification_service import notification_service

__all__ = [
    # Core services
    'auth_service',
    'ai_service', 
    'survey_service',
    'embedding_service',
    'survey_indexing_service',
    
    # Enhanced services
    'background_queue',
    'add_survey_analysis_job',
    'add_file_processing_job', 
    'fast_search_service',
    
    # Other services
    'email_service',
    'logging_service',
    'notification_service'
]