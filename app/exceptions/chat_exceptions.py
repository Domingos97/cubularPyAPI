"""
Custom exceptions for chat-related operations
"""

class ChatProcessingError(Exception):
    """Raised when chat processing fails"""
    def __init__(self, message: str, error_type: str = None, details: dict = None):
        self.message = message
        self.error_type = error_type or "ChatProcessingError"
        self.details = details or {}
        super().__init__(self.message)

class AIResponseError(ChatProcessingError):
    """Raised when AI response generation fails"""
    def __init__(self, message: str, provider: str = None, model: str = None, details: dict = None):
        self.provider = provider
        self.model = model
        super().__init__(message, "AIResponseError", details)

class SearchError(ChatProcessingError):
    """Raised when search operations fail"""
    def __init__(self, message: str, search_type: str = None, details: dict = None):
        self.search_type = search_type
        super().__init__(message, "SearchError", details)

class EmbeddingError(ChatProcessingError):
    """Raised when embedding generation fails"""
    def __init__(self, message: str, embedding_provider: str = None, details: dict = None):
        self.embedding_provider = embedding_provider
        super().__init__(message, "EmbeddingError", details)