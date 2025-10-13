import logging
import structlog
import sys
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

from app.core.config import settings

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Create logs directory
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure standard logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logs_dir / "app.log")
    ]
)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)


class AppLogger:
    """Application logger with structured logging and database integration"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.name = name
    
    def _format_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format extra fields for logging"""
        formatted_extra = extra or {}
        formatted_extra.update({
            "timestamp": datetime.utcnow().isoformat(),
            "logger_name": self.name
        })
        return formatted_extra
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self.logger.debug(message, **self._format_extra(extra))
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self.logger.info(message, **self._format_extra(extra))
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(message, **self._format_extra(extra))
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message"""
        if exc_info:
            extra = extra or {}
            extra["exc_info"] = True
        self.logger.error(message, **self._format_extra(extra))
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message"""
        if exc_info:
            extra = extra or {}
            extra["exc_info"] = True
        self.logger.critical(message, **self._format_extra(extra))
    
    async def log_to_database(
        self,
        level: str,
        message: str,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log to database (async)"""
        try:
            from app.models.models import Log
            from app.core.database import AsyncSessionLocal
            import uuid
            import json
            
            async with AsyncSessionLocal() as db:
                log_entry = Log(
                    level=level.upper(),
                    message=message,
                    user_id=uuid.UUID(user_id) if user_id else None,
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    response_time=response_time,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    metadata_json=json.dumps(metadata) if metadata else None
                )
                
                db.add(log_entry)
                await db.commit()
                
        except Exception as e:
            # Fallback to file logging if database logging fails
            self.error(f"Failed to log to database: {str(e)}", {"original_message": message})


class DatabaseLogHandler(logging.Handler):
    """Custom log handler that writes to database"""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("database_handler")
    
    def emit(self, record):
        """Emit log record to database"""
        # This is a synchronous handler, so we'll just use file logging
        # In a production environment, you might use a queue for async database logging
        pass


# Security-focused logging
class SecurityLogger(AppLogger):
    """Specialized logger for security events"""
    
    def __init__(self):
        super().__init__("security")
    
    def login_attempt(self, email: str, success: bool, ip_address: str, user_agent: str):
        """Log login attempt"""
        self.info(
            f"Login attempt: {email}",
            {
                "event_type": "login_attempt",
                "email": email,
                "success": success,
                "ip_address": ip_address,
                "user_agent": user_agent
            }
        )
    
    def admin_action(self, admin_email: str, action: str, target: Optional[str] = None):
        """Log admin actions"""
        self.info(
            f"Admin action: {action}",
            {
                "event_type": "admin_action",
                "admin_email": admin_email,
                "action": action,
                "target": target
            }
        )
    
    def security_violation(self, violation_type: str, details: Dict[str, Any]):
        """Log security violations"""
        self.warning(
            f"Security violation: {violation_type}",
            {
                "event_type": "security_violation",
                "violation_type": violation_type,
                **details
            }
        )


# Performance logging
class PerformanceLogger(AppLogger):
    """Specialized logger for performance monitoring"""
    
    def __init__(self):
        super().__init__("performance")
    
    def api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: int,
        user_id: Optional[str] = None
    ):
        """Log API call performance"""
        level = "warning" if response_time > 1000 else "info"  # Warn if > 1 second
        
        getattr(self, level)(
            f"{method} {endpoint} - {status_code} - {response_time}ms",
            {
                "event_type": "api_call",
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time": response_time,
                "user_id": user_id
            }
        )
    
    def database_query(self, query_type: str, execution_time: float, table: Optional[str] = None):
        """Log database query performance"""
        level = "warning" if execution_time > 0.5 else "debug"  # Warn if > 500ms
        
        getattr(self, level)(
            f"Database query: {query_type} - {execution_time:.3f}s",
            {
                "event_type": "database_query",
                "query_type": query_type,
                "execution_time": execution_time,
                "table": table
            }
        )


# Create logger instances
security_logger = SecurityLogger()
performance_logger = PerformanceLogger()


# Request logging middleware helper
def log_request(
    method: str,
    url: str,
    status_code: int,
    response_time: int,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
):
    """Helper function to log HTTP requests"""
    performance_logger.api_call(
        endpoint=url,
        method=method,
        status_code=status_code,
        response_time=response_time,
        user_id=user_id
    )
    
    # Also log security-relevant requests
    if status_code >= 400:
        security_logger.info(
            f"HTTP {status_code}: {method} {url}",
            {
                "event_type": "http_error",
                "method": method,
                "url": url,
                "status_code": status_code,
                "user_id": user_id,
                "ip_address": ip_address,
                "user_agent": user_agent
            }
        )


# Context managers for performance logging
class LogExecutionTime:
    """Context manager to log execution time"""
    
    def __init__(self, operation_name: str, logger: AppLogger = None):
        self.operation_name = operation_name
        self.logger = logger or performance_logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            execution_time = (datetime.utcnow() - self.start_time).total_seconds()
            self.logger.info(
                f"Operation completed: {self.operation_name}",
                {
                    "event_type": "operation_timing",
                    "operation": self.operation_name,
                    "execution_time": execution_time,
                    "success": exc_type is None
                }
            )