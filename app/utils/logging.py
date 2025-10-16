import logging
import structlog
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path

from app.core.config import settings

# Async logging setup
log_queue = Queue()
log_thread = None
log_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AsyncLogger")

def _async_log_worker():
    """Background worker for async logging"""
    while True:
        try:
            log_record = log_queue.get()
            if log_record is None:  # Shutdown signal
                break
            
            # Process the log record
            if hasattr(log_record, 'logger'):
                logger = log_record.logger
                method = getattr(logger, log_record.level.lower())
                method(log_record.message, **log_record.extra)
            
            log_queue.task_done()
        except Exception:
            pass  # Ignore logging errors to prevent recursion

# Start async logging worker
log_thread = threading.Thread(target=_async_log_worker, daemon=True)
log_thread.start()

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

# Configure standard logging - Console only, no file logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance"""
    return structlog.get_logger(name)

def get_performance_logger(name: str):
    """Get an optimized logger for performance-critical code"""
    return PerformanceLogger(name)

class LogRecord:
    """Simple log record for async processing"""
    def __init__(self, logger, level, message, extra):
        self.logger = logger
        self.level = level
        self.message = message
        self.extra = extra

class PerformanceLogger:
    """High-performance logger that queues logs for async processing"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.name = name
    
    def _queue_log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Queue log for async processing - Only ERROR and CRITICAL saved to database"""
        # Only process ERROR and CRITICAL logs for database storage
        if level.upper() in ["ERROR", "CRITICAL"]:
            formatted_extra = extra or {}
            formatted_extra.update({
                "timestamp": datetime.utcnow().isoformat(),
                "logger_name": self.name
            })
            
            log_record = LogRecord(self.logger, level, message, formatted_extra)
            try:
                log_queue.put_nowait(log_record)
                
                # Also save to database for errors
                import asyncio
                try:
                    asyncio.create_task(self._save_error_to_database(message, formatted_extra, level))
                except Exception:
                    pass
            except:
                pass  # Drop logs if queue is full (prevents blocking)
        else:
            # For DEBUG, INFO, WARNING - only console logging
            formatted_extra = extra or {}
            formatted_extra.update({
                "timestamp": datetime.utcnow().isoformat(),
                "logger_name": self.name
            })
            
            log_record = LogRecord(self.logger, level, message, formatted_extra)
            try:
                log_queue.put_nowait(log_record)
            except:
                pass
    
    async def _save_error_to_database(self, message: str, extra: Optional[Dict[str, Any]] = None, level: str = "ERROR"):
        """Save error/critical logs to database"""
        try:
            from app.services.lightweight_db_service import lightweight_db
            import uuid
            from datetime import datetime
            
            log_id = str(uuid.uuid4())
            
            query = """
            INSERT INTO logs (id, level, action, error_message, details, created_at, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            await lightweight_db.execute_query(query, [
                log_id, 
                level, 
                f"performance_{level.lower()}", 
                message, 
                extra or {}, 
                datetime.utcnow(),
                datetime.utcnow()
            ])
        except Exception:
            pass  # Silently fail to avoid logging loops
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message (console only)"""
        self._queue_log("DEBUG", message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message (console only)"""
        self._queue_log("INFO", message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message (console only)"""
        self._queue_log("WARNING", message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message (console and database)"""
        if exc_info:
            extra = extra or {}
            extra["exc_info"] = True
        self._queue_log("ERROR", message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message (console and database)"""
        if exc_info:
            extra = extra or {}
            extra["exc_info"] = True
        self._queue_log("CRITICAL", message, extra)


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
        """Log debug message - Console only, not saved to database"""
        self.logger.debug(message, **self._format_extra(extra))
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message - Console only, not saved to database"""
        self.logger.info(message, **self._format_extra(extra))
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message - Console only, not saved to database"""
        self.logger.warning(message, **self._format_extra(extra))
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message - Console AND database"""
        if exc_info:
            extra = extra or {}
            extra["exc_info"] = True
        
        # Log to console
        self.logger.error(message, **self._format_extra(extra))
        
        # Also log to database for ERROR level
        try:
            import asyncio
            asyncio.create_task(self._save_error_to_database(message, extra))
        except Exception:
            pass  # Don't fail if database logging fails
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log critical message - Console AND database"""
        if exc_info:
            extra = extra or {}
            extra["exc_info"] = True
        
        # Log to console
        self.logger.critical(message, **self._format_extra(extra))
        
        # Also log to database for CRITICAL level
        try:
            import asyncio
            asyncio.create_task(self._save_error_to_database(message, extra, level="CRITICAL"))
        except Exception:
            pass  # Don't fail if database logging fails
    
    async def _save_error_to_database(self, message: str, extra: Optional[Dict[str, Any]] = None, level: str = "ERROR"):
        """Save error/critical logs to database"""
        try:
            from app.services.lightweight_db_service import lightweight_db
            import uuid
            from datetime import datetime
            
            log_id = str(uuid.uuid4())
            
            query = """
            INSERT INTO logs (id, level, action, error_message, details, created_at, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            await lightweight_db.execute_query(query, [
                log_id, 
                level, 
                f"application_{level.lower()}", 
                message, 
                extra or {}, 
                datetime.utcnow(),
                datetime.utcnow()
            ])
        except Exception:
            pass  # Silently fail to avoid logging loops
    
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
        """Log to database (async) - Only for ERROR level logs"""
        # Only save ERROR and CRITICAL level logs to database
        if level.upper() not in ["ERROR", "CRITICAL"]:
            return
            
        try:
            from app.services.lightweight_db_service import lightweight_db
            import uuid
            from datetime import datetime
            
            log_id = str(uuid.uuid4())
            
            query = """
            INSERT INTO logs (id, level, action, user_id, method, endpoint, status_code, 
                             response_time, ip_address, user_agent, details, error_message, created_at, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """
            
            action = f"api_{level.lower()}_{method}_{endpoint}" if method and endpoint else f"application_{level.lower()}"
            
            await lightweight_db.execute_query(query, [
                log_id, level, action, user_id, method, endpoint, status_code,
                response_time, ip_address, user_agent, metadata or {}, message,
                datetime.utcnow(), datetime.utcnow()
            ])
                
        except Exception as e:
            # Don't log database errors to avoid infinite loops
            pass




# Security-focused logging
class SecurityLogger(AppLogger):
    """Specialized logger for security events"""
    
    def __init__(self):
        super().__init__("security")
    
    def login_attempt(self, email: str, success: bool, ip_address: str, user_agent: str):
        """Log login attempt - Console only unless failed"""
        level = "ERROR" if not success else "INFO"
        message = f"Login attempt: {email}"
        extra = {
            "event_type": "login_attempt",
            "email": email,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        
        if not success:
            self.error(message, extra)  # This will save to database
        else:
            self.info(message, extra)   # This will only log to console
    
    def admin_action(self, admin_email: str, action: str, target: Optional[str] = None):
        """Log admin actions - Console only"""
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
        """Log security violations - Console AND database (as ERROR)"""
        self.error(
            f"Security violation: {violation_type}",
            {
                "event_type": "security_violation",
                "violation_type": violation_type,
                **details
            }
        )


# Performance logging
class APIPerformanceLogger(AppLogger):
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
        """Log API call performance - Only errors and very slow requests saved to database"""
        extra = {
            "event_type": "api_call",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
            "user_id": user_id
        }
        
        message = f"{method} {endpoint} - {status_code} - {response_time}ms"
        
        # Save to database only if it's an error or very slow (>5 seconds)
        if status_code >= 400:
            self.error(message, extra)  # This will save to database
        elif response_time > 5000:  # Very slow requests (>5 seconds)
            self.error(f"SLOW REQUEST: {message}", extra)  # This will save to database
        elif response_time > 1000:  # Warn for slow requests but don't save to DB
            self.warning(message, extra)  # Console only
        else:
            self.info(message, extra)  # Console only
    
    def database_query(self, query_type: str, execution_time: float, table: Optional[str] = None):
        """Log database query performance - Only very slow queries saved to database"""
        extra = {
            "event_type": "database_query",
            "query_type": query_type,
            "execution_time": execution_time,
            "table": table
        }
        
        message = f"Database query: {query_type} - {execution_time:.3f}s"
        
        # Save to database only if very slow (>2 seconds)
        if execution_time > 2.0:
            self.error(f"SLOW QUERY: {message}", extra)  # This will save to database
        elif execution_time > 0.5:  # Warn for slow queries but don't save to DB
            self.warning(message, extra)  # Console only
        else:
            self.debug(message, extra)  # Console only


# Create logger instances
security_logger = SecurityLogger()
performance_logger = APIPerformanceLogger()


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
    """Helper function to log HTTP requests - Only errors saved to database"""
    performance_logger.api_call(
        endpoint=url,
        method=method,
        status_code=status_code,
        response_time=response_time,
        user_id=user_id
    )
    
    # Log security-relevant requests (errors) to database
    if status_code >= 400:
        security_logger.error(
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
            success = exc_type is None
            
            extra = {
                "event_type": "operation_timing",
                "operation": self.operation_name,
                "execution_time": execution_time,
                "success": success
            }
            
            # Only save to database if operation failed or took too long
            if not success:
                self.logger.error(
                    f"Operation failed: {self.operation_name}",
                    extra
                )
            elif execution_time > 10.0:  # Very slow operations (>10 seconds)
                self.logger.error(
                    f"Very slow operation: {self.operation_name}",
                    extra
                )
            else:
                self.logger.info(
                    f"Operation completed: {self.operation_name}",
                    extra
                )