from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, and_, or_, desc
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import json

from app.models.models import Log
from app.models.schemas import LogCreate, LogResponse, LogLevel
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LoggingService:
    """Service for managing application logs with analytics"""
    
    LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def get_log_levels(self) -> List[str]:
        """Get list of available log levels"""
        return self.LOG_LEVELS.copy()
    
    async def get_logs(
        self, 
        db: AsyncSession, 
        filters: Dict[str, Any]
    ) -> List[LogResponse]:
        """Get logs with filtering and pagination"""
        try:
            query = select(Log)
            
            # Apply filters
            conditions = []
            
            if filters.get("level"):
                conditions.append(Log.level == filters["level"])
            
            if filters.get("user_id"):
                conditions.append(Log.user_id == filters["user_id"])
            
            if filters.get("endpoint"):
                conditions.append(Log.endpoint.ilike(f"%{filters['endpoint']}%"))
            
            if filters.get("method"):
                conditions.append(Log.method == filters["method"])
            
            if filters.get("start_date"):
                conditions.append(Log.created_at >= filters["start_date"])
            
            if filters.get("end_date"):
                conditions.append(Log.created_at <= filters["end_date"])
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Apply pagination and ordering
            query = query.order_by(desc(Log.created_at))
            query = query.offset(filters.get("skip", 0))
            query = query.limit(filters.get("limit", 50))
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            return [
                LogResponse(
                    id=log.id,
                    level=log.level,
                    action=log.action,
                    user_id=log.user_id,
                    resource=log.resource,
                    resource_id=log.resource_id,
                    details=log.details,
                    ip_address=str(log.ip_address) if log.ip_address else None,
                    user_agent=log.user_agent,
                    timestamp=log.timestamp,
                    method=log.method,
                    endpoint=log.endpoint,
                    status_code=log.status_code,
                    session_id=log.session_id,
                    request_body=log.request_body,
                    response_body=log.response_body,
                    response_time=log.response_time,
                    error_message=log.error_message,
                    stack_trace=log.stack_trace,
                    api_key_used=log.api_key_used,
                    provider=log.provider,
                    model=log.model,
                    tokens_used=log.tokens_used,
                    cost=float(log.cost) if log.cost else None,
                    priority=log.priority,
                    analytics_metadata=log.details,  # For backward compatibility
                    created_at=log.created_at
                )
                for log in logs
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving logs: {str(e)}")
            raise
    
    async def get_logs_count(
        self, 
        db: AsyncSession, 
        filters: Dict[str, Any]
    ) -> int:
        """Get count of logs matching filters"""
        try:
            query = select(func.count(Log.id))
            
            # Apply same filters as get_logs
            conditions = []
            
            if filters.get("level"):
                conditions.append(Log.level == filters["level"])
            
            if filters.get("user_id"):
                conditions.append(Log.user_id == filters["user_id"])
            
            if filters.get("endpoint"):
                conditions.append(Log.endpoint.ilike(f"%{filters['endpoint']}%"))
            
            if filters.get("method"):
                conditions.append(Log.method == filters["method"])
            
            if filters.get("start_date"):
                conditions.append(Log.created_at >= filters["start_date"])
            
            if filters.get("end_date"):
                conditions.append(Log.created_at <= filters["end_date"])
            
            if conditions:
                query = query.where(and_(*conditions))
            
            result = await db.execute(query)
            return result.scalar_one()
            
        except Exception as e:
            logger.error(f"Error getting logs count: {str(e)}")
            raise
    
    async def get_recent_logs(
        self, 
        db: AsyncSession, 
        limit: int,
        level: Optional[str] = None
    ) -> List[LogResponse]:
        """Get most recent logs"""
        try:
            query = select(Log)
            
            if level:
                query = query.where(Log.level == level)
            
            query = query.order_by(desc(Log.created_at)).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            return [
                LogResponse(
                    id=log.id,
                    level=log.level,
                    action=log.action,
                    user_id=log.user_id,
                    resource=log.resource,
                    resource_id=log.resource_id,
                    details=log.details,
                    ip_address=str(log.ip_address) if log.ip_address else None,
                    user_agent=log.user_agent,
                    timestamp=log.timestamp,
                    method=log.method,
                    endpoint=log.endpoint,
                    status_code=log.status_code,
                    session_id=log.session_id,
                    request_body=log.request_body,
                    response_body=log.response_body,
                    response_time=log.response_time,
                    error_message=log.error_message,
                    stack_trace=log.stack_trace,
                    api_key_used=log.api_key_used,
                    provider=log.provider,
                    model=log.model,
                    tokens_used=log.tokens_used,
                    cost=float(log.cost) if log.cost else None,
                    priority=log.priority,
                    analytics_metadata=log.details,  # For backward compatibility
                    created_at=log.created_at
                )
                for log in logs
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving recent logs: {str(e)}")
            raise
    
    async def get_error_logs(
        self, 
        db: AsyncSession, 
        start_date: datetime,
        end_date: datetime,
        skip: int = 0,
        limit: int = 50
    ) -> List[LogResponse]:
        """Get error logs from specified time period"""
        try:
            query = select(Log).where(
                and_(
                    Log.level.in_(["ERROR", "CRITICAL"]),
                    Log.created_at >= start_date,
                    Log.created_at <= end_date
                )
            ).order_by(desc(Log.created_at)).offset(skip).limit(limit)
            
            result = await db.execute(query)
            logs = result.scalars().all()
            
            return [
                LogResponse(
                    id=log.id,
                    level=log.level,
                    action=log.action,
                    user_id=log.user_id,
                    resource=log.resource,
                    resource_id=log.resource_id,
                    details=log.details,
                    ip_address=str(log.ip_address) if log.ip_address else None,
                    user_agent=log.user_agent,
                    timestamp=log.timestamp,
                    method=log.method,
                    endpoint=log.endpoint,
                    status_code=log.status_code,
                    session_id=log.session_id,
                    request_body=log.request_body,
                    response_body=log.response_body,
                    response_time=log.response_time,
                    error_message=log.error_message,
                    stack_trace=log.stack_trace,
                    api_key_used=log.api_key_used,
                    provider=log.provider,
                    model=log.model,
                    tokens_used=log.tokens_used,
                    cost=float(log.cost) if log.cost else None,
                    priority=log.priority,
                    analytics_metadata=log.details,  # For backward compatibility
                    created_at=log.created_at
                )
                for log in logs
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving error logs: {str(e)}")
            raise
    
    async def get_log_statistics(
        self, 
        db: AsyncSession, 
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """Get comprehensive log statistics"""
        try:
            base_conditions = [
                Log.created_at >= start_date,
                Log.created_at <= end_date
            ]
            
            if user_id:
                base_conditions.append(Log.user_id == user_id)
            
            # Total logs by level
            level_query = select(
                Log.level,
                func.count(Log.id).label('count')
            ).where(and_(*base_conditions)).group_by(Log.level)
            
            level_result = await db.execute(level_query)
            logs_by_level = {row.level: row.count for row in level_result}
            
            # Top endpoints
            endpoint_query = select(
                Log.endpoint,
                func.count(Log.id).label('count')
            ).where(
                and_(*base_conditions, Log.endpoint.isnot(None))
            ).group_by(Log.endpoint).order_by(desc('count')).limit(10)
            
            endpoint_result = await db.execute(endpoint_query)
            top_endpoints = [
                {"endpoint": row.endpoint, "count": row.count}
                for row in endpoint_result
            ]
            
            # Error trends (by day)
            error_trends_query = select(
                func.date(Log.created_at).label('date'),
                func.count(Log.id).label('count')
            ).where(
                and_(
                    *base_conditions,
                    Log.level.in_(["ERROR", "CRITICAL"])
                )
            ).group_by(func.date(Log.created_at)).order_by(func.date(Log.created_at))
            
            error_trends_result = await db.execute(error_trends_query)
            error_trends = [
                {"date": str(row.date), "count": row.count}
                for row in error_trends_result
            ]
            
            # User activity (if not filtering by specific user)
            user_activity = []
            if not user_id:
                user_activity_query = select(
                    Log.user_id,
                    func.count(Log.id).label('count')
                ).where(
                    and_(*base_conditions, Log.user_id.isnot(None))
                ).group_by(Log.user_id).order_by(desc('count')).limit(10)
                
                user_activity_result = await db.execute(user_activity_query)
                user_activity = [
                    {"user_id": str(row.user_id), "count": row.count}
                    for row in user_activity_result
                ]
            
            # Response time statistics
            response_time_query = select(
                func.avg(Log.response_time).label('avg_response_time'),
                func.min(Log.response_time).label('min_response_time'),
                func.max(Log.response_time).label('max_response_time')
            ).where(
                and_(*base_conditions, Log.response_time.isnot(None))
            )
            
            response_time_result = await db.execute(response_time_query)
            response_time_row = response_time_result.first()
            
            response_time_stats = {
                "avg": float(response_time_row.avg_response_time) if response_time_row.avg_response_time else None,
                "min": response_time_row.min_response_time,
                "max": response_time_row.max_response_time
            }
            
            # Total counts
            total_query = select(func.count(Log.id)).where(and_(*base_conditions))
            total_result = await db.execute(total_query)
            total_logs = total_result.scalar_one()
            
            return {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "user_filter": str(user_id) if user_id else None
                },
                "summary": {
                    "total_logs": total_logs,
                    "logs_by_level": logs_by_level,
                    "error_count": logs_by_level.get("ERROR", 0) + logs_by_level.get("CRITICAL", 0)
                },
                "top_endpoints": top_endpoints,
                "error_trends": error_trends,
                "user_activity": user_activity,
                "response_time_stats": response_time_stats,
                "generated_at": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error getting log statistics: {str(e)}")
            raise
    
    async def create_log_entry(
        self, 
        db: AsyncSession, 
        log_data: LogCreate
    ) -> LogResponse:
        """Create a new log entry"""
        try:
            # Validate log level
            if log_data.level not in self.LOG_LEVELS:
                raise ValueError(f"Invalid log level. Must be one of: {self.LOG_LEVELS}")
            
            # Create log entry
            log_entry = Log(
                level=log_data.level,
                action=log_data.action,
                user_id=log_data.user_id,
                resource=log_data.resource,
                resource_id=log_data.resource_id,
                details=log_data.details or log_data.analytics_metadata,  # Handle both field names
                ip_address=log_data.ip_address,
                user_agent=log_data.user_agent,
                timestamp=log_data.timestamp,
                method=log_data.method,
                endpoint=log_data.endpoint,
                status_code=log_data.status_code,
                session_id=log_data.session_id,
                request_body=log_data.request_body,
                response_body=log_data.response_body,
                response_time=log_data.response_time,
                error_message=log_data.error_message,
                stack_trace=log_data.stack_trace,
                api_key_used=log_data.api_key_used,
                provider=log_data.provider,
                model=log_data.model,
                tokens_used=log_data.tokens_used,
                cost=log_data.cost,
                priority=log_data.priority
            )
            
            db.add(log_entry)
            await db.commit()
            await db.refresh(log_entry)
            
            return LogResponse(
                id=log_entry.id,
                level=log_entry.level,
                action=log_entry.action,
                user_id=log_entry.user_id,
                resource=log_entry.resource,
                resource_id=log_entry.resource_id,
                details=log_entry.details,
                ip_address=str(log_entry.ip_address) if log_entry.ip_address else None,
                user_agent=log_entry.user_agent,
                timestamp=log_entry.timestamp,
                method=log_entry.method,
                endpoint=log_entry.endpoint,
                status_code=log_entry.status_code,
                session_id=log_entry.session_id,
                request_body=log_entry.request_body,
                response_body=log_entry.response_body,
                response_time=log_entry.response_time,
                error_message=log_entry.error_message,
                stack_trace=log_entry.stack_trace,
                api_key_used=log_entry.api_key_used,
                provider=log_entry.provider,
                model=log_entry.model,
                tokens_used=log_entry.tokens_used,
                cost=float(log_entry.cost) if log_entry.cost else None,
                priority=log_entry.priority,
                analytics_metadata=log_entry.details,  # For backward compatibility
                created_at=log_entry.created_at
            )
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating log entry: {str(e)}")
            raise
    
    async def count_logs_for_cleanup(
        self, 
        db: AsyncSession, 
        cutoff_date: datetime,
        level: Optional[str] = None
    ) -> int:
        """Count logs that would be deleted in cleanup"""
        try:
            conditions = [Log.created_at < cutoff_date]
            
            if level:
                conditions.append(Log.level == level)
            
            query = select(func.count(Log.id)).where(and_(*conditions))
            result = await db.execute(query)
            return result.scalar_one()
            
        except Exception as e:
            logger.error(f"Error counting logs for cleanup: {str(e)}")
            raise
    
    async def cleanup_old_logs(
        self, 
        db: AsyncSession, 
        cutoff_date: datetime,
        level: Optional[str] = None
    ) -> int:
        """Delete old logs and return count of deleted entries"""
        try:
            conditions = [Log.created_at < cutoff_date]
            
            if level:
                conditions.append(Log.level == level)
            
            # First count what will be deleted
            count_query = select(func.count(Log.id)).where(and_(*conditions))
            count_result = await db.execute(count_query)
            count_to_delete = count_result.scalar_one()
            
            # Delete the logs
            delete_query = delete(Log).where(and_(*conditions))
            await db.execute(delete_query)
            await db.commit()
            
            return count_to_delete
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error cleaning up logs: {str(e)}")
            raise


# Create a singleton instance
logging_service = LoggingService()