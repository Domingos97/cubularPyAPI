from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Optional
from datetime import datetime, timedelta
import uuid

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import  get_current_regular_user, get_current_admin_user, SimpleUser
from app.models.schemas import LogCreate, LogResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/")
async def get_logs(
    level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by user ID"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    method: Optional[str] = Query(None, description="Filter by HTTP method"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    offset: int = Query(0, ge=0, description="Number of logs to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Number of logs to return"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get logs with filtering and pagination (Admin only)
    
    - **level**: Filter by log level
    - **user_id**: Filter by specific user
    - **endpoint**: Filter by API endpoint
    - **method**: Filter by HTTP method
    - **start_date**: Start date for filtering (ISO format)
    - **end_date**: End date for filtering (ISO format)
    - **offset**: Number of logs to skip (pagination)
    - **limit**: Maximum number of logs to return
    
    Returns filtered and paginated log entries
    """
    try:
        filters = {
            "level": level,
            "user_id": user_id,
            "endpoint": endpoint,
            "method": method,
            "start_date": start_date,
            "end_date": end_date,
            "skip": offset,  # Convert offset to skip for internal use
            "limit": limit
        }
        
        logs = await db.get_logs(filters)
        
        # Get total count for pagination
        count_filters = {k: v for k, v in filters.items() if k not in ['skip', 'limit']}
        total_count = await db.get_logs_count(count_filters)
        
        logger.info(f"Retrieved {len(logs)} logs for admin {current_user.id} with filters: {filters}")
        
        # Return in the format expected by frontend
        return {
            "data": logs,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "hasMore": (offset + limit) < total_count
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve logs"
        )


@router.get("/count")
async def get_logs_count(
    level: Optional[str] = Query(None, description="Filter by log level"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by user ID"),
    endpoint: Optional[str] = Query(None, description="Filter by endpoint"),
    method: Optional[str] = Query(None, description="Filter by HTTP method"),
    start_date: Optional[datetime] = Query(None, description="Start date for filtering"),
    end_date: Optional[datetime] = Query(None, description="End date for filtering"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get count of logs matching filters (Admin only)
    
    Returns total count of logs matching the specified filters
    """
    try:
        filters = {
            "level": level,
            "user_id": user_id,
            "endpoint": endpoint,
            "method": method,
            "start_date": start_date,
            "end_date": end_date
        }
        
        count = await db.get_logs_count(filters)
        
        logger.info(f"Retrieved logs count ({count}) for admin {current_user.id}")
        return {
            "count": count,
            "filters": filters,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving logs count: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve logs count"
        )


@router.get("/stats")
async def get_log_statistics(
    start_date: Optional[datetime] = Query(None, description="Start date for statistics"),
    end_date: Optional[datetime] = Query(None, description="End date for statistics"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by user ID"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get log statistics for dashboard (Admin only)
    
    - **start_date**: Start date for statistics period
    - **end_date**: End date for statistics period
    - **user_id**: Optional filter by specific user
    
    Returns comprehensive log statistics including:
    - Total logs by level
    - Top endpoints
    - Error trends
    - User activity
    """
    try:
        # Default to last 7 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        stats = await db.get_log_statistics(
            start_date, end_date, user_id
        )
        
        logger.info(f"Retrieved log statistics for admin {current_user.id} from {start_date} to {end_date}")
        return stats
        
    except Exception as e:
        logger.error(f"Error retrieving log statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve log statistics"
        )


@router.get("/levels")
async def get_log_levels(
    db: LightweightDBService = Depends(get_lightweight_db)):
    """
    Get available log levels (Admin only)
    
    Returns list of available log levels for filtering
    """
    try:
        levels = db.get_log_levels()
        
        return {
            "levels": levels,
            "total": len(levels),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving log levels: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve log levels"
        )


@router.get("/recent")
async def get_recent_logs(
    limit: int = Query(20, ge=1, le=100, description="Number of recent logs to return"),
    level: Optional[str] = Query(None, description="Filter by log level"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get most recent logs (Admin only)
    
    - **limit**: Maximum number of recent logs to return
    - **level**: Optional filter by log level
    
    Returns most recent log entries
    """
    try:
        logs = await db.get_recent_logs(limit, level)
        
        logger.info(f"Retrieved {len(logs)} recent logs for admin {current_user.id}")
        return logs
        
    except Exception as e:
        logger.error(f"Error retrieving recent logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recent logs"
        )


@router.get("/errors")
async def get_error_logs(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    skip: int = Query(0, ge=0, description="Number of errors to skip"),
    limit: int = Query(50, ge=1, le=200, description="Number of errors to return"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get error logs from specified time period (Admin only)
    
    - **hours**: Number of hours to look back (max 168 = 1 week)
    - **skip**: Number of errors to skip (pagination)
    - **limit**: Maximum number of errors to return
    
    Returns error-level log entries from the specified time period
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)
        
        errors = await db.get_error_logs(
            start_date, end_date, skip, limit
        )
        
        logger.info(f"Retrieved {len(errors)} error logs for admin {current_user.id} from last {hours} hours")
        return {
            "errors": errors,
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "hours": hours
            },
            "pagination": {
                "skip": skip,
                "limit": limit,
                "returned": len(errors)
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving error logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve error logs"
        )


@router.post("/", response_model=LogResponse, status_code=status.HTTP_201_CREATED)
async def create_log_entry(
    log_data: LogCreate,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Create a manual log entry (Admin only)
    
    - **level**: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - **message**: Log message
    - **user_id**: Optional user ID associated with the log
    - **endpoint**: Optional endpoint path
    - **method**: Optional HTTP method
    - **status_code**: Optional HTTP status code
    - **response_time**: Optional response time in milliseconds
    - **ip_address**: Optional IP address
    - **user_agent**: Optional user agent string
    - **metadata**: Optional additional metadata as JSON
    
    Returns the created log entry
    """
    try:
        log_entry = await db.create_log_entry(log_data)
        
        logger.info(f"Created manual log entry {log_entry.id} by admin {current_user.id}")
        return log_entry
        
    except ValueError as e:
        logger.warning(f"Invalid data for log entry creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating log entry: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create log entry"
        )


@router.delete("/cleanup")
async def cleanup_old_logs(
    days: int = Query(30, ge=1, le=365, description="Delete logs older than this many days"),
    level: Optional[str] = Query(None, description="Only delete logs of this level"),
    dry_run: bool = Query(True, description="If true, only count what would be deleted"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Clean up old log entries (Admin only)
    
    - **days**: Delete logs older than this many days
    - **level**: Optional - only delete logs of this level
    - **dry_run**: If true, only count what would be deleted without actually deleting
    
    Returns count of logs that were (or would be) deleted
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        if dry_run:
            count = await db.count_logs_for_cleanup(cutoff_date, level)
            logger.info(f"Dry run: Would delete {count} logs older than {cutoff_date} by admin {current_user.id}")
            return {
                "dry_run": True,
                "would_delete": count,
                "cutoff_date": cutoff_date,
                "level_filter": level,
                "message": f"Would delete {count} log entries older than {days} days"
            }
        else:
            deleted_count = await db.cleanup_old_logs(cutoff_date, level)
            logger.warning(f"Deleted {deleted_count} old logs by admin {current_user.id}")
            return {
                "dry_run": False,
                "deleted": deleted_count,
                "cutoff_date": cutoff_date,
                "level_filter": level,
                "message": f"Deleted {deleted_count} log entries older than {days} days"
            }
        
    except Exception as e:
        logger.error(f"Error during log cleanup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup logs"
        )