from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime
import uuid

from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_admin_user
from app.models.schemas import (
    NotificationCreate,
    NotificationUpdate,
    Notification,
    NotificationType,
    SuccessResponse
)
from app.models.models import User
from app.services.notification_service import notification_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[Notification])
async def get_user_notifications(
    unread_only: bool = Query(False, description="Filter to unread notifications only"),
    notification_type: Optional[NotificationType] = Query(None, description="Filter by notification type"),
    skip: int = Query(0, ge=0, description="Number of notifications to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of notifications to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get notifications for the current user
    
    - **unread_only**: If true, only return unread notifications
    - **notification_type**: Filter by specific notification type
    - **skip**: Number of notifications to skip (pagination)
    - **limit**: Maximum number of notifications to return
    
    Returns list of user's notifications
    """
    try:
        notifications = await notification_service.get_user_notifications(
            db, current_user.id, unread_only, notification_type, skip, limit
        )
        
        logger.info(f"Retrieved {len(notifications)} notifications for user {current_user.id}")
        return notifications
        
    except Exception as e:
        logger.error(f"Error retrieving notifications for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve notifications"
        )


@router.get("/unread/count")
async def get_unread_notifications_count(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get count of unread notifications for the current user
    
    Returns count of unread notifications
    """
    try:
        count = await notification_service.get_unread_count(db, current_user.id)
        
        return {
            "unread_count": count,
            "user_id": current_user.id,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting unread count for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get unread notifications count"
        )


@router.get("/my", response_model=List[Notification])
async def get_my_notifications(
    unread_only: bool = Query(False, description="Filter to unread notifications only"),
    notification_type: Optional[NotificationType] = Query(None, description="Filter by notification type"),
    skip: int = Query(0, ge=0, description="Number of notifications to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of notifications to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get notifications for the current user (alternative endpoint)
    
    - **unread_only**: If true, only return unread notifications
    - **notification_type**: Filter by specific notification type
    - **skip**: Number of notifications to skip (pagination)
    - **limit**: Maximum number of notifications to return
    
    Returns list of user's notifications
    """
    try:
        notifications = await notification_service.get_user_notifications(
            db, current_user.id, unread_only, notification_type, skip, limit
        )
        
        logger.info(f"Retrieved {len(notifications)} notifications for user {current_user.id} via /my endpoint")
        return notifications
        
    except Exception as e:
        logger.error(f"Failed to get user notifications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user notifications"
        )


@router.get("/{notification_id}", response_model=Notification)
async def get_notification_by_id(
    notification_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get specific notification by ID
    
    - **notification_id**: UUID of the notification to retrieve
    
    Returns the notification details (only if it belongs to the current user)
    """
    try:
        notification = await notification_service.get_notification_by_id(
            db, notification_id, current_user.id
        )
        
        if not notification:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found"
            )
        
        logger.info(f"Retrieved notification {notification_id} for user {current_user.id}")
        return notification
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving notification {notification_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve notification"
        )


@router.post("/", response_model=Notification, status_code=status.HTTP_201_CREATED)
async def create_notification(
    notification_data: NotificationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Create a new notification (Admin only)
    
    - **user_id**: UUID of the user to notify
    - **type**: Type of notification
    - **title**: Notification title (max 200 chars)
    - **message**: Notification message (max 1000 chars)
    - **priority**: Priority level (0-10, default 0)
    
    Returns the created notification
    """
    try:
        notification = await notification_service.create_notification(
            db, notification_data
        )
        
        logger.info(f"Created notification {notification.id} for user {notification_data.user_id} by admin {current_user.id}")
        return notification
        
    except ValueError as e:
        logger.warning(f"Invalid data for notification creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating notification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create notification"
        )


@router.put("/{notification_id}", response_model=Notification)
async def update_notification(
    notification_id: uuid.UUID,
    notification_data: NotificationUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update a notification (mark as read/unread)
    
    - **notification_id**: UUID of the notification to update
    - **is_read**: Mark notification as read/unread
    - **read_at**: Timestamp when notification was read (optional)
    
    Returns the updated notification (only if it belongs to the current user)
    """
    try:
        notification = await notification_service.update_notification(
            db, notification_id, notification_data, current_user.id
        )
        
        if not notification:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found"
            )
        
        logger.info(f"Updated notification {notification_id} for user {current_user.id}")
        return notification
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid data for notification update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating notification {notification_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update notification"
        )


@router.delete("/{notification_id}", response_model=SuccessResponse)
async def delete_notification(
    notification_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a notification
    
    - **notification_id**: UUID of the notification to delete
    
    Returns success confirmation (only if notification belongs to current user or user is admin)
    """
    try:
        # Check if user is admin or notification owner
        is_admin = current_user.role.role == "admin"
        
        success = await notification_service.delete_notification(
            db, notification_id, current_user.id if not is_admin else None
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found"
            )
        
        logger.info(f"Deleted notification {notification_id} by user {current_user.id}")
        return SuccessResponse(
            message=f"Notification {notification_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting notification {notification_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete notification"
        )


@router.post("/{notification_id}/mark-read", response_model=Notification)
async def mark_notification_read(
    notification_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Mark a specific notification as read
    
    - **notification_id**: UUID of the notification to mark as read
    
    Returns the updated notification (only if it belongs to the current user)
    """
    try:
        notification = await notification_service.mark_notification_read(
            db, notification_id, current_user.id
        )
        
        if not notification:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found"
            )
        
        logger.info(f"Marked notification {notification_id} as read for user {current_user.id}")
        return notification
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark notification as read"
        )


@router.post("/mark-all-read", response_model=SuccessResponse)
async def mark_all_notifications_read(
    notification_type: Optional[NotificationType] = Query(None, description="Optional filter by notification type"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Mark all unread notifications as read for the current user
    
    - **notification_type**: Optional filter to only mark specific types as read
    
    Returns count of notifications marked as read
    """
    try:
        count = await notification_service.mark_all_notifications_read(
            db, current_user.id, notification_type
        )
        
        logger.info(f"Marked {count} notifications as read for user {current_user.id}")
        return SuccessResponse(
            message=f"Marked {count} notifications as read",
            data={"count": count, "notification_type": notification_type}
        )
        
    except Exception as e:
        logger.error(f"Error marking all notifications as read for user {current_user.id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark all notifications as read"
        )


@router.get("/admin/all", response_model=List[Notification])
async def get_all_notifications_admin(
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by user ID"),
    notification_type: Optional[NotificationType] = Query(None, description="Filter by notification type"),
    unread_only: bool = Query(False, description="Filter to unread notifications only"),
    skip: int = Query(0, ge=0, description="Number of notifications to skip"),
    limit: int = Query(100, ge=1, le=2000, description="Number of notifications to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Get all notifications (Admin only)
    
    - **user_id**: Optional filter by specific user
    - **notification_type**: Optional filter by notification type
    - **unread_only**: If true, only return unread notifications
    - **skip**: Number of notifications to skip (pagination)
    - **limit**: Maximum number of notifications to return
    
    Returns list of all notifications (with filters applied)
    """
    try:
        notifications = await notification_service.get_all_notifications_admin(
            db, user_id, notification_type, unread_only, skip, limit
        )
        
        logger.info(f"Retrieved {len(notifications)} notifications for admin {current_user.id}")
        return notifications
        
    except Exception as e:
        logger.error(f"Error retrieving all notifications: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve notifications"
        )


@router.get("/types/available")
async def get_notification_types(
    current_user: User = Depends(get_current_user)
):
    """
    Get available notification types
    
    Returns list of available notification types for filtering/creation
    """
    try:
        types = notification_service.get_notification_types()
        
        return {
            "types": types,
            "total": len(types),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving notification types: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve notification types"
        )
