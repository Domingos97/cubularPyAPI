from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from datetime import datetime
import uuid

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import get_current_regular_user, get_current_admin_user, SimpleUser
from app.models.schemas import (
    Notification,
    NotificationType
)
from app.utils.logging import get_logger
from pydantic import BaseModel
from typing import Dict, Any

logger = get_logger(__name__)
router = APIRouter()


class CreateNotificationRequest(BaseModel):
    type: str
    title: str
    message: str
    priority: int = 1
    metadata: Dict[str, Any] = {}


class UpdateNotificationRequest(BaseModel):
    status: Optional[str] = None
    admin_response: Optional[str] = None


class NotificationListResponse(BaseModel):
    data: List[Notification]
    unread_count: Optional[int] = None
    total: int


class MarkReadResponse(BaseModel):
    message: str
    success: bool
    count: Optional[int] = None


@router.post("/", status_code=201)
async def create_notification(
    request: CreateNotificationRequest,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Create a new notification for the current user
    """
    try:
        notification_id = await db.create_notification(
            user_id=current_user.id,
            title=request.title,
            message=request.message,
            notification_type=request.type,
            priority=request.priority,
            metadata=request.metadata
        )
        
        logger.info(f"Created notification {notification_id} for user {current_user.id}")
        
        return {
            "id": notification_id,
            "message": "Notification created successfully",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to create notification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create notification"
        )


@router.get("/my", response_model=NotificationListResponse)
async def get_my_notifications(
    unread_only: bool = Query(False, description="Filter to unread notifications only"),
    notification_type: Optional[NotificationType] = Query(None, description="Filter by notification type"),
    skip: int = Query(0, ge=0, description="Number of notifications to skip"),
    limit: int = Query(50, ge=1, le=100, description="Number of notifications to return"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
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
        notifications_data = await db.get_user_notifications(
            user_id=current_user.id,
            unread_only=unread_only,
            notification_type=notification_type.value if notification_type else None,
            skip=skip,
            limit=limit
        )
        
        # Convert to Notification objects
        notifications = []
        for notif_data in notifications_data:
            notifications.append(Notification(
                id=str(notif_data["id"]),
                user_id=str(notif_data["user_id"]),
                title=notif_data["title"],
                message=notif_data["message"],
                type=notif_data["type"],
                is_read=notif_data["is_read"],
                status=notif_data["status"],
                priority=notif_data["priority"],
                admin_response=notif_data.get("admin_response"),
                responded_by=str(notif_data["responded_by"]) if notif_data.get("responded_by") else None,
                responded_at=notif_data.get("responded_at"),
                created_at=notif_data["created_at"],
                updated_at=notif_data["updated_at"]
            ))
        
        logger.info(f"Retrieved {len(notifications)} notifications for user {current_user.id} via /my endpoint")
        
        # Calculate unread count
        unread_count = sum(1 for notif in notifications_data if not notif.get("is_read", False))
        
        # Return in expected frontend format
        return {
            "data": notifications,
            "unread_count": unread_count,
            "total": len(notifications)
        }
        
    except Exception as e:
        logger.error(f"Failed to get user notifications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user notifications"
        )


@router.get("/admin/all", response_model=NotificationListResponse)
async def get_all_notifications_admin(
    unread_only: bool = Query(False, description="Filter to unread notifications only"),
    notification_type: Optional[NotificationType] = Query(None, description="Filter by notification type"),
    skip: int = Query(0, ge=0, description="Number of notifications to skip"),
    limit: int = Query(1000, ge=1, le=1000, description="Number of notifications to return"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_admin: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get all notifications for admin users
    
    - **unread_only**: If true, only return unread notifications
    - **notification_type**: Filter by specific notification type
    - **skip**: Number of notifications to skip (pagination)
    - **limit**: Maximum number of notifications to return
    
    Returns list of all notifications (admin only)
    """
    try:
        notifications_data = await db.get_all_notifications(
            unread_only=unread_only,
            notification_type=notification_type.value if notification_type else None,
            skip=skip,
            limit=limit
        )
        
        # Convert to Notification objects
        notifications = []
        for notif_data in notifications_data:
            notifications.append(Notification(
                id=str(notif_data["id"]),
                user_id=str(notif_data["user_id"]),
                title=notif_data["title"],
                message=notif_data["message"],
                type=notif_data["type"],
                is_read=notif_data["is_read"],
                status=notif_data["status"],
                priority=notif_data["priority"],
                admin_response=notif_data.get("admin_response"),
                responded_by=str(notif_data["responded_by"]) if notif_data.get("responded_by") else None,
                responded_at=notif_data.get("responded_at"),
                created_at=notif_data["created_at"],
                updated_at=notif_data["updated_at"]
            ))
        
        logger.info(f"Retrieved {len(notifications)} notifications for admin {current_admin.id}")
        
        # Return in expected frontend format
        return {
            "data": notifications,
            "total": len(notifications)
        }
        
    except Exception as e:
        logger.error(f"Failed to get all notifications: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get all notifications"
        )


@router.get("/admin/user/{user_id}", response_model=NotificationListResponse)
async def get_user_notifications_admin(
    user_id: str,
    unread_only: bool = Query(False, description="Filter to unread notifications only"),
    notification_type: Optional[NotificationType] = Query(None, description="Filter by notification type"),
    skip: int = Query(0, ge=0, description="Number of notifications to skip"),
    limit: int = Query(1000, ge=1, le=1000, description="Number of notifications to return"),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_admin: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get notifications for a specific user (admin only)
    
    - **user_id**: The ID of the user whose notifications to retrieve
    - **unread_only**: If true, only return unread notifications
    - **notification_type**: Filter by specific notification type
    - **skip**: Number of notifications to skip (pagination)
    - **limit**: Maximum number of notifications to return
    
    Returns list of notifications for the specified user (admin only)
    """
    try:
        notifications_data = await db.get_user_notifications(
            user_id=user_id,
            unread_only=unread_only,
            notification_type=notification_type.value if notification_type else None,
            skip=skip,
            limit=limit
        )
        
        # Convert to Notification objects
        notifications = []
        for notif_data in notifications_data:
            notifications.append(Notification(
                id=str(notif_data["id"]),
                user_id=str(notif_data["user_id"]),
                title=notif_data["title"],
                message=notif_data["message"],
                type=notif_data["type"],
                is_read=notif_data["is_read"],
                status=notif_data["status"],
                priority=notif_data["priority"],
                admin_response=notif_data.get("admin_response"),
                responded_by=str(notif_data["responded_by"]) if notif_data.get("responded_by") else None,
                responded_at=notif_data.get("responded_at"),
                created_at=notif_data["created_at"],
                updated_at=notif_data["updated_at"]
            ))
        
        logger.info(f"Admin {current_admin.id} retrieved {len(notifications)} notifications for user {user_id}")
        
        # Return in expected frontend format
        return {
            "data": notifications,
            "total": len(notifications)
        }
        
    except Exception as e:
        logger.error(f"Failed to get user notifications for admin: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user notifications"
        )


# NOTE: The generic update endpoint is declared after the more specific
# endpoints below (/mark-all-read and /{notification_id}/read). Placing
# specific routes first prevents FastAPI from matching those literal
# paths against the generic `{notification_id}` path (which caused 422s).


@router.put("/{notification_id}/read", response_model=MarkReadResponse)
async def mark_notification_read(
    notification_id: str,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Mark a specific notification as read for the current user
    
    - **notification_id**: The ID of the notification to mark as read
    
    Returns success response
    """
    try:
        # Update the notification to mark as read
        success = await db.mark_notification_read(
            notification_id=notification_id,
            user_id=current_user.id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found or access denied"
            )
        
        logger.info(f"User {current_user.id} marked notification {notification_id} as read")
        
        return MarkReadResponse(
            message="Notification marked as read successfully",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark notification {notification_id} as read: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark notification as read"
        )


@router.put("/mark-all-read", response_model=MarkReadResponse)
async def mark_all_notifications_read(
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Mark all notifications as read for the current user
    
    Returns success response with count of marked notifications
    """
    try:
        # Update all unread notifications for the user
        count = await db.mark_all_notifications_read(user_id=current_user.id)
        
        logger.info(f"User {current_user.id} marked {count} notifications as read")
        
        return MarkReadResponse(
            message=f"Marked {count} notifications as read successfully",
            success=True,
            count=count
        )
        
    except Exception as e:
        logger.error(f"Failed to mark all notifications as read for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark all notifications as read"
        )


@router.put("/{notification_id}")
async def update_notification(
    notification_id: str,
    request: UpdateNotificationRequest,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_admin: SimpleUser = Depends(get_current_admin_user)
):
    """
    Update a notification (admin only)
    
    - **notification_id**: The ID of the notification to update
    - **status**: New status for the notification (optional)
    - **admin_response**: Admin response to the notification (optional)
    
    Returns updated notification (admin only)
    """
    try:
        # Prepare update data
        update_data = {}
        if request.status is not None:
            update_data["status"] = request.status
        if request.admin_response is not None:
            update_data["admin_response"] = request.admin_response
            update_data["responded_by"] = current_admin.id
            update_data["responded_at"] = datetime.utcnow()
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No update data provided"
            )
        
        # Update the notification
        success = await db.update_notification(
            notification_id=notification_id,
            **update_data
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification not found"
            )
        
        logger.info(f"Admin {current_admin.id} updated notification {notification_id}")
        
        return {
            "message": "Notification updated successfully",
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update notification {notification_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update notification"
        )