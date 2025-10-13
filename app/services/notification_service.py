from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, desc, func
from typing import List, Optional
import uuid
from datetime import datetime

from app.models.models import Notification, User
from app.models.schemas import (
    NotificationCreate,
    NotificationUpdate,
    NotificationType
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class NotificationService:
    """Service for managing user notifications"""
    
    NOTIFICATION_TYPES = ["INFO", "WARNING", "ERROR", "SUCCESS", "SYSTEM", "PROMOTION", "REMINDER"]
    
    async def get_user_notifications(
        self, 
        db: AsyncSession, 
        user_id: uuid.UUID,
        unread_only: bool = False,
        notification_type: Optional[NotificationType] = None,
        skip: int = 0,
        limit: int = 50
    ) -> List[Notification]:
        """Get notifications for a specific user"""
        try:
            query = select(Notification).where(Notification.user_id == user_id)
            
            if unread_only:
                query = query.where(Notification.is_read == False)
            
            if notification_type:
                query = query.where(Notification.type == notification_type)
            
            query = query.order_by(desc(Notification.created_at))
            query = query.offset(skip).limit(limit)
            
            result = await db.execute(query)
            notifications = result.scalars().all()
            
            return notifications
            
        except Exception as e:
            logger.error(f"Error retrieving notifications for user {user_id}: {str(e)}")
            raise
    
    async def get_notification_by_id(
        self, 
        db: AsyncSession, 
        notification_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> Optional[Notification]:
        """Get notification by ID (only if it belongs to the user)"""
        try:
            result = await db.execute(
                select(Notification)
                .where(
                    and_(
                        Notification.id == notification_id,
                        Notification.user_id == user_id
                    )
                )
            )
            notification = result.scalar_one_or_none()
            return notification
            
        except Exception as e:
            logger.error(f"Error retrieving notification {notification_id}: {str(e)}")
            raise
    
    async def get_unread_count(
        self, 
        db: AsyncSession, 
        user_id: uuid.UUID
    ) -> int:
        """Get count of unread notifications for a user"""
        try:
            result = await db.execute(
                select(func.count(Notification.id))
                .where(
                    and_(
                        Notification.user_id == user_id,
                        Notification.is_read == False
                    )
                )
            )
            return result.scalar_one()
            
        except Exception as e:
            logger.error(f"Error getting unread count for user {user_id}: {str(e)}")
            raise
    
    async def create_notification(
        self, 
        db: AsyncSession, 
        notification_data: NotificationCreate
    ) -> Notification:
        """Create a new notification"""
        try:
            # Validate user exists
            user_result = await db.execute(
                select(User).where(User.id == notification_data.user_id)
            )
            if not user_result.scalar_one_or_none():
                raise ValueError(f"User {notification_data.user_id} not found")
            
            # Validate notification type
            if notification_data.type not in self.NOTIFICATION_TYPES:
                raise ValueError(f"Invalid notification type. Must be one of: {self.NOTIFICATION_TYPES}")
            
            # Validate priority
            if notification_data.priority < 0 or notification_data.priority > 10:
                raise ValueError("Priority must be between 0 and 10")
            
            # Create notification
            new_notification = Notification(
                user_id=notification_data.user_id,
                type=notification_data.type,
                title=notification_data.title,
                message=notification_data.message,
                priority=notification_data.priority,
                is_read=False
            )
            
            db.add(new_notification)
            await db.commit()
            await db.refresh(new_notification)
            
            logger.info(f"Created notification {new_notification.id} for user {notification_data.user_id}")
            return new_notification
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating notification: {str(e)}")
            raise
    
    async def update_notification(
        self, 
        db: AsyncSession, 
        notification_id: uuid.UUID,
        notification_data: NotificationUpdate,
        user_id: uuid.UUID
    ) -> Optional[Notification]:
        """Update a notification (only if it belongs to the user)"""
        try:
            # Get existing notification
            result = await db.execute(
                select(Notification)
                .where(
                    and_(
                        Notification.id == notification_id,
                        Notification.user_id == user_id
                    )
                )
            )
            notification = result.scalar_one_or_none()
            
            if not notification:
                return None
            
            # Update fields if provided
            update_data = {}
            
            if notification_data.is_read is not None:
                update_data['is_read'] = notification_data.is_read
                if notification_data.is_read:
                    update_data['read_at'] = notification_data.read_at or datetime.utcnow()
                else:
                    update_data['read_at'] = None
            
            if update_data:
                await db.execute(
                    update(Notification)
                    .where(Notification.id == notification_id)
                    .values(**update_data)
                )
                await db.commit()
                
                # Refresh the notification
                await db.refresh(notification)
            
            logger.info(f"Updated notification {notification_id}")
            return notification
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating notification {notification_id}: {str(e)}")
            raise
    
    async def delete_notification(
        self, 
        db: AsyncSession, 
        notification_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None
    ) -> bool:
        """Delete a notification (with optional user ownership check)"""
        try:
            conditions = [Notification.id == notification_id]
            
            if user_id:
                conditions.append(Notification.user_id == user_id)
            
            # Check if notification exists
            result = await db.execute(
                select(Notification).where(and_(*conditions))
            )
            notification = result.scalar_one_or_none()
            
            if not notification:
                return False
            
            # Delete the notification
            await db.execute(
                delete(Notification).where(and_(*conditions))
            )
            await db.commit()
            
            logger.info(f"Deleted notification {notification_id}")
            return True
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting notification {notification_id}: {str(e)}")
            raise
    
    async def mark_notification_read(
        self, 
        db: AsyncSession, 
        notification_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> Optional[Notification]:
        """Mark a specific notification as read"""
        try:
            update_data = NotificationUpdate(
                is_read=True,
                read_at=datetime.utcnow()
            )
            
            return await self.update_notification(
                db, notification_id, update_data, user_id
            )
            
        except Exception as e:
            logger.error(f"Error marking notification {notification_id} as read: {str(e)}")
            raise
    
    async def mark_all_notifications_read(
        self, 
        db: AsyncSession, 
        user_id: uuid.UUID,
        notification_type: Optional[NotificationType] = None
    ) -> int:
        """Mark all notifications as read for a user"""
        try:
            conditions = [
                Notification.user_id == user_id,
                Notification.is_read == False
            ]
            
            if notification_type:
                conditions.append(Notification.type == notification_type)
            
            # Count unread notifications first
            count_result = await db.execute(
                select(func.count(Notification.id))
                .where(and_(*conditions))
            )
            count = count_result.scalar_one()
            
            if count > 0:
                # Mark all as read
                await db.execute(
                    update(Notification)
                    .where(and_(*conditions))
                    .values(is_read=True, read_at=datetime.utcnow())
                )
                await db.commit()
            
            logger.info(f"Marked {count} notifications as read for user {user_id}")
            return count
            
        except Exception as e:
            await db.rollback()
            logger.error(f"Error marking all notifications as read for user {user_id}: {str(e)}")
            raise
    
    async def get_all_notifications_admin(
        self, 
        db: AsyncSession,
        user_id: Optional[uuid.UUID] = None,
        notification_type: Optional[NotificationType] = None,
        unread_only: bool = False,
        skip: int = 0,
        limit: int = 100
    ) -> List[Notification]:
        """Get all notifications (admin view)"""
        try:
            query = select(Notification)
            conditions = []
            
            if user_id:
                conditions.append(Notification.user_id == user_id)
            
            if notification_type:
                conditions.append(Notification.type == notification_type)
            
            if unread_only:
                conditions.append(Notification.is_read == False)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            query = query.order_by(desc(Notification.created_at))
            query = query.offset(skip).limit(limit)
            
            result = await db.execute(query)
            notifications = result.scalars().all()
            
            return notifications
            
        except Exception as e:
            logger.error(f"Error retrieving all notifications: {str(e)}")
            raise
    
    async def create_system_notification(
        self, 
        db: AsyncSession, 
        user_id: uuid.UUID,
        title: str,
        message: str,
        priority: int = 5
    ) -> Notification:
        """Create a system notification"""
        try:
            notification_data = NotificationCreate(
                user_id=user_id,
                type="SYSTEM",
                title=title,
                message=message,
                priority=priority
            )
            
            return await self.create_notification(db, notification_data)
            
        except Exception as e:
            logger.error(f"Error creating system notification for user {user_id}: {str(e)}")
            raise
    
    def get_notification_types(self) -> List[str]:
        """Get list of available notification types"""
        return self.NOTIFICATION_TYPES.copy()


# Global instance
notification_service = NotificationService()