"""
Plan Service - Lightweight Database Implementation
===============================================
Direct database operations using LightweightDBService.
No SQLAlchemy overhead for better performance.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID

from app.services.lightweight_db_service import LightweightDBService
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PlanService:
    """Plan management service using LightweightDBService"""
    
    @staticmethod
    async def create_plan(db: LightweightDBService, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new plan"""
        # Convert PlanCreate schema to dict if needed
        if hasattr(plan_data, 'dict'):
            plan_data = plan_data.dict()
        return await db.create_plan(plan_data)
    
    @staticmethod
    async def get_plans(
        db: LightweightDBService,
        skip: int = 0,
        limit: int = 100,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all plans with pagination"""
        if active_only:
            return await db.get_active_plans()
        else:
            return await db.get_all_plans(skip=skip, limit=limit)
    
    @staticmethod
    async def get_plan(db: LightweightDBService, plan_id: UUID) -> Optional[Dict[str, Any]]:
        """Get plan by ID"""
        return await db.get_plan_by_id(str(plan_id))
    
    @staticmethod
    async def update_plan(
        db: LightweightDBService, 
        plan_id: UUID, 
        plan_update: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update an existing plan"""
        # Convert PlanUpdate schema to dict if needed
        if hasattr(plan_update, 'dict'):
            plan_update = plan_update.dict(exclude_unset=True)
        return await db.update_plan(str(plan_id), plan_update)
    
    @staticmethod
    async def delete_plan(db: LightweightDBService, plan_id: UUID) -> bool:
        """Delete a plan (soft delete by setting is_active to False)"""
        plan_data = {"is_active": False}
        result = await db.update_plan(str(plan_id), plan_data)
        return result is not None
    
    @staticmethod
    async def assign_plan_to_user(
        db: LightweightDBService, 
        user_id: UUID, 
        plan_id: UUID,
        admin_user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Assign a plan to a user"""
        return await db.assign_plan_to_user(str(user_id), str(plan_id))

    @staticmethod
    async def revoke_plan_from_user(
        db: LightweightDBService,
        user_id: UUID,
        user_plan_id: Optional[UUID] = None,
        admin_user_id: Optional[UUID] = None
    ) -> bool:
        """Revoke a plan from a user (sets status to cancelled)"""
        user_plan_id_str = str(user_plan_id) if user_plan_id else None
        return await db.revoke_plan_from_user(str(user_id), user_plan_id_str)
    
    @staticmethod
    async def remove_user_plan_access(
        db: LightweightDBService,
        user_id: UUID,
        admin_user_id: Optional[UUID] = None
    ) -> bool:
        """Completely remove plan access for a user (deletes the record)"""
        return await db.remove_user_plan_access(str(user_id))
    
    @staticmethod
    async def get_user_plan(db: LightweightDBService, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get the active plan for a user"""
        return await db.get_user_plan(str(user_id))
    
    @staticmethod
    async def cancel_user_plan(db: LightweightDBService, user_id: UUID) -> bool:
        """Cancel a user's active plan"""
        return await db.cancel_user_plan(str(user_id))
    
    @staticmethod
    async def get_plan_users(
        db: LightweightDBService, 
        plan_id: UUID,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all users with a specific plan"""
        return await db.get_plan_users(str(plan_id), active_only)

    @staticmethod
    async def check_plan_feature(
        db: LightweightDBService, 
        user_id: UUID, 
        feature_name: str
    ) -> bool:
        """Check if user's plan includes a specific feature"""
        return await db.check_plan_feature(str(user_id), feature_name)
    
    @staticmethod
    async def get_plan_usage_stats(db: LightweightDBService, plan_id: UUID) -> Dict[str, Any]:
        """Get usage statistics for a plan"""
        return await db.get_plan_usage_stats(str(plan_id))
    
    @staticmethod
    async def get_available_plans(db: LightweightDBService) -> List[Dict[str, Any]]:
        """Get all available (active) plans for public consumption"""
        return await db.get_active_plans()
    
    @staticmethod
    async def get_user_plan_usage(db: LightweightDBService, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get usage statistics for user's current plan"""
        user_plan = await db.get_user_plan(str(user_id))
        if not user_plan:
            return None
        
        # Get plan details
        plan = await db.get_plan_by_id(user_plan['plan_id'])
        if not plan:
            return None
        
        # Parse features if available
        features = {}
        if plan.get('features'):
            import json
            try:
                features = json.loads(plan['features'])
            except (json.JSONDecodeError, TypeError):
                features = {}
        
        usage = {
            "plan_name": plan['name'],
            "plan_id": user_plan['plan_id'],
            "start_date": user_plan.get('start_date'),
            "features": features,
            "usage_stats": {
                "surveys_created": 0,  # TODO: Calculate from user's surveys
                "chat_sessions": 0,    # TODO: Calculate from user's chat sessions
                "ai_queries": 0        # TODO: Calculate from user's AI usage
            }
        }
        
        return usage
    
    @staticmethod
    async def upgrade_user_plan(
        db: LightweightDBService,
        user_id: UUID,
        new_plan_id: UUID
    ) -> bool:
        """Upgrade user to a new plan"""
        try:
            # Check if the new plan exists
            new_plan = await db.get_plan_by_id(str(new_plan_id))
            if not new_plan:
                return False
            
            # This will handle deactivating the old plan and assigning the new one
            result = await db.assign_plan_to_user(str(user_id), str(new_plan_id))
            
            logger.info(f"User {user_id} upgraded to plan {new_plan_id}")
            return result is not None
            
        except Exception as e:
            logger.error(f"Failed to upgrade user {user_id} to plan {new_plan_id}: {e}")
            return False


# Service instance
plan_service = PlanService()