from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update
from sqlalchemy.orm import selectinload

from app.models.models import Plan, UserPlan, User
from app.models.schemas import PlanCreate, PlanUpdate, UserPlanCreate


class PlanService:
    """Service for managing plans and user plan assignments."""
    
    @staticmethod
    async def create_plan(db: AsyncSession, plan_data: PlanCreate) -> Plan:
        """Create a new plan."""
        plan = Plan(**plan_data.dict())
        db.add(plan)
        await db.commit()
        await db.refresh(plan)
        return plan
    
    @staticmethod
    async def get_plan(db: AsyncSession, plan_id: UUID) -> Optional[Plan]:
        """Get a plan by ID."""
        stmt = select(Plan).where(Plan.id == plan_id)
        result = await db.execute(stmt)
        return result.scalars().first()
    
    @staticmethod
    async def get_plans(
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True
    ) -> List[Plan]:
        """Get all plans with pagination."""
        stmt = select(Plan)
        
        if active_only:
            stmt = stmt.where(Plan.is_active == True)
        
        stmt = stmt.offset(skip).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def update_plan(
        db: AsyncSession, 
        plan_id: UUID, 
        plan_update: PlanUpdate
    ) -> Optional[Plan]:
        """Update a plan."""
        stmt = select(Plan).where(Plan.id == plan_id)
        result = await db.execute(stmt)
        plan = result.scalars().first()
        
        if not plan:
            return None
        
        update_data = plan_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(plan, field, value)
        
        await db.commit()
        await db.refresh(plan)
        return plan
    
    @staticmethod
    async def delete_plan(db: AsyncSession, plan_id: UUID) -> bool:
        """Delete a plan (soft delete by setting is_active to False)."""
        stmt = update(Plan).where(Plan.id == plan_id).values(is_active=False)
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0
    
    @staticmethod
    async def assign_plan_to_user(
        db: AsyncSession, 
        user_id: UUID, 
        plan_id: UUID,
        admin_user_id: Optional[UUID] = None
    ) -> UserPlan:
        """Assign a plan to a user."""
        # Check if user already has an active plan
        stmt = select(UserPlan).where(
            UserPlan.user_id == user_id,
            UserPlan.is_active == True
        )
        result = await db.execute(stmt)
        existing_plan = result.scalars().first()
        
        if existing_plan:
            # Deactivate existing plan
            existing_plan.is_active = False
        
        # Create new user plan
        user_plan = UserPlan(
            user_id=user_id,
            plan_id=plan_id,
            is_active=True
        )
        db.add(user_plan)
        await db.commit()
        await db.refresh(user_plan)
        return user_plan

    @staticmethod
    async def revoke_plan_from_user(
        db: AsyncSession,
        user_id: UUID,
        user_plan_id: Optional[UUID] = None,
        admin_user_id: Optional[UUID] = None
    ) -> bool:
        """Revoke a plan from a user."""
        if user_plan_id:
            # Revoke specific plan
            stmt = update(UserPlan).where(
                UserPlan.id == user_plan_id,
                UserPlan.user_id == user_id
            ).values(is_active=False)
        else:
            # Revoke current active plan
            stmt = update(UserPlan).where(
                UserPlan.user_id == user_id,
                UserPlan.is_active == True
            ).values(is_active=False)
        
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0
    
    @staticmethod
    async def get_user_plan(db: AsyncSession, user_id: UUID) -> Optional[UserPlan]:
        """Get the active plan for a user."""
        stmt = select(UserPlan).options(
            selectinload(UserPlan.plan)
        ).where(
            UserPlan.user_id == user_id,
            UserPlan.is_active == True
        )
        result = await db.execute(stmt)
        return result.scalars().first()
    
    @staticmethod
    async def cancel_user_plan(db: AsyncSession, user_id: UUID) -> bool:
        """Cancel a user's active plan."""
        stmt = update(UserPlan).where(
            UserPlan.user_id == user_id,
            UserPlan.is_active == True
        ).values(is_active=False)
        result = await db.execute(stmt)
        await db.commit()
        return result.rowcount > 0
    
    @staticmethod
    async def get_plan_users(
        db: AsyncSession, 
        plan_id: UUID,
        active_only: bool = True
    ) -> List[UserPlan]:
        """Get all users with a specific plan."""
        stmt = select(UserPlan).options(
            selectinload(UserPlan.user)
        ).where(UserPlan.plan_id == plan_id)
        
        if active_only:
            stmt = stmt.where(UserPlan.is_active == True)
        
        result = await db.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def check_plan_feature(
        db: AsyncSession, 
        user_id: UUID, 
        feature_name: str
    ) -> bool:
        """Check if user's plan includes a specific feature."""
        user_plan = await PlanService.get_user_plan(db, user_id)
        
        if not user_plan or not user_plan.plan:
            return False
        
        # Parse features JSON string
        import json
        try:
            features = json.loads(user_plan.plan.features or "{}")
            return features.get(feature_name, False)
        except (json.JSONDecodeError, AttributeError):
            return False
    
    @staticmethod
    async def get_plan_usage_stats(db: AsyncSession, plan_id: UUID) -> dict:
        """Get usage statistics for a plan."""
        # Count total users
        stmt = select(UserPlan).where(UserPlan.plan_id == plan_id)
        result = await db.execute(stmt)
        total_users = len(result.scalars().all())
        
        # Count active users
        stmt = select(UserPlan).where(
            UserPlan.plan_id == plan_id,
            UserPlan.is_active == True
        )
        result = await db.execute(stmt)
        active_users = len(result.scalars().all())
        
        return {
            "plan_id": plan_id,
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users
        }
    
    @staticmethod
    async def get_available_plans(db: AsyncSession) -> List[Plan]:
        """Get all available plans for users to choose from"""
        stmt = select(Plan).where(Plan.is_active == True)
        result = await db.execute(stmt)
        return result.scalars().all()
    
    @staticmethod
    async def get_user_plan_usage(db: AsyncSession, user_id: UUID) -> Optional[dict]:
        """Get usage statistics for user's current plan"""
        user_plan = await PlanService.get_user_plan(db, user_id)
        if not user_plan:
            return None
        
        # Calculate usage based on plan features
        # This is a simplified implementation
        usage = {
            "plan_name": user_plan.plan.name,
            "plan_id": str(user_plan.plan.id),
            "assigned_at": user_plan.assigned_at.isoformat() if user_plan.assigned_at else None,
            "features": user_plan.plan.features if hasattr(user_plan.plan, 'features') else {},
            "usage_stats": {
                "surveys_created": 0,  # TODO: Calculate from user's surveys
                "chat_sessions": 0,    # TODO: Calculate from user's chat sessions
                "ai_queries": 0        # TODO: Calculate from user's AI usage
            }
        }
        
        return usage
    
    @staticmethod
    async def upgrade_user_plan(
        db: AsyncSession, 
        user_id: UUID, 
        new_plan_id: UUID
    ) -> bool:
        """Upgrade user to a new plan"""
        try:
            # Check if the new plan exists
            new_plan = await PlanService.get_plan(db, new_plan_id)
            if not new_plan:
                return False
            
            # Cancel current plan if exists
            await PlanService.cancel_user_plan(db, user_id)
            
            # Assign new plan
            user_plan = await PlanService.assign_plan_to_user(db, user_id, new_plan_id)
            
            return user_plan is not None
            
        except Exception:
            await db.rollback()
            return False


# Service instance
plan_service = PlanService()