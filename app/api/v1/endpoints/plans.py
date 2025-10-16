from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import get_current_user, get_current_admin_user, SimpleUser
from app.models.schemas import (
    Plan as PlanSchema,
    PlanCreate,
    PlanUpdate,
    UserPlan as UserPlanSchema
)
from app.services.plan_service import PlanService

router = APIRouter()


@router.post("/", response_model=PlanSchema)
async def create_plan(
    plan_data: PlanCreate,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """Create a new plan. Admin only."""
    try:
        plan_dict = plan_data.dict()
        plan_result = await db.create_plan(plan_dict)
        return PlanSchema(**plan_result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create plan: {str(e)}"
        )


@router.get("/", response_model=List[PlanSchema])
async def get_plans(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True),
    db: LightweightDBService = Depends(get_lightweight_db),
):
    """Get all plans with pagination."""
    try:
        if active_only:
            plans_data = await db.get_active_plans()
        else:
            plans_data = await db.get_all_plans(skip=skip, limit=limit)
        
        # Convert to response format
        plans = [
            PlanSchema(
                id=p["id"],
                name=p["name"],
                display_name=p["display_name"],
                description=p["description"],
                price=p["price"],
                currency=p["currency"],
                billing=p["billing"],
                features=p["features"] if p.get("features") else [],
                max_surveys=p.get("max_surveys"),
                max_responses=p.get("max_responses"),
                priority_support=p.get("priority_support", False),
                api_access=p.get("api_access", False),
                is_active=p.get("is_active", True),
                created_at=p["created_at"],
                updated_at=p["updated_at"]
            )
            for p in plans_data
        ]
        
        return plans
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get plans: {str(e)}"
        )


@router.get("/available", response_model=List[PlanSchema])
async def get_available_plans(
    db: LightweightDBService = Depends(get_lightweight_db),
):
    """Get available plans for user - matches TypeScript API pattern"""
    try:
        plans_data = await db.get_active_plans()
        
        # Convert to response format
        plans = [
            PlanSchema(
                id=p["id"],
                name=p["name"],
                display_name=p["display_name"],
                description=p["description"],
                price=p["price"],
                currency=p["currency"],
                billing=p["billing"],
                features=p["features"] if isinstance(p["features"], list) else [],
                max_surveys=p["max_surveys"],
                max_responses=p["max_responses"],
                priority_support=p["priority_support"],
                api_access=p["api_access"],
                is_active=p["is_active"],
                created_at=p["created_at"],
                updated_at=p["updated_at"]
            )
            for p in plans_data
        ]
        
        return plans
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available plans"
        )


@router.get("/{plan_id}", response_model=PlanSchema)
async def get_plan(
    plan_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
):
    """Get a specific plan by ID."""
    plan = await PlanService.get_plan(db, plan_id)
    
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan not found"
        )
    
    return plan


@router.put("/{plan_id}", response_model=PlanSchema)
async def update_plan(
    plan_id: UUID,
    plan_update: PlanUpdate,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """Update a plan. Admin only."""
    try:
        plan_dict = plan_update.dict(exclude_unset=True)
        plan_result = await db.update_plan(str(plan_id), plan_dict)
        
        if not plan_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found"
            )
        
        return PlanSchema(**plan_result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update plan: {str(e)}"
        )


@router.delete("/{plan_id}")
async def delete_plan(
    plan_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """Delete a plan (soft delete). Admin only."""
    success = await PlanService.delete_plan(db, plan_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Plan not found"
        )
    
    return {"message": "Plan deleted successfully"}


@router.post("/users/{user_id}/plan/{plan_id}", response_model=UserPlanSchema)
async def assign_plan_to_user(
    user_id: UUID,
    plan_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """Assign a plan to a user. Admin only."""
    try:
        user_plan = await PlanService.assign_plan_to_user(db, user_id, plan_id)
        return user_plan
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to assign plan: {str(e)}"
        )


@router.get("/users/{user_id}/plan", response_model=UserPlanSchema)
async def get_user_plan(
    user_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_user)
):
    """Get the active plan for a user."""
    # Users can check their own plan, admins can check anyone's
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view user plan"
        )
    
    user_plan = await PlanService.get_user_plan(db, user_id)
    
    if not user_plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User has no active plan"
        )
    
    return user_plan


@router.delete("/users/{user_id}/plan")
async def cancel_user_plan(
    user_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_user)
):
    """Cancel a user's active plan."""
    # Users can cancel their own plan, admins can cancel anyone's
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to cancel user plan"
        )
    
    success = await PlanService.cancel_user_plan(db, user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User has no active plan to cancel"
        )
    
    return {"message": "Plan cancelled successfully"}


@router.get("/{plan_id}/users", response_model=List[UserPlanSchema])
async def get_plan_users(
    plan_id: UUID,
    active_only: bool = Query(True),
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """Get all users with a specific plan. Admin only."""
    users = await PlanService.get_plan_users(db, plan_id, active_only)
    return users


@router.get("/users/{user_id}/plan/feature/{feature_name}")
async def check_user_plan_feature(
    user_id: UUID,
    feature_name: str,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_user)
):
    """Check if user's plan includes a specific feature."""
    # Users can check their own features, admins can check anyone's
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to check user plan features"
        )
    
    has_feature = await PlanService.check_plan_feature(db, user_id, feature_name)
    
    return {
        "user_id": user_id,
        "feature_name": feature_name,
        "has_feature": has_feature
    }


@router.get("/{plan_id}/stats")
async def get_plan_stats(
    plan_id: UUID,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """Get usage statistics for a plan. Admin only."""
    stats = await PlanService.get_plan_usage_stats(db, plan_id)
    return stats


@router.get("/my-plan", response_model=UserPlanSchema)
async def get_my_plan(
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_user)
):
    """Get the current user's active plan."""
    user_plan = await PlanService.get_user_plan(db, current_user.id)
    
    if not user_plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="You have no active plan"
        )
    
    return user_plan


@router.get("/my-plan/feature/{feature_name}")
async def check_my_plan_feature(
    feature_name: str,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_user)
):
    """Check if current user's plan includes a specific feature."""
    has_feature = await PlanService.check_plan_feature(db, current_user.id, feature_name)
    
    return {
        "user_id": current_user.id,
        "feature_name": feature_name,
        "has_feature": has_feature
    }