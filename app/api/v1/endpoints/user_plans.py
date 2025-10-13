from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.models.schemas import (
    SuccessResponse,
    Plan as PlanSchema,
    UserPlan as UserPlanSchema
)
from app.models.models import User
from app.services.plan_service import plan_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/current", response_model=UserPlanSchema)
async def get_current_plan(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user's plan
    """
    try:
        user_plan = await plan_service.get_user_plan(db, current_user.id)
        if not user_plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No plan assigned to user"
            )
        return UserPlanSchema.from_orm(user_plan)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current plan error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get current plan"
        )


@router.get("/available", response_model=List[PlanSchema])
async def get_available_plans(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get available plans for user
    """
    try:
        plans = await plan_service.get_available_plans(db)
        return [PlanSchema.from_orm(plan) for plan in plans]
    except Exception as e:
        logger.error(f"Get available plans error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available plans"
        )


@router.get("/usage")
async def get_plan_usage(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current user's plan usage statistics
    """
    try:
        usage = await plan_service.get_user_plan_usage(db, current_user.id)
        if not usage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No plan usage data found"
            )
        return usage
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get plan usage error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get plan usage"
        )


@router.post("/upgrade", response_model=SuccessResponse)
async def upgrade_plan(
    plan_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upgrade user to a new plan
    """
    try:
        import uuid
        plan_uuid = uuid.UUID(plan_id)
        
        success = await plan_service.upgrade_user_plan(
            db, 
            current_user.id, 
            plan_uuid
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to upgrade plan"
            )
        
        logger.info(f"User {current_user.email} upgraded to plan {plan_id}")
        return SuccessResponse(message="Plan upgraded successfully")
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid plan ID format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan upgrade error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upgrade plan"
        )


@router.post("/cancel", response_model=SuccessResponse)
async def cancel_plan(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel user's current plan (downgrade to free)
    """
    try:
        success = await plan_service.cancel_user_plan(db, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cancel plan"
            )
        
        logger.info(f"User {current_user.email} cancelled their plan")
        return SuccessResponse(message="Plan cancelled successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan cancellation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel plan"
        )