from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import get_current_regular_user, get_current_admin_user, SimpleUser
from app.models.schemas import (
    SuccessResponse,
    Plan as PlanSchema,
    UserPlan as UserPlanSchema
)
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/current")
async def get_current_plan(
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get current user's plan
    """
    try:
        # Get user's current plan from user_plans table
        user_plan_data = await db.execute_fetchrow(
            """
            SELECT up.id, up.user_id, up.plan_id, up.start_date, up.end_date, 
                   up.status, up.trial_ends_at, up.auto_renew, up.created_at, up.updated_at,
                   p.id as plan_db_id, p.name, p.description, p.price, 
                   p.billing, p.features, p.max_surveys, p.max_responses, 
                   p.priority_support, p.api_access, p.is_active as plan_is_active
            FROM user_plans up
            INNER JOIN plans p ON up.plan_id = p.id
            WHERE up.user_id = $1 AND up.status = 'active'
            ORDER BY up.start_date DESC
            LIMIT 1
            """,
            [current_user.id]
        )
        
        if not user_plan_data:
            # No plan found - create and assign a default free plan
            logger.info(f"No plan found for user {current_user.id}, assigning default free plan")
            
            # Get the free plan
            free_plan = await db.execute_fetchrow(
                "SELECT id, name, description, price, billing, features, max_surveys, max_responses, priority_support, api_access, is_active FROM plans WHERE name ILIKE '%free%' AND is_active = true LIMIT 1"
            )
            
            if not free_plan:
                # If no free plan exists, create a basic response
                from datetime import datetime, timedelta
                return {
                    "id": "temp-free",
                    "user_id": str(current_user.id),
                    "plan_id": "temp-free",
                    "started_at": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(days=365)).isoformat(),
                    "trial_ends_at": None,
                    "is_active": True,
                    "auto_renew": False,
                    "usage_count": 0,
                    "usage_limit": 1000,
                    "plan": {
                        "id": "temp-free",
                        "name": "Free",
                        "description": "Basic free plan",
                        "price": 0.0,
                        "billing_period": "monthly",
                        "features": ["Basic surveys", "Up to 1000 responses"],
                        "limits": {"monthly_usage": 1000},
                        "is_active": True
                    }
                }
            
            # Assign the free plan to the user
            from datetime import datetime, timedelta
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=30)
            
            await db.execute_command(
                """
                INSERT INTO user_plans (user_id, plan_id, start_date, end_date, status, auto_renew)
                VALUES ($1, $2, $3, $4, 'active', false)
                """,
                [current_user.id, free_plan["id"], start_date, end_date]
            )
            
            # Return the newly assigned plan
            return {
                "id": "new-assignment",
                "user_id": str(current_user.id),
                "plan_id": str(free_plan["id"]),
                "status": "active",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "trial_ends_at": None,
                "auto_renew": False,
                "payment_method_id": None,
                "stripe_subscription_id": None,
                "created_at": start_date.isoformat(),
                "updated_at": start_date.isoformat(),
                "usage_count": 0,
                "usage_limit": free_plan["max_responses"] or 1000,
                "plan": {
                    "id": str(free_plan["id"]),
                    "name": free_plan["name"],
                    "description": free_plan["description"],
                    "price": float(free_plan["price"]) if free_plan["price"] else 0.0,
                    "billing_period": free_plan["billing"] or "monthly",
                    "features": free_plan["features"] or [],
                    "limits": {
                        "max_surveys": free_plan["max_surveys"] or "unlimited",
                        "max_responses": free_plan["max_responses"] or "unlimited",
                        "max_users": 1,
                        "max_storage": "1GB",
                        "priority_support": free_plan["priority_support"] or False,
                        "api_access": free_plan["api_access"] or False,
                        "ai_analysis": True,
                        "custom_branding": False
                    },
                    "is_active": free_plan["is_active"]
                }
            }
        
        return {
            "id": str(user_plan_data["id"]),
            "user_id": str(user_plan_data["user_id"]),
            "plan_id": str(user_plan_data["plan_id"]),
            "status": user_plan_data["status"],
            "start_date": user_plan_data["start_date"].isoformat() if user_plan_data["start_date"] else None,
            "end_date": user_plan_data["end_date"].isoformat() if user_plan_data["end_date"] else None,
            "trial_ends_at": user_plan_data["trial_ends_at"].isoformat() if user_plan_data["trial_ends_at"] else None,
            "auto_renew": user_plan_data["auto_renew"],
            "payment_method_id": None,  # TODO: Implement payment methods
            "stripe_subscription_id": None,  # TODO: Implement Stripe integration
            "created_at": user_plan_data["created_at"].isoformat() if user_plan_data["created_at"] else None,
            "updated_at": user_plan_data["updated_at"].isoformat() if user_plan_data["updated_at"] else None,
            "usage_count": 0,  # TODO: Implement usage tracking
            "usage_limit": user_plan_data["max_responses"] or 10000,
            "plan": {
                "id": str(user_plan_data["plan_db_id"]),
                "name": user_plan_data["name"],
                "description": user_plan_data["description"],
                "price": float(user_plan_data["price"]) if user_plan_data["price"] else 0.0,
                "billing_period": user_plan_data["billing"] or "monthly",
                "features": user_plan_data["features"] or [],
                "limits": {
                    "max_surveys": user_plan_data["max_surveys"] or "unlimited",
                    "max_responses": user_plan_data["max_responses"] or "unlimited",
                    "max_users": 1,  # Default for now
                    "max_storage": "10GB",  # Default for now
                    "priority_support": user_plan_data["priority_support"] or False,
                    "api_access": user_plan_data["api_access"] or False,
                    "ai_analysis": True,  # Default for now
                    "custom_branding": False  # Default for now
                },
                "is_active": user_plan_data["plan_is_active"]
            }
        }
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
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get available plans for user
    """
    try:
        # Get all active plans
        plans_data = await db.execute_query(
            """
            SELECT id, name, description, price, billing, 
                   features, max_surveys, max_responses, priority_support, 
                   api_access, is_active, created_at, updated_at
            FROM plans
            WHERE is_active = true
            ORDER BY price ASC
            """
        )
        
        plans = []
        for plan_data in plans_data:
            plans.append({
                "id": str(plan_data["id"]),
                "name": plan_data["name"],
                "description": plan_data["description"],
                "price": float(plan_data["price"]) if plan_data["price"] else 0.0,
                "billing_period": plan_data["billing"] or "monthly",
                "features": plan_data["features"] or [],
                "limits": {
                    "max_surveys": plan_data["max_surveys"] or "unlimited",
                    "max_responses": plan_data["max_responses"] or "unlimited",
                    "max_users": 1,
                    "max_storage": "10GB",
                    "priority_support": plan_data["priority_support"] or False,
                    "api_access": plan_data["api_access"] or False,
                    "ai_analysis": True,
                    "custom_branding": False
                },
                "is_active": plan_data["is_active"],
                "created_at": plan_data["created_at"].isoformat() if plan_data["created_at"] else None,
                "updated_at": plan_data["updated_at"].isoformat() if plan_data["updated_at"] else None
            })
        
        return plans
    except Exception as e:
        logger.error(f"Get available plans error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available plans"
        )


@router.get("/usage")
async def get_plan_usage(
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get current user's plan usage statistics
    """
    try:
        # Get user's current plan usage
        usage_data = await db.execute_fetchrow(
            """
            SELECT up.start_date, up.end_date, up.status,
                   p.name as plan_name, p.billing, p.max_surveys, p.max_responses
            FROM user_plans up
            INNER JOIN plans p ON up.plan_id = p.id
            WHERE up.user_id = $1 AND up.status = 'active'
            ORDER BY up.start_date DESC
            LIMIT 1
            """,
            [current_user.id]
        )
        
        if not usage_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No plan usage data found"
            )
        
        # TODO: Implement proper usage tracking - for now return defaults
        usage_count = 0
        usage_limit = usage_data["max_responses"] or 10000  # Use plan's max_responses
        
        return {
            "usage_count": usage_count,
            "usage_limit": usage_limit,
            "usage_percentage": (usage_count / usage_limit) * 100 if usage_limit > 0 else 0,
            "remaining": max(0, usage_limit - usage_count) if usage_limit else None,
            "plan_name": usage_data["plan_name"],
            "billing_period": usage_data["billing"] or "monthly",
            "period_start": usage_data["start_date"].isoformat() if usage_data["start_date"] else None,
            "period_end": usage_data["end_date"].isoformat() if usage_data["end_date"] else None
        }
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
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Upgrade user to a new plan
    """
    try:
        import uuid
        plan_uuid = uuid.UUID(plan_id)
        
        # Check if plan exists and is active
        plan_exists = await db.execute_fetchrow(
            "SELECT id, name FROM plans WHERE id = $1 AND is_active = true",
            [plan_uuid]
        )
        
        if not plan_exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Plan not found or inactive"
            )
        
        # Deactivate current plan
        await db.execute_command(
            "UPDATE user_plans SET status = 'cancelled' WHERE user_id = $1 AND status = 'active'",
            [current_user.id]
        )
        
        # Create new plan assignment
        from datetime import datetime, timedelta
        new_plan_result = await db.execute_command(
            """
            INSERT INTO user_plans (user_id, plan_id, start_date, end_date, status, auto_renew)
            VALUES ($1, $2, $3, $4, 'active', false)
            """,
            [current_user.id, plan_uuid, datetime.utcnow(), datetime.utcnow() + timedelta(days=30)]
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
    current_user: SimpleUser = Depends(get_current_regular_user),
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Cancel user's current plan (downgrade to free)
    """
    try:
        # Deactivate current plan
        result = await db.execute_command(
            "UPDATE user_plans SET status = 'cancelled' WHERE user_id = $1 AND status = 'active'",
            [current_user.id]
        )
        
        # Get free plan and assign it
        free_plan = await db.execute_fetchrow(
            "SELECT id FROM plans WHERE name ILIKE '%free%' AND is_active = true LIMIT 1"
        )
        
        if free_plan:
            from datetime import datetime, timedelta
            await db.execute_command(
                """
                INSERT INTO user_plans (user_id, plan_id, start_date, end_date, status, auto_renew)
                VALUES ($1, $2, $3, $4, 'active', false)
                """,
                [current_user.id, free_plan["id"], datetime.utcnow(), datetime.utcnow() + timedelta(days=30)]
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