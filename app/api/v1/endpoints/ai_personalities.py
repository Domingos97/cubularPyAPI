from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import uuid

from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_admin_user
from app.models.schemas import (
    AIPersonalityCreate,
    AIPersonalityUpdate,
    AIPersonality,
    SuccessResponse,
    ErrorResponse
)
from app.models.models import User
from app.services.ai_personality_service import ai_personality_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[AIPersonality])
async def get_ai_personalities(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all AI personalities
    
    - **skip**: Number of personalities to skip (for pagination)
    - **limit**: Maximum number of personalities to return
    
    Returns list of AI personalities available to the user
    """
    try:
        personalities = await ai_personality_service.get_all_personalities(
            db, skip=skip, limit=limit
        )
        
        logger.info(f"Retrieved {len(personalities)} AI personalities for user {current_user.id}")
        return personalities
        
    except Exception as e:
        logger.error(f"Error retrieving AI personalities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve AI personalities"
        )


@router.get("/{personality_id}", response_model=AIPersonality)
async def get_ai_personality_by_id(
    personality_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get specific AI personality by ID
    
    - **personality_id**: UUID of the AI personality to retrieve
    
    Returns the AI personality details
    """
    try:
        personality = await ai_personality_service.get_personality_by_id(
            db, personality_id
        )
        
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI personality not found"
            )
        
        logger.info(f"Retrieved AI personality {personality_id} for user {current_user.id}")
        return personality
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving AI personality {personality_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve AI personality"
        )


@router.post("/", response_model=AIPersonality, status_code=status.HTTP_201_CREATED)
async def create_ai_personality(
    personality_data: AIPersonalityCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Create a new AI personality (Admin only)
    
    - **name**: Name of the AI personality
    - **description**: Description of the personality's characteristics
    - **system_prompt**: Optional system prompt override
    - **detailed_analysis_prompt**: Prompt for detailed analysis
    - **suggestions_prompt**: Prompt for generating suggestions
    - **model_override**: Optional model override
    - **temperature_override**: Optional temperature override
    
    Returns the created AI personality
    """
    try:
        personality = await ai_personality_service.create_personality(
            db, personality_data, created_by=current_user.id
        )
        
        logger.info(f"Created AI personality {personality.id} by admin {current_user.id}")
        return personality
        
    except ValueError as e:
        logger.warning(f"Invalid data for AI personality creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating AI personality: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create AI personality"
        )


@router.put("/{personality_id}", response_model=AIPersonality)
async def update_ai_personality(
    personality_id: uuid.UUID,
    personality_data: AIPersonalityUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Update an existing AI personality (Admin only)
    
    - **personality_id**: UUID of the AI personality to update
    - **name**: Optional new name
    - **description**: Optional new description
    - **is_active**: Optional active status
    - **system_prompt**: Optional system prompt
    - **detailed_analysis_prompt**: Optional detailed analysis prompt
    - **suggestions_prompt**: Optional suggestions prompt
    - **model_override**: Optional model override
    - **temperature_override**: Optional temperature override
    
    Returns the updated AI personality
    """
    try:
        personality = await ai_personality_service.update_personality(
            db, personality_id, personality_data
        )
        
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI personality not found"
            )
        
        logger.info(f"Updated AI personality {personality_id} by admin {current_user.id}")
        return personality
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid data for AI personality update: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating AI personality {personality_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update AI personality"
        )


@router.delete("/{personality_id}", response_model=SuccessResponse)
async def delete_ai_personality(
    personality_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Delete an AI personality (Admin only)
    
    - **personality_id**: UUID of the AI personality to delete
    
    Returns success confirmation
    """
    try:
        success = await ai_personality_service.delete_personality(
            db, personality_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI personality not found"
            )
        
        logger.info(f"Deleted AI personality {personality_id} by admin {current_user.id}")
        return SuccessResponse(
            message=f"AI personality {personality_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting AI personality {personality_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete AI personality"
        )


@router.post("/{personality_id}/set-default", response_model=AIPersonality)
async def set_default_ai_personality(
    personality_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """
    Set an AI personality as default (Admin only)
    
    - **personality_id**: UUID of the AI personality to set as default
    
    Returns the updated AI personality
    """
    try:
        personality = await ai_personality_service.set_as_default(
            db, personality_id
        )
        
        if not personality:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI personality not found"
            )
        
        logger.info(f"Set AI personality {personality_id} as default by admin {current_user.id}")
        return personality
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting AI personality {personality_id} as default: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set AI personality as default"
        )


@router.get("/active/list", response_model=List[AIPersonality])
async def get_active_ai_personalities(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get only active AI personalities
    
    Returns list of active AI personalities available to the user
    """
    try:
        personalities = await ai_personality_service.get_active_personalities(db)
        
        logger.info(f"Retrieved {len(personalities)} active AI personalities for user {current_user.id}")
        return personalities
        
    except Exception as e:
        logger.error(f"Error retrieving active AI personalities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active AI personalities"
        )
