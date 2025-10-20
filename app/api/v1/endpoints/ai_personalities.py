from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import uuid

from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.core.lightweight_dependencies import get_current_regular_user, get_current_admin_user, SimpleUser
from app.models.schemas import (
    AIPersonalityCreate,
    AIPersonalityUpdate,
    AIPersonality,
    SuccessResponse
)
from app.services.ai_personality_service import ai_personality_service
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=List[AIPersonality])
async def get_ai_personalities(
    skip: int = 0,
    limit: int = 100,
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get all AI personalities
    
    - **skip**: Number of personalities to skip (for pagination)
    - **limit**: Maximum number of personalities to return
    
    Returns list of AI personalities available to the user
    """
    try:
        personalities_data = await db.get_all_ai_personalities(skip=skip, limit=limit)
        
        # Convert to response format
        personalities = [
            AIPersonality(
                id=str(p["id"]),
                name=p["name"],
                description=p["description"],
                detailed_analysis_prompt=p["detailed_analysis_prompt"] or "",
                suggestions_prompt=p["suggestions_prompt"] or "",
                created_by=str(p.get("created_by")) if p.get("created_by") else None,
                model_override=None,
                temperature_override=None,
                is_default=p.get("is_default", False),
                is_active=p.get("is_active", True),
                created_at=p["created_at"].isoformat() if p.get("created_at") else "",
                updated_at=p["updated_at"].isoformat() if p.get("updated_at") else ""
            )
            for p in personalities_data
        ]
        
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Get specific AI personality by ID
    
    - **personality_id**: UUID of the AI personality to retrieve
    
    Returns the AI personality details
    """
    try:
        personality_data = await db.get_ai_personality(str(personality_id))
        
        if not personality_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI personality not found"
            )
        
        personality = AIPersonality(
            id=str(personality_data["id"]),
            name=personality_data["name"],
            description=personality_data["description"],
            detailed_analysis_prompt=personality_data["detailed_analysis_prompt"] or "",
            suggestions_prompt=personality_data["suggestions_prompt"] or "",
            created_by=str(personality_data.get("created_by")) if personality_data.get("created_by") else None,
            model_override=None,
            temperature_override=None,
            is_default=personality_data.get("is_default", False),
            is_active=personality_data.get("is_active", True),
            created_at=personality_data["created_at"].isoformat() if personality_data.get("created_at") else "",
            updated_at=personality_data["updated_at"].isoformat() if personality_data.get("updated_at") else ""
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
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
        personality_dict = personality_data.dict()
        personality_dict['created_by'] = current_user.id
        
        personality_result = await db.create_ai_personality(personality_dict)
        
        personality = AIPersonality(
            id=str(personality_result["id"]),
            name=personality_result["name"],
            description=personality_result["description"],
            detailed_analysis_prompt=personality_result["detailed_analysis_prompt"] or "",
            suggestions_prompt=personality_result["suggestions_prompt"] or "",
            model_override=None,
            temperature_override=None,
            is_default=personality_result.get("is_default", False),
            is_active=personality_result.get("is_active", True),
            created_at=personality_result["created_at"].isoformat() if personality_result.get("created_at") else "",
            updated_at=personality_result["updated_at"].isoformat() if personality_result.get("updated_at") else ""
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
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
        personality_dict = personality_data.dict(exclude_unset=True)
        personality_result = await db.update_ai_personality(str(personality_id), personality_dict)
        
        if not personality_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI personality not found"
            )
        
        personality = AIPersonality(
            id=str(personality_result["id"]),
            name=personality_result["name"],
            description=personality_result["description"],
            detailed_analysis_prompt=personality_result["detailed_analysis_prompt"] or "",
            suggestions_prompt=personality_result["suggestions_prompt"] or "",
            model_override=None,
            temperature_override=None,
            is_default=personality_result.get("is_default", False),
            is_active=personality_result.get("is_active", True),
            created_at=personality_result["created_at"].isoformat() if personality_result.get("created_at") else "",
            updated_at=personality_result["updated_at"].isoformat() if personality_result.get("updated_at") else ""
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Delete an AI personality (Admin only)
    
    - **personality_id**: UUID of the AI personality to delete
    
    Returns success confirmation
    """
    try:
        success = await db.delete_ai_personality(str(personality_id))
        
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_admin_user)
):
    """
    Set an AI personality as default (Admin only)
    
    - **personality_id**: UUID of the AI personality to set as default
    
    Returns the updated AI personality
    """
    try:
        personality_result = await db.set_ai_personality_as_default(str(personality_id))
        
        if not personality_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="AI personality not found"
            )
        
        personality = AIPersonality(
            id=str(personality_result["id"]),
            name=personality_result["name"],
            description=personality_result["description"],
            detailed_analysis_prompt=personality_result["detailed_analysis_prompt"] or "",
            suggestions_prompt=personality_result["suggestions_prompt"] or "",
            model_override=None,
            temperature_override=None,
            is_default=personality_result.get("is_default", False),
            is_active=personality_result.get("is_active", True),
            created_at=personality_result["created_at"].isoformat() if personality_result.get("created_at") else "",
            updated_at=personality_result["updated_at"].isoformat() if personality_result.get("updated_at") else ""
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
    db: LightweightDBService = Depends(get_lightweight_db),
    current_user: SimpleUser = Depends(get_current_regular_user)
):
    """
    Get only active AI personalities
    
    Returns list of active AI personalities available to the user
    """
    try:
        personalities_data = await db.get_active_ai_personalities()
        
        personalities = [
            AIPersonality(
                id=str(p["id"]),
                name=p["name"],
                description=p["description"],
                detailed_analysis_prompt=p["detailed_analysis_prompt"] or "",
                suggestions_prompt=p["suggestions_prompt"] or "",
                created_by=str(p.get("created_by")) if p.get("created_by") else None,
                model_override=None,
                temperature_override=None,
                is_default=p.get("is_default", False),
                is_active=p.get("is_active", True),
                created_at=p["created_at"].isoformat() if p.get("created_at") else "",
                updated_at=p["updated_at"].isoformat() if p.get("updated_at") else ""
            )
            for p in personalities_data
        ]
        
        logger.info(f"Retrieved {len(personalities)} active AI personalities for user {current_user.id}")
        return personalities
        
    except Exception as e:
        logger.error(f"Error retrieving active AI personalities: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active AI personalities"
        )
