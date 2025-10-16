from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from app.services.lightweight_db_service import get_lightweight_db, LightweightDBService
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Public language endpoints (no authentication required)
@router.get("/enabled", response_model=Dict[str, Any])
async def get_enabled_languages(
    db: LightweightDBService = Depends(get_lightweight_db)
):
    """
    Get list of enabled/supported languages.
    Public endpoint for frontend language selection.
    """
    try:
        languages_data = await db.get_enabled_languages()
        
        return {
            "languages": [
                {
                    "code": lang["code"],
                    "name": lang["name"],
                    "native_name": lang["native_name"],
                    "is_rtl": False  # Default to False for now, can be added to DB later
                }
                for lang in languages_data
            ],
            "total_languages": len(languages_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting enabled languages: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch enabled languages"
        )