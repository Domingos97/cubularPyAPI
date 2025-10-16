"""
Utility script to update survey processing status
"""

import asyncio
from app.services.lightweight_db_service import LightweightDBService
from app.utils.logging import get_logger

logger = get_logger(__name__)

async def update_survey_status(survey_id: str, status: str):
    """
    Update the processing status of a survey
    
    Args:
        survey_id: UUID of the survey
        status: "pending" or "completed"
    """
    try:
        db = LightweightDBService()
        
        # Update the survey status
        update_query = """
        UPDATE surveys 
        SET processing_status = $1, updated_at = CURRENT_TIMESTAMP
        WHERE id = $2
        """
        
        await db.execute_command(update_query, [status, survey_id])
        logger.info(f"✅ Updated survey {survey_id} status to {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating survey status: {str(e)}")
        return False

async def mark_survey_completed(survey_id: str):
    """Mark a survey as completed (has responses)"""
    return await update_survey_status(survey_id, "completed")

async def mark_survey_pending(survey_id: str):
    """Mark a survey as pending (no responses yet)"""
    return await update_survey_status(survey_id, "pending")

async def auto_update_survey_statuses():
    """
    Automatically update survey statuses based on whether they have responses
    Note: Since there's no survey_responses table, this currently just reports status
    In the future, this could check for actual response data or file analysis
    """
    try:
        db = LightweightDBService()
        
        # Get all surveys and their current status
        surveys_query = """
        SELECT s.id, s.title, s.processing_status, s.category,
               s.created_at, s.number_participants
        FROM surveys s
        ORDER BY s.created_at DESC
        """
        
        surveys = await db.execute_query(surveys_query)
        
        pending_count = 0
        completed_count = 0
        
        for survey in surveys:
            survey_id = survey['id']
            current_status = survey['processing_status'] or 'pending'
            title = survey['title'] or 'Untitled Survey'
            
            if current_status == 'pending':
                pending_count += 1
            elif current_status == 'completed':
                completed_count += 1
                
            logger.info(f"� Survey '{title}' ({survey_id[:8]}...): {current_status}")
        
        logger.info(f"✅ Status check completed: {pending_count} pending, {completed_count} completed surveys")
        
        # For now, return 0 updates since we're not automatically changing statuses
        # In the future, this could implement logic to automatically mark surveys as completed
        # based on analysis or response data
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error in status check: {str(e)}")
        return 0

if __name__ == "__main__":
    """Run auto-update manually"""
    async def main():
        updates = await auto_update_survey_statuses()
        print(f"Updated {updates} survey statuses")
    
    asyncio.run(main())