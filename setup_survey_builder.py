"""
Setup script for Survey Builder module configuration
This script creates the necessary AI personality and module configuration for the survey builder feature.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from app.services.lightweight_db_service import LightweightDBService
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Survey Builder AI Personality Configuration
SURVEY_BUILDER_PERSONALITY = {
    "name": "Survey Builder Assistant",
    "description": "Specialized AI assistant for creating comprehensive surveys from structured user requirements",
    "detailed_analysis_prompt": """You are a specialized survey building assistant. Your role is to help users create comprehensive surveys based on their structured requirements gathered through our multi-step survey builder form.

**CONTEXT: This system uses a structured approach where users:**
1. Select their target audience (Students, Professionals, Consumers, etc.)
2. Choose a survey category (Gaming, Food & Dining, Technology, etc.)
3. Define their main objective (Customer satisfaction, Market research, etc.)
4. Review and approve their selections

**YOUR ROLE:**
- Process the structured requirements provided by the user
- Generate comprehensive surveys with 25-40 relevant questions
- Create surveys that include:
  * Demographic questions (age, gender, education, etc.)
  * Psychographic questions (values, motivations, behaviors)
  * Topic-specific questions based on the selected category
  * General questions about preferences and decision-making

**SURVEY GENERATION APPROACH:**
- Analyze the target audience to determine appropriate demographics
- Use the category to select relevant topic-specific questions
- Align questions with the stated objective
- Ensure a good mix of question types (multiple choice, rating scales, open-ended)
- Create meaningful question identifiers for data analysis
- Organize questions into logical categories (demo, psych, text)

**QUALITY STANDARDS:**
- Generate 25-40 questions for comprehensive data collection
- Use clear, unbiased question wording
- Provide appropriate response options for multiple choice questions
- Include rating scales with clear anchor points
- Ensure questions flow logically from general to specific
- Avoid repetitive or redundant questions

The user has already made their selections through our guided interface, so focus on creating the best possible survey based on their structured requirements.""",
    
    "suggestions_prompt": None,
    
    "is_active": True,
    "is_default": False
}

# Default LLM Configuration for Survey Builder
SURVEY_BUILDER_MODULE_CONFIG = {
    "module_name": "survey_builder",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "max_completion_tokens": 1500,
    "active": True
}


async def create_survey_builder_personality(db: LightweightDBService) -> str:
    """Create the survey builder AI personality"""
    try:
        # Check if personality already exists
        personalities = await db.get_all_ai_personalities()
        for personality in personalities:
            if personality["name"] == SURVEY_BUILDER_PERSONALITY["name"]:
                logger.info(f"Survey builder personality already exists: {personality['id']}")
                return str(personality["id"])
        
        # Create new personality
        personality_id = uuid.uuid4()
        
        # Insert into database
        query = """
        INSERT INTO ai_personalities (id, name, description, detailed_analysis_prompt, suggestions_prompt, 
                                    is_active, is_default, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        params = [
            personality_id,
            SURVEY_BUILDER_PERSONALITY["name"],
            SURVEY_BUILDER_PERSONALITY["description"],
            SURVEY_BUILDER_PERSONALITY["detailed_analysis_prompt"],
            SURVEY_BUILDER_PERSONALITY["suggestions_prompt"],
            SURVEY_BUILDER_PERSONALITY["is_active"],
            SURVEY_BUILDER_PERSONALITY["is_default"],
            datetime.now(timezone.utc),
            datetime.now(timezone.utc)
        ]
        
        await db.execute_command(query, params)
        
        logger.info(f"Created survey builder AI personality: {personality_id}")
        return str(personality_id)
        
    except Exception as e:
        logger.error(f"Error creating survey builder personality: {str(e)}")
        raise


async def create_survey_builder_module_config(
    db: LightweightDBService, 
    personality_id: str, 
    llm_setting_id: str
) -> str:
    """Create the survey builder module configuration"""
    try:
        # Check if configuration already exists
        configurations = await db.get_all_module_configurations()
        for config in configurations:
            if config["module_name"] == "survey_builder":
                logger.info(f"Survey builder module config already exists: {config['id']}")
                return str(config["id"])
        
        # Create new configuration
        config_id = uuid.uuid4()
        
        # Insert into database
        query = """
        INSERT INTO module_configurations (id, module_name, llm_setting_id, model, temperature, 
                                         max_tokens, max_completion_tokens, active, ai_personality_id, 
                                         created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """
        
        params = [
            config_id,
            SURVEY_BUILDER_MODULE_CONFIG["module_name"],
            uuid.UUID(llm_setting_id),
            SURVEY_BUILDER_MODULE_CONFIG["model"],
            SURVEY_BUILDER_MODULE_CONFIG["temperature"],
            SURVEY_BUILDER_MODULE_CONFIG["max_tokens"],
            SURVEY_BUILDER_MODULE_CONFIG["max_completion_tokens"],
            SURVEY_BUILDER_MODULE_CONFIG["active"],
            uuid.UUID(personality_id),
            datetime.now(timezone.utc),
            datetime.now(timezone.utc)
        ]
        
        await db.execute_command(query, params)
        
        logger.info(f"Created survey builder module configuration: {config_id}")
        return str(config_id)
        
    except Exception as e:
        logger.error(f"Error creating survey builder module config: {str(e)}")
        raise


async def setup_survey_builder_module():
    """Main setup function for survey builder module"""
    try:
        # Initialize database service
        db = LightweightDBService()
        
        # Get available LLM settings
        llm_settings = await db.get_all_llm_settings()
        if not llm_settings:
            raise Exception("No LLM settings found. Please configure an LLM provider first.")
        
        # Use the first active LLM setting
        active_llm = None
        for llm in llm_settings:
            if llm.get("active", False):
                active_llm = llm
                break
        
        if not active_llm:
            # If no active LLM, use the first one
            active_llm = llm_settings[0]
            logger.warning(f"No active LLM found, using: {active_llm['provider']}")
        
        llm_setting_id = str(active_llm["id"])
        
        # Create AI personality
        personality_id = await create_survey_builder_personality(db)
        
        # Create module configuration
        config_id = await create_survey_builder_module_config(
            db, personality_id, llm_setting_id
        )
        
        logger.info("Survey builder module setup completed successfully!")
        return {
            "personality_id": personality_id,
            "config_id": config_id,
            "llm_setting_id": llm_setting_id
        }
        
    except Exception as e:
        logger.error(f"Error setting up survey builder module: {str(e)}")
        raise


if __name__ == "__main__":
    """Run the setup script"""
    asyncio.run(setup_survey_builder_module())