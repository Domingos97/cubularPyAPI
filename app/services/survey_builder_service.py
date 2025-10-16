from typing import Optional, Dict, Any
import uuid
from app.services.lightweight_db_service import LightweightDBService
from app.models.schemas import ModuleConfigurationResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SurveyBuilderService:
    """Service for managing survey builder module configuration and AI interactions"""
    
    MODULE_NAME = "survey_builder"
    DEFAULT_MODEL = "gpt-4"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000
    DEFAULT_MAX_COMPLETION_TOKENS = 1500
    
    # Define the conversation flow steps
    CONVERSATION_STEPS = [
        {
            "step": "topic",
            "question": "What is the main topic or purpose of your survey? Please describe what you want to study or measure.",
            "required": True,
            "examples": ["customer satisfaction", "product feedback", "market research", "employee engagement"]
        },
        {
            "step": "target_audience", 
            "question": "Who is your target audience? Please describe the people you want to survey.",
            "required": True,
            "examples": ["customers aged 25-45", "employees", "students", "general public"]
        },
        {
            "step": "location",
            "question": "What is the geographic location or region for your survey?",
            "required": False,
            "examples": ["Portugal", "Europe", "Worldwide", "North America"]
        },
        {
            "step": "objectives",
            "question": "What specific insights or outcomes do you hope to achieve with this survey?",
            "required": True,
            "examples": ["improve customer service", "understand buying behavior", "measure satisfaction"]
        },
        {
            "step": "sample_size",
            "question": "Approximately how many participants do you expect to survey?",
            "required": False,
            "examples": ["100-500", "1000+", "50-100", "not sure"]
        },
        {
            "step": "demographics",
            "question": "What demographic information do you need to collect? (age, gender, income, education, etc.)",
            "required": False,
            "examples": ["age and gender", "education and income", "basic demographics", "none needed"]
        },
        {
            "step": "completion",
            "question": "Perfect! I have all the information needed. Would you like me to generate your survey now?",
            "required": True,
            "examples": ["yes", "generate survey", "create it", "proceed"]
        }
    ]
    
    def __init__(self):
        logger.info("SurveyBuilderService initialized")
    
    async def get_survey_builder_config(
        self, 
        db: LightweightDBService
    ) -> Optional[ModuleConfigurationResponse]:
        """
        Get the survey builder module configuration with AI personality and LLM settings
        
        Returns:
            ModuleConfigurationResponse or None if not configured
        """
        try:
            # Get all module configurations and filter for survey_builder
            configurations = await db.get_all_module_configurations()
            
            survey_builder_config = None
            for config in configurations:
                if config.get("module_name") == self.MODULE_NAME:
                    survey_builder_config = config
                    break
            
            if not survey_builder_config:
                logger.warning("Survey builder module configuration not found")
                return None
            
            # Convert to response format
            config_response = ModuleConfigurationResponse(
                id=str(survey_builder_config["id"]),
                module_name=survey_builder_config["module_name"],
                llm_setting_id=str(survey_builder_config["llm_setting_id"]) if survey_builder_config.get("llm_setting_id") else None,
                temperature=float(survey_builder_config["temperature"]) if survey_builder_config.get("temperature") else self.DEFAULT_TEMPERATURE,
                max_tokens=survey_builder_config.get("max_tokens", self.DEFAULT_MAX_TOKENS),
                max_completion_tokens=survey_builder_config.get("max_completion_tokens", self.DEFAULT_MAX_COMPLETION_TOKENS),
                active=survey_builder_config.get("active", True),
                created_at=survey_builder_config["created_at"].isoformat() if survey_builder_config.get("created_at") else "",
                updated_at=survey_builder_config["updated_at"].isoformat() if survey_builder_config.get("updated_at") else "",
                created_by=str(survey_builder_config["created_by"]) if survey_builder_config.get("created_by") else None,
                ai_personality_id=str(survey_builder_config["ai_personality_id"]) if survey_builder_config.get("ai_personality_id") else None,
                model=survey_builder_config.get("model", self.DEFAULT_MODEL)
            )
            
            logger.info(f"Retrieved survey builder configuration: {config_response.id}")
            return config_response
            
        except Exception as e:
            logger.error(f"Error retrieving survey builder configuration: {str(e)}")
            raise
    
    async def get_survey_builder_ai_personality(
        self, 
        db: LightweightDBService
    ) -> Optional[Dict[str, Any]]:
        """
        Get the AI personality associated with the survey builder module
        
        Returns:
            Dictionary with AI personality data or None if not found
        """
        try:
            config = await self.get_survey_builder_config(db)
            
            if not config or not config.ai_personality_id:
                logger.warning("No AI personality configured for survey builder")
                return None
            
            # Get the AI personality details
            personality_id = uuid.UUID(config.ai_personality_id)
            personalities = await db.get_all_ai_personalities()
            
            for personality in personalities:
                if personality["id"] == personality_id:
                    logger.info(f"Retrieved survey builder AI personality: {personality['name']}")
                    return personality
            
            logger.warning(f"AI personality {personality_id} not found")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving survey builder AI personality: {str(e)}")
            raise
    
    def analyze_conversation_state(self, conversation: list) -> Dict[str, Any]:
        """
        Analyze the conversation to determine current state and next step
        
        Args:
            conversation: List of chat messages
            
        Returns:
            Dictionary with conversation state and next action
        """
        try:
            # Extract collected information
            collected_info = self._extract_collected_information(conversation)
            
            # Determine current step
            current_step_index = self._determine_current_step(collected_info)
            
            # Check if conversation is complete
            is_complete = current_step_index >= len(self.CONVERSATION_STEPS)
            
            # Get next question if not complete
            next_question = None
            next_step = None
            if not is_complete:
                next_step = self.CONVERSATION_STEPS[current_step_index]
                next_question = next_step["question"]
            
            return {
                "is_complete": is_complete,
                "current_step_index": current_step_index,
                "next_step": next_step,
                "next_question": next_question,
                "collected_info": collected_info,
                "completion_percentage": min(100, (current_step_index / len(self.CONVERSATION_STEPS)) * 100)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation state: {str(e)}")
            raise

    def _extract_collected_information(self, conversation: list) -> Dict[str, Any]:
        """Extract and organize information collected from the conversation"""
        collected = {}
        
        # Look for user responses after specific questions
        for i, message in enumerate(conversation):
            if message.get("sender") == "assistant":
                content = message.get("content", "").lower()
                
                # Look for the next user message as a response
                if i + 1 < len(conversation) and conversation[i + 1].get("sender") == "user":
                    user_response = conversation[i + 1].get("content", "").strip()
                    
                    # Match question patterns to steps
                    if "topic" in content or "purpose" in content or "what you want to study" in content:
                        collected["topic"] = user_response
                    elif "target audience" in content or "who" in content and "survey" in content:
                        collected["target_audience"] = user_response
                    elif "location" in content or "geographic" in content or "region" in content:
                        collected["location"] = user_response
                    elif "objectives" in content or "insights" in content or "outcomes" in content:
                        collected["objectives"] = user_response
                    elif "sample size" in content or "participants" in content or "how many" in content:
                        collected["sample_size"] = user_response
                    elif "demographic" in content or "age" in content or "gender" in content:
                        collected["demographics"] = user_response
                    elif "generate" in content and "survey" in content:
                        collected["completion"] = user_response
        
        # Also extract from initial user message if it contains survey description
        first_user_message = None
        for message in conversation:
            if message.get("sender") == "user":
                first_user_message = message.get("content", "").strip()
                break
        
        if first_user_message and not collected.get("topic"):
            # If first message contains survey info, use it as topic
            if any(keyword in first_user_message.lower() for keyword in ["survey", "study", "research", "measure", "feedback"]):
                collected["topic"] = first_user_message
        
        return collected

    def _determine_current_step(self, collected_info: Dict[str, Any]) -> int:
        """Determine which step we're currently on based on collected information"""
        step_index = 0
        
        for step in self.CONVERSATION_STEPS:
            step_name = step["step"]
            
            # Check if this step has been completed
            if step_name in collected_info and collected_info[step_name]:
                step_index += 1
            else:
                # If required step is missing, we're at this step
                if step["required"]:
                    break
                # If optional step is missing, skip it for now
                else:
                    step_index += 1
        
        return step_index

    def validate_survey_completion_intent(self, conversation: list) -> bool:
        """
        Check if the guided conversation is complete and ready for survey generation
        
        Args:
            conversation: List of chat messages
            
        Returns:
            Boolean indicating if ready to generate survey
        """
        try:
            # First check for explicit completion signals from user
            completion_signals = [
                "just proceed", "proceed", "let's create", "let's generate", 
                "generate survey", "create survey", "that's all", "i'm ready",
                "ready to generate", "wrap the conversation", "let wrap",
                "that covers everything", "go ahead", "create it now",
                "finish", "complete", "done", "generate it"
            ]
            
            user_messages = [msg.get("content", "").lower() for msg in conversation if msg.get("sender") == "user"]
            last_user_messages = user_messages[-3:]  # Check last 3 user messages for completion intent
            
            # Check if user has given completion signals
            for message in last_user_messages:
                if any(signal in message for signal in completion_signals):
                    logger.info(f"Found completion signal in user message: '{message}'")
                    return True
            
            # Check using the regular state analysis
            state = self.analyze_conversation_state(conversation)
            
            # More flexible completion criteria:
            # 1. If we have at least a topic and the user has expressed completion intent
            # 2. If conversation has enough substance (topic + some interaction)
            collected_info = state["collected_info"]
            has_topic = bool(collected_info.get("topic"))
            
            if has_topic and len(conversation) >= 4:  # At least 2 exchanges (4 messages)
                # Check if the conversation shows clear intent to proceed
                conversation_text = " ".join(user_messages).lower()
                if any(proceed_word in conversation_text for proceed_word in ["proceed", "correlations", "just need", "create"]):
                    logger.info("Found sufficient information and proceed intent - marking as complete")
                    return True
            
            # Fallback to the original strict validation
            return state["is_complete"]
            
        except Exception as e:
            logger.error(f"Error validating completion: {str(e)}")
            # Fallback: if there's any substantial conversation with a clear topic, allow generation
            try:
                user_messages = [msg.get("content", "").lower() for msg in conversation if msg.get("sender") == "user"]
                if len(user_messages) >= 2:
                    first_message = user_messages[0]
                    # If first message mentions survey/study and user has responded, that's enough
                    if any(word in first_message for word in ["survey", "study", "research", "measure", "correlations"]):
                        return True
                return len(conversation) >= 6
            except:
                return len(conversation) >= 6

    def get_next_question(self, conversation: list) -> Dict[str, Any]:
        """
        Get the next question to ask the user based on conversation state
        
        Args:
            conversation: List of chat messages
            
        Returns:
            Dictionary with next question and guidance
        """
        try:
            state = self.analyze_conversation_state(conversation)
            
            if state["is_complete"]:
                return {
                    "question": "Perfect! I have all the information needed to create your survey. Would you like me to generate it now?",
                    "is_complete": True,
                    "completion_percentage": 100,
                    "collected_info": state["collected_info"]
                }
            
            next_step = state["next_step"]
            return {
                "question": next_step["question"],
                "examples": next_step.get("examples", []),
                "step": next_step["step"],
                "is_complete": False,
                "completion_percentage": state["completion_percentage"],
                "collected_info": state["collected_info"]
            }
            
        except Exception as e:
            logger.error(f"Error getting next question: {str(e)}")
            return {
                "question": "What would you like to create a survey about?",
                "is_complete": False,
                "completion_percentage": 0,
                "collected_info": {}
            }
    
    def extract_survey_requirements(self, conversation: list) -> Dict[str, Any]:
        """
        Extract comprehensive survey requirements from the guided conversation
        
        Args:
            conversation: List of chat messages
            
        Returns:
            Dictionary with extracted survey information
        """
        try:
            # Get collected information from conversation analysis
            state = self.analyze_conversation_state(conversation)
            collected_info = state["collected_info"]
            
            # Build comprehensive requirements
            requirements = {
                "target_audience": collected_info.get("target_audience", ""),
                "study_objectives": collected_info.get("objectives", ""),
                "topic": collected_info.get("topic", ""),
                "location": collected_info.get("location", ""),
                "sample_size": collected_info.get("sample_size", ""),
                "demographics_requested": collected_info.get("demographics", ""),
                "completion_percentage": state["completion_percentage"],
                "is_complete": state["is_complete"]
            }
            
            # Parse demographics into list
            demographics_text = requirements["demographics_requested"].lower()
            demographics_list = []
            
            demographic_keywords = {
                "age": ["age"],
                "gender": ["gender", "sex"],
                "education": ["education", "degree", "school"],
                "income": ["income", "salary", "earnings"],
                "occupation": ["job", "occupation", "profession", "work"],
                "location": ["location", "city", "country", "region"]
            }
            
            for demo, keywords in demographic_keywords.items():
                if any(keyword in demographics_text for keyword in keywords):
                    demographics_list.append(demo)
            
            # If no specific demographics mentioned but demographics requested
            if not demographics_list and demographics_text and "none" not in demographics_text:
                demographics_list = ["age", "gender"]  # Default basic demographics
            
            requirements["demographics"] = demographics_list
            
            # Infer psychographics based on topic
            topic_lower = requirements["topic"].lower()
            psychographics = []
            
            if any(word in topic_lower for word in ["satisfaction", "experience", "feedback"]):
                psychographics.extend(["satisfaction_level", "likelihood_to_recommend", "overall_experience"])
            
            if any(word in topic_lower for word in ["product", "service", "quality"]):
                psychographics.extend(["quality_perception", "value_for_money", "brand_loyalty"])
            
            if any(word in topic_lower for word in ["customer", "client", "user"]):
                psychographics.extend(["usage_frequency", "purchase_behavior", "preferences"])
            
            requirements["psychographics"] = psychographics
            
            return requirements
            
        except Exception as e:
            logger.error(f"Error extracting survey requirements: {str(e)}")
            # Fallback to basic extraction
            return {
                "target_audience": "",
                "study_objectives": "",
                "topic": "",
                "demographics": ["age", "gender"],
                "psychographics": [],
                "location": "",
                "sample_size": ""
            }


# Global instance
survey_builder_service = SurveyBuilderService()