"""
Survey Generation Service for converting chat conversations into structured survey formats
"""

import pandas as pd
import uuid
from typing import List, Dict, Any, Tuple
from datetime import datetime
import io
import os
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SurveyGenerationService:
    """Service for generating CSV surveys from chat conversations"""
    
    def __init__(self):
        self.output_directory = "survey_generated"
        os.makedirs(self.output_directory, exist_ok=True)
        logger.info("SurveyGenerationService initialized")
    
    def analyze_conversation_for_survey(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the conversation to extract survey requirements and generate questions
        
        Args:
            conversation: List of chat messages
            
        Returns:
            Dictionary with survey structure and questions
        """
        try:
            # Extract survey requirements
            requirements = self._extract_requirements(conversation)
            
            # Generate questions based on requirements
            questions = self._generate_questions(requirements)
            
            # Categorize questions
            categorized_questions = self._categorize_questions(questions, requirements)
            
            return {
                "requirements": requirements,
                "questions": categorized_questions,
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "conversation_length": len(conversation),
                    "user_messages": len([msg for msg in conversation if msg.get("sender") == "user"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}")
            raise
    
    def generate_survey_files(
        self,
        survey_data: Dict[str, Any],
        survey_title: str = None
    ) -> str:
        """
        Generate a CSV file from survey data

        Args:
            survey_data: Survey structure from analyze_conversation_for_survey
            survey_title: Optional title for the survey file

        Returns:
            csv_file_path (str)
        """
        try:
            # Create survey DataFrame with the required structure
            df = self._create_survey_dataframe(survey_data)

            # Generate file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = survey_title or "generated_survey"
            safe_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_')

            csv_filename = f"{safe_name}_{timestamp}.csv"
            csv_path = os.path.join(self.output_directory, csv_filename)

            # Save CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')

            logger.info(f"Generated survey file: {csv_filename}")
            return csv_path

        except Exception as e:
            logger.error(f"Error generating survey file: {str(e)}")
            raise
    
    def _extract_requirements(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract survey requirements from conversation using guided flow data"""
        
        # First try to get requirements from survey builder service (guided flow)
        try:
            from app.services.survey_builder_service import survey_builder_service
            guided_requirements = survey_builder_service.extract_survey_requirements(conversation)
            
            # If we have guided requirements, use them as base
            if guided_requirements.get("topic"):
                requirements = {
                    "target_audience": guided_requirements.get("target_audience", ""),
                    "study_objectives": guided_requirements.get("study_objectives", ""),
                    "topic": guided_requirements.get("topic", ""),
                    "location": guided_requirements.get("location", ""),
                    "demographics_needed": guided_requirements.get("demographics", []),
                    "psychographics_needed": guided_requirements.get("psychographics", []),
                    "specific_areas": [],
                    "survey_goals": []
                }
                
                # Enhance with topic-specific areas and goals
                topic_lower = requirements["topic"].lower()
                
                # Determine specific areas based on topic
                if any(word in topic_lower for word in ["customer", "satisfaction", "service", "experience"]):
                    requirements["specific_areas"] = [
                        "customer_satisfaction", "service_quality", "overall_experience", 
                        "likelihood_to_recommend", "future_usage_intent"
                    ]
                    requirements["survey_goals"] = [
                        "measure_satisfaction_levels", "identify_improvement_areas", 
                        "track_customer_loyalty", "benchmark_service_quality"
                    ]
                
                elif any(word in topic_lower for word in ["product", "feedback", "review"]):
                    requirements["specific_areas"] = [
                        "product_quality", "feature_satisfaction", "usability", 
                        "value_for_money", "purchase_experience"
                    ]
                    requirements["survey_goals"] = [
                        "evaluate_product_performance", "gather_improvement_feedback",
                        "understand_user_needs", "guide_product_development"
                    ]
                
                elif any(word in topic_lower for word in ["market", "research", "consumer"]):
                    requirements["specific_areas"] = [
                        "market_preferences", "buying_behavior", "brand_awareness",
                        "price_sensitivity", "feature_importance"
                    ]
                    requirements["survey_goals"] = [
                        "understand_market_trends", "identify_opportunities",
                        "analyze_competitive_position", "inform_strategy"
                    ]
                
                elif any(word in topic_lower for word in ["employee", "workplace", "job"]):
                    requirements["specific_areas"] = [
                        "job_satisfaction", "work_environment", "management_effectiveness",
                        "career_development", "work_life_balance"
                    ]
                    requirements["survey_goals"] = [
                        "measure_employee_engagement", "identify_workplace_issues",
                        "improve_retention", "enhance_productivity"
                    ]
                
                # Default areas if none match
                if not requirements["specific_areas"]:
                    requirements["specific_areas"] = [
                        "general_satisfaction", "usage_frequency", "preferences",
                        "overall_experience", "recommendations"
                    ]
                    requirements["survey_goals"] = [
                        "gather_feedback", "understand_preferences", 
                        "measure_satisfaction", "identify_trends"
                    ]
                
                return requirements
                
        except Exception as e:
            logger.warning(f"Could not get guided requirements, falling back to keyword extraction: {str(e)}")
        
        # Fallback to original keyword-based extraction
        requirements = {
            "target_audience": "",
            "study_objectives": "",
            "topic": "",
            "location": "",
            "demographics_needed": [],
            "psychographics_needed": [],
            "specific_areas": [],
            "survey_goals": []
        }
        
        # Combine all user messages
        user_content = " ".join([
            msg.get("content", "") 
            for msg in conversation 
            if msg.get("sender") == "user"
        ])
        
        user_content_lower = user_content.lower()
        
        # Extract location
        location_keywords = {
            "portugal": "Portugal",
            "spain": "Spain", 
            "brazil": "Brazil",
            "usa": "USA",
            "uk": "UK",
            "germany": "Germany",
            "france": "France"
        }
        
        for keyword, location in location_keywords.items():
            if keyword in user_content_lower:
                requirements["location"] = location
                break
        
        # Extract topic with more sophisticated detection
        topic_keywords = {
            "Gaming": ["video game", "gaming", "game", "gamer", "esports", "console", "pc gaming", "mobile gaming"],
            "Food & Dining": ["food", "restaurant", "dining", "cuisine", "cooking", "meal", "eating", "chef"],
            "Health & Wellness": ["health", "fitness", "exercise", "wellness", "nutrition", "medical", "diet"],
            "Beauty & Cosmetics": ["makeup", "cosmetics", "beauty", "skincare", "fashion", "style"],
            "Technology": ["tech", "software", "app", "digital", "internet", "computer", "smartphone"],
            "Travel": ["travel", "vacation", "tourism", "trip", "destination", "hotel"],
            "Education": ["education", "learning", "school", "university", "student", "teaching"],
            "Shopping": ["shopping", "retail", "purchase", "buying", "consumer", "ecommerce"],
            "Entertainment": ["movie", "music", "tv", "entertainment", "streaming", "media"],
            "Finance": ["money", "finance", "banking", "investment", "budget", "salary"]
        }
        
        # Find the best matching topic
        best_topic = ""
        max_matches = 0
        
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in user_content_lower)
            if matches > max_matches:
                max_matches = matches
                best_topic = topic
        
        requirements["topic"] = best_topic
        
        # Set specific areas based on detected topic
        topic_areas = {
            "Gaming": ["gaming_habits", "game_preferences", "gaming_frequency", "platform_usage", "gaming_spending"],
            "Food & Dining": ["dining_preferences", "food_habits", "restaurant_choices", "cuisine_preferences"],
            "Health & Wellness": ["exercise_habits", "health_concerns", "wellness_practices", "nutrition_habits"],
            "Beauty & Cosmetics": ["beauty_routine", "product_preferences", "brand_loyalty", "spending_habits"],
            "Technology": ["tech_usage", "device_preferences", "software_habits", "digital_behavior"],
            "Travel": ["travel_frequency", "destination_preferences", "travel_budget", "accommodation_choices"],
            "Education": ["learning_preferences", "educational_background", "skill_development", "study_habits"],
            "Shopping": ["shopping_behavior", "brand_preferences", "purchase_decisions", "spending_patterns"],
            "Entertainment": ["content_preferences", "viewing_habits", "platform_usage", "entertainment_spending"],
            "Finance": ["financial_habits", "investment_preferences", "budgeting_behavior", "financial_goals"]
        }
        
        requirements["specific_areas"] = topic_areas.get(best_topic, [])
        
        # Extract demographics typically needed
        requirements["demographics_needed"] = ["age", "gender", "location", "education", "occupation", "income"]
        
        # Extract psychographics based on topic
        psychographic_areas = {
            "Gaming": ["gaming_motivation", "competitive_behavior", "social_gaming_preferences", "technology_adoption"],
            "Food & Dining": ["taste_preferences", "dining_motivation", "health_consciousness", "culinary_adventurousness"],
            "Health & Wellness": ["health_motivation", "lifestyle_preferences", "wellness_priorities", "exercise_attitudes"],
            "Beauty & Cosmetics": ["beauty_values", "self_image", "style_preferences", "brand_loyalty"],
            "Technology": ["innovation_adoption", "privacy_concerns", "digital_habits", "tech_attitudes"],
            "Travel": ["travel_motivation", "adventure_seeking", "cultural_interests", "comfort_preferences"],
            "Education": ["learning_motivation", "knowledge_interests", "educational_values", "skill_priorities"],
            "Shopping": ["purchase_motivation", "brand_loyalty", "quality_vs_price", "shopping_enjoyment"],
            "Entertainment": ["content_preferences", "entertainment_values", "social_viewing", "genre_preferences"],
            "Finance": ["risk_tolerance", "financial_values", "spending_attitudes", "investment_motivation"]
        }
        
        requirements["psychographics_needed"] = psychographic_areas.get(best_topic, [])
        
        return requirements
    
    def _generate_questions(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive survey questions based on requirements"""
        questions = []
        
        # Core Demographics (8 questions)
        questions.extend([
            {
                "identifier": "age",
                "text": "What is your age?", 
                "type": "number", 
                "category": "demo"
            },
            {
                "identifier": "gender",
                "text": "What is your gender?", 
                "type": "multiple_choice", 
                "category": "demo", 
                "options": ["Male", "Female", "Non-binary", "Prefer not to say"]
            },
            {
                "identifier": "education",
                "text": "What is your highest level of education?", 
                "type": "multiple_choice", 
                "category": "demo",
                "options": ["High school", "Bachelor's degree", "Master's degree", "PhD", "Other"]
            },
            {
                "identifier": "occupation",
                "text": "What is your current occupation?", 
                "type": "text", 
                "category": "demo"
            },
            {
                "identifier": "income",
                "text": "What is your monthly income (after taxes)?",
                "type": "multiple_choice",
                "category": "demo",
                "options": ["Under €1,000", "€1,000-€2,000", "€2,000-€3,000", "€3,000-€5,000", "Over €5,000"]
            },
            {
                "identifier": "household_size",
                "text": "How many people live in your household?",
                "type": "multiple_choice",
                "category": "demo",
                "options": ["1", "2", "3", "4", "5+"]
            },
            {
                "identifier": "relationship_status",
                "text": "What is your current relationship status?",
                "type": "multiple_choice",
                "category": "demo",
                "options": ["Single", "In a relationship", "Married", "Divorced", "Widowed", "Prefer not to say"]
            },
            {
                "identifier": "employment_status",
                "text": "What is your current employment status?",
                "type": "multiple_choice",
                "category": "demo",
                "options": ["Full-time employed", "Part-time employed", "Self-employed", "Student", "Retired", "Unemployed", "Other"]
            }
        ])
        
        # Location-specific questions (1 question)
        if requirements.get("location"):
            questions.append({
                "identifier": "location_residence",
                "text": f"Are you currently residing in {requirements['location']}?",
                "type": "yes_no", 
                "category": "demo"
            })
        
        # General Psychographic Questions (6 questions)
        questions.extend([
            {
                "identifier": "lifestyle_values",
                "text": "Which values are most important to you?",
                "type": "multiple_select",
                "category": "psych",
                "options": ["Quality", "Value for money", "Sustainability", "Innovation", "Tradition", "Social responsibility"]
            },
            {
                "identifier": "decision_style",
                "text": "How do you typically make purchasing decisions?",
                "type": "multiple_choice",
                "category": "psych",
                "options": ["Research thoroughly", "Ask friends/family", "Read reviews", "Trust my instincts", "Go with familiar brands"]
            },
            {
                "identifier": "social_influence",
                "text": "How much do social media influencers affect your choices?",
                "type": "rating",
                "category": "psych",
                "scale": "1-5 (No influence to Strong influence)"
            },
            {
                "identifier": "risk_tolerance",
                "text": "How would you describe your approach to trying new things?",
                "type": "multiple_choice",
                "category": "psych",
                "options": ["Very adventurous", "Somewhat adventurous", "Cautious", "Very cautious", "Prefer familiar options"]
            },
            {
                "identifier": "brand_loyalty",
                "text": "How important is brand reputation in your decision making?",
                "type": "rating",
                "category": "psych",
                "scale": "1-5 (Not important to Very important)"
            },
            {
                "identifier": "environmental_consciousness",
                "text": "How important are environmental considerations in your choices?",
                "type": "rating",
                "category": "psych",
                "scale": "1-5 (Not important to Very important)"
            }
        ])
        
        # Topic-specific questions (15+ questions per topic)
        if requirements.get("topic") == "Gaming":
            questions.extend([
                {
                    "identifier": "gaming_frequency",
                    "text": "How often do you play video games?", 
                    "type": "multiple_choice", 
                    "category": "psych",
                    "options": ["Daily", "Several times a week", "Weekly", "Monthly", "Rarely", "Never"]
                },
                {
                    "identifier": "gaming_hours_per_week",
                    "text": "How many hours per week do you spend gaming?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Less than 5", "5-10", "11-20", "21-35", "36-50", "More than 50"]
                },
                {
                    "identifier": "gaming_platforms",
                    "text": "What gaming platforms do you primarily use?", 
                    "type": "multiple_select", 
                    "category": "text",
                    "options": ["PC", "PlayStation", "Xbox", "Nintendo Switch", "Mobile", "VR", "Handheld consoles"]
                },
                {
                    "identifier": "gaming_motivation",
                    "text": "What motivates you to play video games?", 
                    "type": "multiple_select", 
                    "category": "psych",
                    "options": ["Entertainment", "Competition", "Social interaction", "Stress relief", "Achievement", "Escapism", "Learning", "Creativity"]
                },
                {
                    "identifier": "gaming_spending",
                    "text": "How much do you typically spend on video games per month?", 
                    "type": "multiple_choice", 
                    "category": "demo",
                    "options": ["€0", "€1-25", "€26-50", "€51-100", "€100+"]
                },
                {
                    "identifier": "gaming_preference",
                    "text": "Do you prefer single-player or multiplayer games?", 
                    "type": "multiple_choice", 
                    "category": "psych",
                    "options": ["Single-player", "Multiplayer", "Both equally", "No preference"]
                },
                {
                    "identifier": "gaming_genres",
                    "text": "What genres of games do you enjoy most?", 
                    "type": "multiple_select", 
                    "category": "text",
                    "options": ["Action", "RPG", "Strategy", "Sports", "Simulation", "Puzzle", "Horror", "Adventure", "Racing", "Fighting"]
                },
                {
                    "identifier": "gaming_social_importance",
                    "text": "How important is the social aspect of gaming to you?", 
                    "type": "rating", 
                    "category": "psych",
                    "scale": "1-5 (Not important to Very important)"
                },
                {
                    "identifier": "gaming_news_following",
                    "text": "Do you follow gaming news and trends?", 
                    "type": "yes_no", 
                    "category": "psych"
                },
                {
                    "identifier": "gaming_purchase_influence",
                    "text": "What influences your decision to purchase a new game?", 
                    "type": "multiple_select", 
                    "category": "psych",
                    "options": ["Reviews", "Friends' recommendations", "Trailers", "Price", "Brand/Developer", "Genre", "Streamer recommendations"]
                },
                {
                    "identifier": "gaming_device_preference",
                    "text": "What type of gaming device do you prefer for different scenarios?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["High-end PC for everything", "Console for comfort", "Mobile for convenience", "Mix depending on game", "Portable devices"]
                },
                {
                    "identifier": "gaming_competitive_level",
                    "text": "How competitive are you when gaming?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Casual to Highly competitive)"
                },
                {
                    "identifier": "gaming_time_of_day",
                    "text": "When do you prefer to game?",
                    "type": "multiple_select",
                    "category": "psych",
                    "options": ["Early morning", "Afternoon", "Evening", "Late night", "Weekends only", "Any time available"]
                },
                {
                    "identifier": "gaming_technology_adoption",
                    "text": "How do you approach new gaming technology?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Early adopter", "Wait for reviews", "Wait for price drops", "Only when necessary", "Never upgrade"]
                },
                {
                    "identifier": "gaming_content_creation",
                    "text": "Do you create gaming-related content?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Streaming", "YouTube videos", "Social media posts", "Reviews", "Guides", "None"]
                },
                {
                    "identifier": "gaming_subscription_services",
                    "text": "Which gaming subscription services do you use?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Xbox Game Pass", "PlayStation Plus", "Nintendo Switch Online", "EA Play", "Ubisoft+", "None"]
                },
                {
                    "identifier": "gaming_graphics_importance",
                    "text": "How important are high-quality graphics in games to you?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Not important to Very important)"
                },
                {
                    "identifier": "gaming_story_importance",
                    "text": "How important is a compelling story in games?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Not important to Very important)"
                }
            ])
        elif requirements.get("topic") == "Food & Dining":
            questions.extend([
                {
                    "identifier": "dining_frequency",
                    "text": "How often do you dine out per week?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Never", "1-2 times", "3-4 times", "5-6 times", "Daily"]
                },
                {
                    "identifier": "cuisine_preferences",
                    "text": "What types of cuisine do you prefer?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Italian", "Asian", "Mediterranean", "Mexican", "American", "Indian", "French", "Spanish", "Middle Eastern", "Other"]
                },
                {
                    "identifier": "dining_budget",
                    "text": "What is your typical budget for dining out per meal?",
                    "type": "multiple_choice",
                    "category": "demo",
                    "options": ["Under €15", "€15-€30", "€30-€50", "€50-€100", "Over €100"]
                },
                {
                    "identifier": "cooking_frequency",
                    "text": "How often do you cook at home?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Daily", "5-6 times per week", "3-4 times per week", "1-2 times per week", "Rarely", "Never"]
                },
                {
                    "identifier": "food_delivery_usage",
                    "text": "How often do you order food delivery?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Daily", "Several times a week", "Weekly", "Monthly", "Rarely", "Never"]
                },
                {
                    "identifier": "dietary_restrictions",
                    "text": "Do you follow any specific dietary restrictions?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Vegetarian", "Vegan", "Gluten-free", "Keto", "Paleo", "Halal", "Kosher", "Low-carb", "None"]
                },
                {
                    "identifier": "food_shopping_frequency",
                    "text": "How often do you shop for groceries?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Daily", "2-3 times per week", "Weekly", "Bi-weekly", "Monthly"]
                },
                {
                    "identifier": "health_consciousness",
                    "text": "How important is eating healthy to you?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Not important to Very important)"
                },
                {
                    "identifier": "organic_preference",
                    "text": "How often do you choose organic food products?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Always", "Often", "Sometimes", "Rarely", "Never"]
                },
                {
                    "identifier": "local_food_preference",
                    "text": "How important is buying local/seasonal food to you?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Not important to Very important)"
                },
                {
                    "identifier": "food_adventure_level",
                    "text": "How adventurous are you with trying new foods?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Very adventurous", "Somewhat adventurous", "Moderate", "Cautious", "Stick to familiar foods"]
                },
                {
                    "identifier": "meal_planning_habits",
                    "text": "Do you plan your meals in advance?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Always plan weekly", "Plan a few days ahead", "Plan day by day", "Rarely plan", "Never plan"]
                },
                {
                    "identifier": "food_waste_concern",
                    "text": "How concerned are you about food waste?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Not concerned to Very concerned)"
                },
                {
                    "identifier": "restaurant_selection_factors",
                    "text": "What factors most influence your restaurant choice?",
                    "type": "multiple_select",
                    "category": "psych",
                    "options": ["Price", "Food quality", "Location", "Reviews", "Atmosphere", "Service", "Healthy options", "Portion size"]
                },
                {
                    "identifier": "cooking_skill_level",
                    "text": "How would you rate your cooking skills?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Expert", "Advanced", "Intermediate", "Basic", "Beginner"]
                }
            ])
        elif requirements.get("topic") == "Beauty & Cosmetics":
            questions.extend([
                {
                    "identifier": "makeup_frequency",
                    "text": "How often do you wear makeup?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Daily", "Several times a week", "Weekly", "Occasionally", "Rarely", "Never"]
                },
                {
                    "identifier": "skincare_routine",
                    "text": "How would you describe your skincare routine?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Minimal", "Basic", "Moderate", "Extensive", "Professional"]
                },
                {
                    "identifier": "beauty_spending",
                    "text": "How much do you spend on beauty products per month?",
                    "type": "multiple_choice",
                    "category": "demo",
                    "options": ["Under €25", "€25-€50", "€50-€100", "€100-€200", "Over €200"]
                },
                {
                    "identifier": "beauty_motivation",
                    "text": "What motivates your beauty routine?",
                    "type": "multiple_select",
                    "category": "psych",
                    "options": ["Self-confidence", "Professional appearance", "Self-care", "Creativity", "Social expectations", "Personal enjoyment"]
                },
                {
                    "identifier": "skincare_concerns",
                    "text": "What are your main skincare concerns?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Acne", "Aging", "Dryness", "Oiliness", "Sensitivity", "Dark spots", "Wrinkles", "None"]
                },
                {
                    "identifier": "beauty_product_research",
                    "text": "How do you research beauty products before purchasing?",
                    "type": "multiple_select",
                    "category": "psych",
                    "options": ["Online reviews", "Social media", "Friends/family", "Beauty consultants", "Magazines", "YouTube/TikTok", "Trial/samples"]
                },
                {
                    "identifier": "beauty_brand_loyalty",
                    "text": "How loyal are you to specific beauty brands?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Very loyal", "Somewhat loyal", "Willing to try new brands", "Always switching", "Price-driven only"]
                },
                {
                    "identifier": "natural_vs_synthetic",
                    "text": "Do you prefer natural or synthetic beauty products?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Prefer natural", "Lean toward natural", "No preference", "Lean toward synthetic", "Prefer synthetic"]
                },
                {
                    "identifier": "beauty_trends_following",
                    "text": "How closely do you follow beauty trends?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Very closely", "Somewhat", "Occasionally", "Rarely", "Never"]
                },
                {
                    "identifier": "professional_beauty_services",
                    "text": "How often do you use professional beauty services?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Weekly", "Monthly", "Few times a year", "Annually", "Special occasions only", "Never"]
                },
                {
                    "identifier": "beauty_sustainability_importance",
                    "text": "How important is sustainability in beauty products to you?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Not important to Very important)"
                },
                {
                    "identifier": "beauty_influencer_impact",
                    "text": "How much do beauty influencers impact your purchasing decisions?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (No impact to Strong impact)"
                },
                {
                    "identifier": "makeup_skill_level",
                    "text": "How would you rate your makeup application skills?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Expert", "Advanced", "Intermediate", "Basic", "Beginner", "Don't wear makeup"]
                }
            ])
        elif requirements.get("topic") == "Technology":
            questions.extend([
                {
                    "identifier": "tech_adoption",
                    "text": "How quickly do you adopt new technology?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["I'm always first", "Early adopter", "Wait for reviews", "Follow the crowd", "Very late adopter"]
                },
                {
                    "identifier": "device_usage",
                    "text": "What devices do you use daily?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Smartphone", "Laptop", "Desktop", "Tablet", "Smart watch", "Smart home devices", "Gaming console"]
                },
                {
                    "identifier": "privacy_concern",
                    "text": "How concerned are you about digital privacy?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Not concerned to Very concerned)"
                },
                {
                    "identifier": "screen_time_daily",
                    "text": "How many hours do you spend on digital devices daily?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Less than 2", "2-4", "5-8", "9-12", "More than 12"]
                },
                {
                    "identifier": "tech_spending",
                    "text": "How much do you spend on technology per year?",
                    "type": "multiple_choice",
                    "category": "demo",
                    "options": ["Under €500", "€500-€1,000", "€1,000-€2,000", "€2,000-€5,000", "Over €5,000"]
                },
                {
                    "identifier": "social_media_usage",
                    "text": "Which social media platforms do you use regularly?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Facebook", "Instagram", "Twitter/X", "LinkedIn", "TikTok", "YouTube", "Snapchat", "Discord", "None"]
                },
                {
                    "identifier": "ai_comfort_level",
                    "text": "How comfortable are you with AI-powered services?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Very comfortable", "Somewhat comfortable", "Neutral", "Somewhat uncomfortable", "Very uncomfortable"]
                },
                {
                    "identifier": "cloud_storage_usage",
                    "text": "Do you use cloud storage services?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Google Drive", "iCloud", "Dropbox", "OneDrive", "Other", "None"]
                },
                {
                    "identifier": "tech_news_consumption",
                    "text": "How often do you follow technology news?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Daily", "Weekly", "Monthly", "Occasionally", "Never"]
                },
                {
                    "identifier": "data_sharing_comfort",
                    "text": "How comfortable are you with sharing personal data for better services?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Very uncomfortable to Very comfortable)"
                },
                {
                    "identifier": "tech_support_preference",
                    "text": "How do you prefer to get technical support?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Self-service/online", "Chat support", "Phone support", "In-person", "Ask friends/family"]
                },
                {
                    "identifier": "subscription_services_usage",
                    "text": "Which digital subscription services do you use?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Netflix", "Spotify", "YouTube Premium", "Amazon Prime", "Office 365", "Adobe Creative", "Other", "None"]
                }
            ])
        elif requirements.get("topic") == "Health & Wellness":
            questions.extend([
                {
                    "identifier": "exercise_frequency",
                    "text": "How often do you exercise per week?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Never", "1-2 times", "3-4 times", "5-6 times", "Daily"]
                },
                {
                    "identifier": "wellness_priorities",
                    "text": "What are your top wellness priorities?",
                    "type": "multiple_select",
                    "category": "psych",
                    "options": ["Physical fitness", "Mental health", "Nutrition", "Sleep quality", "Stress management", "Preventive care"]
                },
                {
                    "identifier": "health_spending",
                    "text": "How much do you spend on health and wellness per month?",
                    "type": "multiple_choice",
                    "category": "demo",
                    "options": ["Under €50", "€50-€100", "€100-€200", "€200-€500", "Over €500"]
                },
                {
                    "identifier": "fitness_activities",
                    "text": "What types of physical activities do you engage in?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Running", "Gym workouts", "Yoga", "Swimming", "Cycling", "Team sports", "Walking", "Dancing", "None"]
                },
                {
                    "identifier": "sleep_quality",
                    "text": "How would you rate your sleep quality?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Very poor to Excellent)"
                },
                {
                    "identifier": "stress_level",
                    "text": "How would you rate your current stress level?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Very low to Very high)"
                },
                {
                    "identifier": "nutrition_awareness",
                    "text": "How conscious are you about your nutrition?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Very conscious", "Somewhat conscious", "Moderately conscious", "Slightly conscious", "Not conscious"]
                },
                {
                    "identifier": "wellness_apps_usage",
                    "text": "Do you use health and wellness apps?",
                    "type": "multiple_select",
                    "category": "text",
                    "options": ["Fitness tracking", "Meditation", "Nutrition", "Sleep tracking", "Mental health", "None"]
                },
                {
                    "identifier": "preventive_care_frequency",
                    "text": "How often do you visit healthcare providers for preventive care?",
                    "type": "multiple_choice",
                    "category": "psych",
                    "options": ["Annually", "Every 6 months", "As needed", "Rarely", "Never"]
                },
                {
                    "identifier": "wellness_information_sources",
                    "text": "Where do you get wellness information?",
                    "type": "multiple_select",
                    "category": "psych",
                    "options": ["Healthcare providers", "Online articles", "Social media", "Books", "Friends/family", "Apps", "Podcasts"]
                },
                {
                    "identifier": "mental_health_awareness",
                    "text": "How important is mental health care to you?",
                    "type": "rating",
                    "category": "psych",
                    "scale": "1-5 (Not important to Very important)"
                },
                {
                    "identifier": "wellness_goals",
                    "text": "What are your main wellness goals?",
                    "type": "multiple_select",
                    "category": "psych",
                    "options": ["Weight management", "Increased energy", "Better sleep", "Stress reduction", "Improved fitness", "Disease prevention", "Mental clarity"]
                }
            ])
        
        # Add comprehensive general questions regardless of topic (8 additional questions)
        questions.extend([
            {
                "identifier": "internet_usage_hours",
                "text": "How many hours do you spend online daily?",
                "type": "multiple_choice",
                "category": "psych",
                "options": ["Less than 1", "1-3", "4-6", "7-9", "10+"]
            },
            {
                "identifier": "shopping_preference",
                "text": "Do you prefer online or in-store shopping?",
                "type": "multiple_choice",
                "category": "psych",
                "options": ["Always online", "Mostly online", "Mixed", "Mostly in-store", "Always in-store"]
            },
            {
                "identifier": "price_sensitivity",
                "text": "How important is price when making purchasing decisions?",
                "type": "rating",
                "category": "psych",
                "scale": "1-5 (Not important to Very important)"
            },
            {
                "identifier": "customer_service_importance",
                "text": "How important is good customer service to you?",
                "type": "rating",
                "category": "psych",
                "scale": "1-5 (Not important to Very important)"
            },
            {
                "identifier": "recommendation_likelihood",
                "text": "How likely are you to recommend products/services you like to others?",
                "type": "rating",
                "category": "psych",
                "scale": "1-5 (Very unlikely to Very likely)"
            },
            {
                "identifier": "innovation_interest",
                "text": "How interested are you in innovative products and services?",
                "type": "rating",
                "category": "psych",
                "scale": "1-5 (Not interested to Very interested)"
            },
            {
                "identifier": "convenience_importance",
                "text": "How important is convenience in your daily choices?",
                "type": "rating",
                "category": "psych",
                "scale": "1-5 (Not important to Very important)"
            },
            {
                "identifier": "future_outlook",
                "text": "How optimistic are you about the future?",
                "type": "multiple_choice",
                "category": "psych",
                "options": ["Very optimistic", "Somewhat optimistic", "Neutral", "Somewhat pessimistic", "Very pessimistic"]
            }
        ])
        
        return questions
    
    def _categorize_questions(
        self, 
        questions: List[Dict[str, Any]], 
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Categorize questions and assign proper identifiers
        
        Args:
            questions: List of question dictionaries
            requirements: Survey requirements
            
        Returns:
            List of categorized questions with proper structure
        """
        categorized = []
        
        for question in questions:
            # Use the meaningful identifier from the question definition
            question_id = question.get("identifier", f"q_{len(categorized) + 1}")
            
            categorized.append({
                "question_id": question_id,
                "category": question["category"],
                "question_text": question["text"],
                "question_type": question.get("type", "text"),
                "options": question.get("options", []),
                "scale": question.get("scale", ""),
                "required": True  # Default to required
            })
        
        return categorized
    
    def _create_survey_dataframe(self, survey_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a DataFrame in the required format for the survey system
        
        Format (matching existing survey structure):
        Row 1: Meaningful identifiers (age, gender, gaming_frequency, etc.)
        Row 2: Categories (demo, psych, text, ignore)
        Row 3: Question text/names
        """
        questions = survey_data["questions"]
        
        # Extract data for each row
        identifiers = [q["question_id"] for q in questions]
        categories = [q["category"] for q in questions]
        question_texts = [q["question_text"] for q in questions]
        
        # Create DataFrame with identifiers as column names
        data = {
            identifier: [category, question_text] 
            for identifier, category, question_text in zip(identifiers, categories, question_texts)
        }
        
        df = pd.DataFrame(data)
        
        # Set row index to match the expected format
        df.index = ['Categories', 'Questions']
        
        # Transpose so that each column represents a question
        # This creates the structure where:
        # - Column headers are the meaningful identifiers (age, gender, etc.)
        # - Row 1 contains categories (demo, psych, text, ignore)
        # - Row 2 contains question text
        df = df.T
        
        # Reset index to make identifiers part of the data
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Identifier'}, inplace=True)
        
        # Reorganize to match expected format:
        # We want identifiers as the first row, not as a separate column
        final_data = []
        
        # Row 1: Identifiers
        final_data.append(identifiers)
        
        # Row 2: Categories  
        final_data.append(categories)
        
        # Row 3: Question texts
        final_data.append(question_texts)
        
        # Create final DataFrame with identifiers as column names
        final_df = pd.DataFrame(final_data, columns=identifiers)
        
        return final_df
    
    def get_generated_files(self) -> List[Dict[str, Any]]:
        """Get list of all generated survey files"""
        try:
            files = []
            for filename in os.listdir(self.output_directory):
                if filename.endswith(('.csv', '.xlsx')):
                    filepath = os.path.join(self.output_directory, filename)
                    stat = os.stat(filepath)
                    files.append({
                        "filename": filename,
                        "filepath": filepath,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "type": "Excel" if filename.endswith('.xlsx') else "CSV"
                    })
            
            # Sort by creation time (newest first)
            files.sort(key=lambda x: x["created"], reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"Error getting generated files: {str(e)}")
            return []


# Global instance
survey_generation_service = SurveyGenerationService()