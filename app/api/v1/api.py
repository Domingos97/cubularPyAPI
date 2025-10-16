from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth, 
    users,  
    surveys, 
    fast_search,
    streamlined_chat,  
    access_control,  
    plans,  
    user_plans, 
    ai_personalities, 
    llm_settings,
    module_configurations,
    survey_builder, 
    health,
    logs,  
    notifications,  
    languages
)

api_router = APIRouter()

# Core functionality (working with lightweight_db)
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])  # CONVERTED: Using lightweight_db
api_router.include_router(surveys.router, prefix="/surveys", tags=["surveys"])  # CONVERTED: my-surveys endpoint

# NEW: Ultra-fast search (direct file access, no database) - consolidated search functionality
api_router.include_router(fast_search.router, prefix="/fast-search", tags=["fast-search", "performance"])

api_router.include_router(streamlined_chat.router, prefix="/chat", tags=["streamlined-chat", "ultra-fast"])

# Access Control (CONVERTED: Using lightweight_db)
api_router.include_router(access_control.router, prefix="/admin/access", tags=["access-control", "admin"])

# CONVERTED: All these endpoints now use lightweight_db
api_router.include_router(plans.router, prefix="/plans", tags=["plans"])
api_router.include_router(user_plans.router, prefix="/user-plans", tags=["user-plans"])
api_router.include_router(ai_personalities.router, prefix="/personalities", tags=["ai-personalities"])
api_router.include_router(llm_settings.router, prefix="/llm-settings", tags=["llm-settings"])
api_router.include_router(module_configurations.router, prefix="/module-configurations", tags=["module-configurations"])

# NEW: Survey Builder - Chat-based survey creation
api_router.include_router(survey_builder.router, prefix="/survey-builder", tags=["survey-builder", "ai-chat", "survey-generation"])

api_router.include_router(logs.router, prefix="/logs", tags=["logs"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])  # CONVERTED: my endpoint

# System Management (working)
api_router.include_router(health.router, prefix="/health", tags=["health"])

# Enhanced Features (CONVERTED: enabled endpoint)
api_router.include_router(languages.router, prefix="/languages", tags=["languages", "multilingual"])
