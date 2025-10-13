from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth, 
    users, 
    surveys, 
    # vector_search,  # REMOVED - Consolidated into fast_search for performance
    fast_search,  # Ultra-fast direct file search (consolidated search functionality)
    # chat,  # REMOVED - Replaced with streamlined_chat for performance
    # optimized_chat,  # REMOVED - Replaced with streamlined_chat
    streamlined_chat,  # NEWEST - Ultra-lightweight chat with minimal overhead
    access_control, 
    plans,
    user_plans,
    ai_personalities,
    llm_settings,
    module_configurations,
    health,
    logs,
    notifications,
    languages,
    # enhanced_surveys,  # COMMENTED OUT - Extra feature
    # advanced_chat,  # REMOVED - Heavy overhead, replaced with streamlined_chat
    # enhanced_survey_rag_chat,  # REMOVED - Heavy overhead, functionality in streamlined_chat
    # Phase 4: Critical Infrastructure (COMMENTED OUT - Not exposed in TypeScript)
    # roles,
    # refresh_tokens,
    # email,
    prompt_translations,  # UNCOMMENTED - TypeScript has this active
    # simple_suggestions
)

api_router = APIRouter()

# Core functionality
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(surveys.router, prefix="/surveys", tags=["surveys"])

# NEW: Ultra-fast search (direct file access, no database) - consolidated search functionality
api_router.include_router(fast_search.router, prefix="/fast-search", tags=["fast-search", "performance"])

# REMOVED - Vector search consolidated into fast_search for better performance
# api_router.include_router(vector_search.router, prefix="/vector", tags=["vector-search"])

# REMOVED - Old chat endpoints replaced with streamlined_chat for performance
# api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
# api_router.include_router(optimized_chat.router, prefix="/chat", tags=["optimized-chat", "performance"])

# ULTRA-FAST: New streamlined chat with minimal overhead (replaces all chat endpoints)
api_router.include_router(streamlined_chat.router, prefix="/chat", tags=["streamlined-chat", "ultra-fast"])
api_router.include_router(access_control.router, prefix="/admin/access", tags=["access-control"])
api_router.include_router(plans.router, prefix="/plans", tags=["plans"])

# User plan management (matches TypeScript /api/user/plan pattern)
api_router.include_router(user_plans.router, prefix="/user/plan", tags=["user-plans", "billing"])

# Route aliases to match TypeScript API patterns
# Admin access routes alias
api_router.include_router(access_control.router, prefix="/admin", tags=["admin", "access-control"])

# Plans alias route (matches TypeScript /api/plans pattern)
api_router.include_router(plans.router, prefix="/plans", tags=["plans-alias", "admin-plans"])

# AI Infrastructure (Phase 1)
api_router.include_router(ai_personalities.router, prefix="/personalities", tags=["ai-personalities"])
api_router.include_router(llm_settings.router, prefix="/llm-settings", tags=["llm-settings"])
api_router.include_router(module_configurations.router, prefix="/module-configurations", tags=["module-configurations"])

# System Management (Phase 2)
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(logs.router, prefix="/logs", tags=["logs"])
api_router.include_router(notifications.router, prefix="/notifications", tags=["notifications"])

# Enhanced Features (Phase 3)
api_router.include_router(languages.router, prefix="/languages", tags=["languages", "multilingual"])

# COMMENTED OUT - Extra features not in active TypeScript API
# api_router.include_router(enhanced_surveys.router, prefix="/enhanced-surveys", tags=["enhanced-surveys", "ai-suggestions"])
# api_router.include_router(advanced_chat.router, prefix="/advanced-chat", tags=["advanced-chat", "file-uploads"])

# COMMENTED OUT - Enhanced Survey RAG Chat (TypeScript has this but commented out)
# api_router.include_router(enhanced_survey_rag_chat.router, prefix="/enhanced-survey-rag-chat", tags=["enhanced-survey-rag-chat", "semantic-chat"])

# REACTIVATED - Prompt translations (TypeScript has this active under /api/admin)
api_router.include_router(prompt_translations.router, prefix="/admin", tags=["prompt-translations", "multilingual"])

# COMMENTED OUT - Extra infrastructure not exposed in TypeScript API
# api_router.include_router(roles.router, prefix="/roles", tags=["roles", "authorization"])
# api_router.include_router(refresh_tokens.router, prefix="/tokens", tags=["tokens", "authentication"])
# api_router.include_router(email.router, prefix="/email", tags=["email", "notifications"])
# api_router.include_router(simple_suggestions.router, prefix="/surveys", tags=["simple-suggestions", "ai-analysis"])