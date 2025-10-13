from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import asyncio
import uuid
import psutil
import platform

from app.core.database import get_db, get_database_health
from app.models.schemas import HealthCheck
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=HealthCheck)
async def health_check():
    """
    Basic health check endpoint
    
    Returns system health status, version, and timestamp
    """
    try:
        return HealthCheck(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.utcnow(),
            database="not_checked",
            services={}
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@router.get("/detailed")
async def detailed_health_check(db: AsyncSession = Depends(get_db)):
    """
    Detailed health check with database connectivity and system metrics
    
    Returns comprehensive system health information
    """
    try:
        health_data = {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.utcnow(),
            "database": "unknown",
            "services": {},
            "system": {},
            "checks": {}
        }
        
        # Database connectivity check with pool monitoring
        try:
            db_health = await get_database_health()
            health_data["database"] = db_health
            
            if db_health["status"] == "healthy":
                health_data["checks"]["database"] = "passed"
            else:
                health_data["checks"]["database"] = f"failed: {db_health.get('error', 'unknown')}"
                health_data["status"] = "degraded"
        except Exception as db_error:
            logger.error(f"Database health check failed: {str(db_error)}")
            health_data["database"] = {"status": "error", "error": str(db_error)}
            health_data["checks"]["database"] = f"failed: {str(db_error)}"
            health_data["status"] = "degraded"
        
        # System metrics
        try:
            health_data["system"] = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent if platform.system() != "Windows" else psutil.disk_usage('C:').percent,
                "uptime_seconds": (datetime.utcnow() - datetime.fromtimestamp(psutil.boot_time())).total_seconds()
            }
            health_data["checks"]["system_metrics"] = "passed"
        except Exception as sys_error:
            logger.error(f"System metrics check failed: {str(sys_error)}")
            health_data["checks"]["system_metrics"] = f"failed: {str(sys_error)}"
        
        # Service checks
        health_data["services"] = {
            "api": "running",
            "database": health_data["database"],
            "encryption": "configured" if _check_encryption_service() else "not_configured"
        }
        
        # Overall status determination
        if health_data["database"] == "error":
            health_data["status"] = "unhealthy"
        elif "failed" in str(health_data["checks"]):
            health_data["status"] = "degraded"
        
        return health_data
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Detailed health check failed: {str(e)}"
        )


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Readiness check for Kubernetes/container orchestration
    
    Returns 200 if service is ready to handle requests
    """
    try:
        # Test critical components
        checks = {}
        overall_ready = True
        
        # Database readiness
        try:
            await db.execute("SELECT 1")
            checks["database"] = "ready"
        except Exception as e:
            checks["database"] = f"not_ready: {str(e)}"
            overall_ready = False
        
        # Encryption service readiness
        try:
            encryption_ready = _check_encryption_service()
            checks["encryption"] = "ready" if encryption_ready else "not_ready"
            if not encryption_ready:
                overall_ready = False
        except Exception as e:
            checks["encryption"] = f"not_ready: {str(e)}"
            overall_ready = False
        
        if overall_ready:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow(),
                "checks": checks
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "status": "not_ready",
                    "timestamp": datetime.utcnow(),
                    "checks": checks
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes/container orchestration
    
    Returns 200 if service is alive (basic functionality check)
    """
    try:
        # Basic liveness checks
        checks = {
            "process": "alive",
            "timestamp": datetime.utcnow(),
            "response_time_ms": 0
        }
        
        start_time = datetime.utcnow()
        
        # Simple computation to verify process is responsive
        test_value = sum(range(1000))
        if test_value == 499500:  # Expected sum of 0 to 999
            checks["computation"] = "responsive"
        else:
            checks["computation"] = "error"
        
        end_time = datetime.utcnow()
        checks["response_time_ms"] = (end_time - start_time).total_seconds() * 1000
        
        return {
            "status": "alive",
            "checks": checks
        }
        
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Liveness check failed: {str(e)}"
        )


@router.get("/dependencies")
async def dependencies_check(db: AsyncSession = Depends(get_db)):
    """
    Check health of external dependencies
    
    Returns status of all external services and dependencies
    """
    try:
        dependencies = {}
        overall_healthy = True
        
        # Database dependency
        try:
            await db.execute("SELECT version()")
            dependencies["postgresql"] = {
                "status": "healthy",
                "type": "database",
                "critical": True
            }
        except Exception as e:
            dependencies["postgresql"] = {
                "status": "unhealthy",
                "type": "database", 
                "critical": True,
                "error": str(e)
            }
            overall_healthy = False
        
        # AI Services (would be checked if configured)
        dependencies["openai_service"] = {
            "status": "not_configured",
            "type": "ai_service",
            "critical": False,
            "note": "Requires API key configuration"
        }
        
        dependencies["anthropic_service"] = {
            "status": "not_configured",
            "type": "ai_service", 
            "critical": False,
            "note": "Requires API key configuration"
        }
        
        # Fast search service (replaced vector_search)
        dependencies["fast_search"] = {
            "status": "internal",
            "type": "search_service", 
            "critical": True,
            "note": "Uses direct file access for ultra-fast search"
        }
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow(),
            "dependencies": dependencies,
            "summary": {
                "total": len(dependencies),
                "healthy": sum(1 for d in dependencies.values() if d["status"] == "healthy"),
                "unhealthy": sum(1 for d in dependencies.values() if d["status"] == "unhealthy"),
                "not_configured": sum(1 for d in dependencies.values() if d["status"] == "not_configured")
            }
        }
        
    except Exception as e:
        logger.error(f"Dependencies check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Dependencies check failed: {str(e)}"
        )


def _check_encryption_service() -> bool:
    """Check if encryption service is properly configured"""
    try:
        from app.utils.encryption import encryption_service
        # Try to encrypt and decrypt a test string
        test_data = "health_check_test"
        encrypted = encryption_service.encrypt(test_data)
        decrypted = encryption_service.decrypt(encrypted)
        return decrypted == test_data
    except Exception:
        return False


@router.get("/db")
async def database_health_check(db: AsyncSession = Depends(get_db)):
    """
    Database health check endpoint (matches TypeScript API)
    
    Tests database connectivity and basic operations
    """
    try:
        # Test basic database connectivity
        from sqlalchemy import text
        result = await db.execute(text("SELECT 1"))
        db_value = result.scalar()
        
        if db_value != 1:
            raise Exception("Database query returned unexpected result")
        
        # Test transaction capability
        await db.execute(text("BEGIN"))
        await db.execute(text("ROLLBACK"))
        
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "connectivity": "pass",
                "transactions": "pass"
            }
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "database": "disconnected",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )