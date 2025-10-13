# Database connection and session management
# The actual database models are in models.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
from datetime import datetime
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create async engine with optimized connection pooling
engine = create_async_engine(
    str(settings.database_url),
    echo=settings.debug,  # Log SQL queries in debug mode
    future=True,
    # Connection pool settings (matching TypeScript API configuration)
    pool_size=10,           # Number of connections to maintain
    max_overflow=10,        # Additional connections that can be created
    pool_timeout=30,        # Seconds to wait for connection from pool
    pool_recycle=3600,      # Recycle connections after 1 hour
    pool_pre_ping=True,     # Validate connections before use
    # Async-specific optimizations
    pool_reset_on_return='rollback',  # Reset connection state on return
)

# Create async session factory with optimized settings
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
    # Additional optimizations
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            await session.close()


def get_db_session():
    """Context manager to get database session for use in services"""
    return AsyncSessionLocal()


async def close_db() -> None:
    """Close database connections"""
    await engine.dispose()


async def get_database_health() -> dict:
    """Get database connection health status"""
    try:
        async with AsyncSessionLocal() as session:
            # Simple health check query
            result = await session.execute(text("SELECT 1"))
            result.fetchone()
            
            # Get pool status
            pool = engine.pool
            return {
                "status": "healthy",
                "pool_size": pool.size() if hasattr(pool, 'size') else "unknown",
                "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else "unknown",
                "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else "unknown",
                "overflow": pool.overflow() if hasattr(pool, 'overflow') else "unknown",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    logger.info("Database connections closed")