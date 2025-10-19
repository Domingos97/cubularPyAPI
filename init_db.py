"""
Database initialization script for CubularPyAPI

This script sets up the database tables using SQLAlchemy models.
Run this before starting the API for the first time.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.database import engine
from app.core.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def init_database():
    """Database connection test - tables should already exist"""
    try:
        logger.info("Testing database connection...")
        logger.info(f"Database URL: {settings.database_url}")
        
        # Test connection only - no table creation
        await check_database_connection()
        
        logger.info("✅ Database connection successful!")
        logger.info("Note: This script no longer creates tables - they should already exist")
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")
        raise


async def check_database_connection():
    """Check if we can connect to the database"""
    try:
        from sqlalchemy import text
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")
        return False


async def main():
    """Main initialization function"""
    print("🚀 CubularPyAPI Database Initialization")
    print("=" * 50)
    
    # Check database connection first
    if not await check_database_connection():
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your DATABASE_URL in .env file")
        print("3. Ensure the database exists and credentials are correct")
        return
    
    # Initialize database
    await init_database()
    
    print("\n🎉 Setup complete! Next steps:")
    print("1. Start the API: python start_server.py")
    print("2. Visit https://cubularpyfront-production.up.railway.app/docs for API documentation")
    print("3. Create your first user via the /auth/register endpoint")


if __name__ == "__main__":
    asyncio.run(main())