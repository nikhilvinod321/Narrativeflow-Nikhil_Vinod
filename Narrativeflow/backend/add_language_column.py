"""
Migration script to add language column to stories table
Run this script to add multi-language support to existing stories
"""
import asyncio
from sqlalchemy import text
from app.database import async_session_maker, engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def add_language_column():
    """Add language column to stories table"""
    async with engine.begin() as conn:
        try:
            # Add language column with default value
            await conn.execute(text("""
                ALTER TABLE stories 
                ADD COLUMN IF NOT EXISTS language VARCHAR(50) DEFAULT 'English'
            """))
            
            # Update existing records to have English as default
            await conn.execute(text("""
                UPDATE stories 
                SET language = 'English' 
                WHERE language IS NULL
            """))
            
            logger.info("✓ Successfully added language column to stories table")
            logger.info("✓ All existing stories set to English")
            
        except Exception as e:
            logger.error(f"✗ Error adding language column: {e}")
            raise


async def main():
    """Run the migration"""
    logger.info("Starting migration: Adding language support...")
    await add_language_column()
    logger.info("Migration complete!")


if __name__ == "__main__":
    asyncio.run(main())
