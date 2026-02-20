"""
Migration script to add missing columns to characters table
"""
import asyncio
from sqlalchemy import text
from app.database import engine

async def add_columns():
    async with engine.begin() as conn:
        # List of columns that might be missing from the characters table
        columns_to_add = [
            ("image_generation_seed", "INTEGER"),
            ("visual_style", "VARCHAR(100)"),
            ("reference_images", "JSONB DEFAULT '[]'::jsonb"),
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                await conn.execute(text(
                    f"ALTER TABLE characters ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                ))
                print(f"✓ Added/verified column: {col_name}")
            except Exception as e:
                print(f"✗ Error with {col_name}: {e}")
        
        print("\n✓ Migration complete!")

if __name__ == "__main__":
    asyncio.run(add_columns())
    print("You can now restart the backend server.")
