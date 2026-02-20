"""
Character Service - Business logic for character management
"""
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.character import Character, CharacterRole, CharacterStatus


class CharacterService:
    """Service for managing characters"""
    
    async def create_character(
        self,
        db: AsyncSession,
        story_id: UUID,
        name: str,
        role: CharacterRole = CharacterRole.SUPPORTING,
        full_name: Optional[str] = None,
        age: Optional[str] = None,
        gender: Optional[str] = None,
        species: Optional[str] = "human",
        occupation: Optional[str] = None,
        physical_description: Optional[str] = None,
        personality_summary: Optional[str] = None,
        personality_traits: Optional[List[str]] = None,
        backstory: Optional[str] = None,
        motivation: Optional[str] = None,
        speaking_style: Optional[str] = None,
        distinguishing_features: Optional[List[str]] = None
    ) -> Character:
        """Create a new character"""
        character = Character(
            story_id=story_id,
            name=name,
            role=role,
            full_name=full_name,
            age=age,
            gender=gender,
            species=species or "human",
            occupation=occupation,
            physical_description=physical_description,
            personality_summary=personality_summary,
            personality_traits=personality_traits or [],
            backstory=backstory,
            motivation=motivation,
            speaking_style=speaking_style,
            distinguishing_features=distinguishing_features or [],
            status=CharacterStatus.ALIVE,
            importance=self._calculate_importance(role)
        )
        
        db.add(character)
        await db.flush()
        return character
    
    async def get_character(
        self,
        db: AsyncSession,
        character_id: UUID
    ) -> Optional[Character]:
        """Get a character by ID"""
        query = select(Character).where(Character.id == character_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_characters_by_story(
        self,
        db: AsyncSession,
        story_id: UUID,
        role_filter: Optional[CharacterRole] = None
    ) -> List[Character]:
        """Get all characters for a story"""
        query = select(Character).where(Character.story_id == story_id)
        
        if role_filter:
            query = query.where(Character.role == role_filter)
        
        query = query.order_by(Character.importance.desc(), Character.name)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def get_main_characters(
        self,
        db: AsyncSession,
        story_id: UUID
    ) -> List[Character]:
        """Get main characters (protagonist, antagonist, deuteragonist)"""
        main_roles = [
            CharacterRole.PROTAGONIST,
            CharacterRole.ANTAGONIST,
            CharacterRole.DEUTERAGONIST
        ]
        
        query = (
            select(Character)
            .where(Character.story_id == story_id)
            .where(Character.role.in_(main_roles))
            .order_by(Character.importance.desc())
        )
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def update_character(
        self,
        db: AsyncSession,
        character_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[Character]:
        """Update character fields"""
        character = await self.get_character(db, character_id)
        if not character:
            return None
        
        for key, value in updates.items():
            if hasattr(character, key):
                setattr(character, key, value)
        
        character.updated_at = datetime.utcnow()
        await db.flush()
        return character
    
    async def update_character_state(
        self,
        db: AsyncSession,
        character_id: UUID,
        emotional_state: Optional[str] = None,
        location: Optional[str] = None,
        goals: Optional[List[str]] = None,
        knowledge: Optional[List[str]] = None
    ) -> Optional[Character]:
        """Update character's current state (for tracking during story)"""
        updates = {}
        if emotional_state is not None:
            updates["current_emotional_state"] = emotional_state
        if location is not None:
            updates["current_location"] = location
        if goals is not None:
            updates["current_goals"] = goals
        if knowledge is not None:
            updates["knowledge"] = knowledge
        
        return await self.update_character(db, character_id, updates)
    
    async def delete_character(
        self,
        db: AsyncSession,
        character_id: UUID
    ) -> bool:
        """Delete a character"""
        character = await self.get_character(db, character_id)
        if not character:
            return False
        
        await db.delete(character)
        await db.flush()
        return True
    
    async def add_relationship(
        self,
        db: AsyncSession,
        character_id: UUID,
        related_character_id: UUID,
        relationship_type: str,
        description: Optional[str] = None
    ) -> Optional[Character]:
        """Add a relationship to another character"""
        character = await self.get_character(db, character_id)
        if not character:
            return None
        
        relationships = character.relationships or []
        relationships.append({
            "character_id": str(related_character_id),
            "type": relationship_type,
            "description": description
        })
        
        character.relationships = relationships
        await db.flush()
        return character
    
    async def get_character_for_ai(
        self,
        db: AsyncSession,
        character_id: UUID
    ) -> Dict[str, Any]:
        """Get character data formatted for AI context"""
        character = await self.get_character(db, character_id)
        if not character:
            return {}
        
        return {
            "name": character.name,
            "role": character.role.value,
            "personality": character.personality_summary,
            "speaking_style": character.speaking_style,
            "catchphrases": character.catchphrases,
            "backstory_summary": character.backstory[:500] if character.backstory else None,
            "motivation": character.motivation,
            "current_state": character.current_emotional_state,
            "current_location": character.current_location
        }
    
    def _calculate_importance(self, role: CharacterRole) -> int:
        """Calculate default importance based on role"""
        importance_map = {
            CharacterRole.PROTAGONIST: 10,
            CharacterRole.ANTAGONIST: 9,
            CharacterRole.DEUTERAGONIST: 8,
            CharacterRole.LOVE_INTEREST: 7,
            CharacterRole.MENTOR: 6,
            CharacterRole.SIDEKICK: 6,
            CharacterRole.SUPPORTING: 5,
            CharacterRole.FOIL: 5,
            CharacterRole.NARRATOR: 4,
            CharacterRole.MINOR: 3
        }
        return importance_map.get(role, 5)
