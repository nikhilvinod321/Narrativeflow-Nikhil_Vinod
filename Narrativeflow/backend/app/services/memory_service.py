"""
Memory Service - Vector embeddings and semantic retrieval (RAG)
Implements long-term narrative memory for story consistency using Ollama embeddings
"""
from typing import Optional, List, Dict, Any, Tuple
import asyncio
import logging
import hashlib
import re
import httpx
from datetime import datetime
import numpy as np

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.config import settings
from app.models.embedding import StoryEmbedding, CharacterEmbedding

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Service for managing story memory using vector embeddings
    Implements RAG (Retrieval-Augmented Generation) for narrative consistency
    Uses Ollama for local embeddings (nomic-embed-text)
    """
    
    # Scene type patterns for classification
    SCENE_PATTERNS = {
        "dialogue": [r'"[^"]+"\s*,?\s*(said|asked|replied|whispered|shouted|muttered)', r"dialogue", r"conversation"],
        "action": [r"(ran|jumped|fought|attacked|dodged|sprinted|grabbed)", r"action sequence", r"battle"],
        "description": [r"(the room|the landscape|the building|the sky|the scene)", r"description", r"setting"],
        "introspection": [r"(thought|wondered|felt|realized|remembered)", r"reflection", r"inner thought"],
        "flashback": [r"(years ago|remembered when|back then|in the past)", r"memory", r"flashback"],
        "revelation": [r"(revealed|discovered|learned|found out|the truth)", r"twist", r"revelation"],
    }
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.embedding_model = settings.embedding_model
        self.embedding_dimension = settings.embedding_dimension
        self.ollama_base_url = settings.ollama_base_url
        
        # Initialize ChromaDB client with new API
        if CHROMA_AVAILABLE:
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=settings.chroma_persist_directory
                )
                logger.info("ChromaDB initialized with persistent storage")
            except Exception as e:
                logger.warning(f"Failed to initialize persistent ChromaDB: {e}")
                self.chroma_client = chromadb.EphemeralClient()
                logger.info("ChromaDB initialized with ephemeral storage")
        else:
            self.chroma_client = None
            logger.warning("ChromaDB not available - using fallback storage")
        
        # HTTP client for Ollama
        self.http_client = httpx.AsyncClient(timeout=120.0)
    
    # ==========================================================================
    # CHAPTER EMBEDDING
    # ==========================================================================
    
    async def embed_chapter(
        self,
        db: AsyncSession,
        story_id: str,
        chapter_id: str,
        content: str,
        chapter_metadata: Dict[str, Any]
    ) -> List[StoryEmbedding]:
        """
        Embed a chapter's content into semantic chunks with rich metadata
        
        Process:
        1. Chunk the text into semantic units (paragraphs/scenes)
        2. Extract metadata for each chunk (characters, scene type, importance)
        3. Generate embeddings using Ollama
        4. Store in PostgreSQL and ChromaDB
        """
        if not content or not content.strip():
            logger.warning(f"Empty content for chapter {chapter_id}, skipping embedding")
            return []
        
        # Clear existing embeddings for this chapter
        await self._clear_chapter_embeddings(db, chapter_id)
        
        # Chunk the content with smart boundaries
        chunks = self._chunk_text(content)
        
        if not chunks:
            logger.warning(f"No chunks generated for chapter {chapter_id}")
            return []
        
        # Extract metadata for each chunk
        enriched_chunks = []
        for chunk in chunks:
            metadata = self._extract_chunk_metadata(
                chunk["text"],
                chapter_metadata.get("characters", [])
            )
            enriched_chunks.append({
                **chunk,
                **metadata
            })
        
        # Generate embeddings using Ollama
        embeddings = await self._generate_embeddings([c["text"] for c in enriched_chunks])
        
        if not embeddings:
            logger.error(f"Failed to generate embeddings for chapter {chapter_id}")
            return []
        
        # Create embedding records
        embedding_records = []
        for i, (chunk, embedding) in enumerate(zip(enriched_chunks, embeddings)):
            record = StoryEmbedding(
                story_id=story_id,
                chapter_id=chapter_id,
                content=chunk["text"],
                content_type="chapter",
                chunk_index=i,
                chunk_size=len(chunk["text"]),
                start_position=chunk["start"],
                end_position=chunk["end"],
                embedding=embedding,
                embedding_model=self.embedding_model,
                summary=chunk.get("summary"),
                key_entities=chunk.get("characters", []),
                key_events=chunk.get("events", []),
                emotional_tone=chunk.get("emotional_tone")
            )
            embedding_records.append(record)
            db.add(record)
        
        await db.flush()
        
        # Store in ChromaDB for fast retrieval
        if self.chroma_client:
            await self._store_in_chroma(
                collection_name=f"story_{story_id}",
                chunks=enriched_chunks,
                embeddings=embeddings,
                metadata=[{
                    "chapter_id": chapter_id,
                    "chunk_index": i,
                    "scene_type": enriched_chunks[i].get("scene_type", "unknown"),
                    "importance": enriched_chunks[i].get("importance", 5),
                    "content_type": "chapter"
                } for i in range(len(enriched_chunks))]
            )
        
        logger.info(f"✓ Embedded chapter {chapter_id} into {len(enriched_chunks)} chunks with metadata")
        return embedding_records
    
    # ==========================================================================
    # CHARACTER EMBEDDING
    # ==========================================================================
    
    async def embed_character(
        self,
        db: AsyncSession,
        character_id: str,
        story_id: str,
        character_data: Dict[str, Any]
    ) -> List[CharacterEmbedding]:
        """
        Embed all aspects of a character for semantic retrieval
        Creates multiple embeddings: profile, personality, backstory, relationships
        """
        records = []
        
        # Clear existing embeddings
        await self._clear_character_embeddings(db, character_id)
        
        # Build profile text
        profile_parts = []
        if character_data.get("name"):
            profile_parts.append(f"Character: {character_data['name']}")
        if character_data.get("role"):
            profile_parts.append(f"Role: {character_data['role']}")
        if character_data.get("physical_description"):
            profile_parts.append(f"Appearance: {character_data['physical_description']}")
        if character_data.get("personality_summary"):
            profile_parts.append(f"Personality: {character_data['personality_summary']}")
        if character_data.get("occupation"):
            profile_parts.append(f"Occupation: {character_data['occupation']}")
        
        profile_text = "\n".join(profile_parts)
        
        if profile_text:
            embedding = await self._generate_embeddings([profile_text])
            if embedding:
                record = CharacterEmbedding(
                    character_id=character_id,
                    content_type="profile",
                    content=profile_text,
                    embedding=embedding[0]
                )
                db.add(record)
                records.append(record)
        
        # Embed backstory separately if present
        if character_data.get("backstory"):
            backstory_text = f"{character_data['name']}'s backstory: {character_data['backstory']}"
            embedding = await self._generate_embeddings([backstory_text])
            if embedding:
                record = CharacterEmbedding(
                    character_id=character_id,
                    content_type="backstory",
                    content=backstory_text,
                    embedding=embedding[0]
                )
                db.add(record)
                records.append(record)
        
        # Embed speaking style for dialogue matching
        if character_data.get("speaking_style"):
            voice_text = f"{character_data['name']}'s voice and speaking style: {character_data['speaking_style']}"
            if character_data.get("catchphrases"):
                voice_text += f"\nCatchphrases: {', '.join(character_data['catchphrases'])}"
            embedding = await self._generate_embeddings([voice_text])
            if embedding:
                record = CharacterEmbedding(
                    character_id=character_id,
                    content_type="voice",
                    content=voice_text,
                    embedding=embedding[0]
                )
                db.add(record)
                records.append(record)
        
        # Embed motivation/goals
        if character_data.get("motivation") or character_data.get("current_goals"):
            goals_text = f"{character_data['name']}'s motivations and goals:\n"
            if character_data.get("motivation"):
                goals_text += f"Motivation: {character_data['motivation']}\n"
            if character_data.get("current_goals"):
                goals_text += f"Current goals: {', '.join(character_data['current_goals'])}"
            embedding = await self._generate_embeddings([goals_text])
            if embedding:
                record = CharacterEmbedding(
                    character_id=character_id,
                    content_type="motivation",
                    content=goals_text,
                    embedding=embedding[0]
                )
                db.add(record)
                records.append(record)
        
        await db.flush()
        
        # Also store in character-specific ChromaDB collection
        if self.chroma_client and records:
            await self._store_in_chroma(
                collection_name=f"story_{story_id}_characters",
                chunks=[{"text": r.content} for r in records],
                embeddings=[r.embedding for r in records],
                metadata=[{
                    "character_id": character_id,
                    "character_name": character_data.get("name", ""),
                    "content_type": r.content_type
                } for r in records]
            )
        
        logger.info(f"✓ Embedded character {character_data.get('name', character_id)} with {len(records)} aspects")
        return records
    
    # ==========================================================================
    # STORY BIBLE EMBEDDING
    # ==========================================================================
    
    async def embed_story_bible(
        self,
        db: AsyncSession,
        story_id: str,
        story_bible: Any  # StoryBible model
    ) -> int:
        """
        Embed story bible entries for semantic retrieval
        Embeds: world rules, locations, glossary terms, magic system, etc.
        """
        if not story_bible:
            return 0
        
        embedded_count = 0
        bible_collection = f"story_{story_id}_bible"
        
        chunks = []
        embeddings_list = []
        metadata_list = []
        
        # Embed world rules
        if story_bible.world_rules:
            for rule in story_bible.world_rules:
                rule_text = f"WORLD RULE [{rule.category.value if hasattr(rule.category, 'value') else rule.category}]: {rule.title}\n{rule.description}"
                embedding = await self._generate_embeddings([rule_text])
                if embedding:
                    chunks.append({"text": rule_text})
                    embeddings_list.append(embedding[0])
                    metadata_list.append({
                        "type": "world_rule",
                        "category": rule.category.value if hasattr(rule.category, 'value') else str(rule.category),
                        "importance": rule.importance,
                        "is_strict": rule.is_strict
                    })
                    embedded_count += 1
        
        # Embed key locations
        if story_bible.primary_locations:
            for loc in story_bible.primary_locations:
                if isinstance(loc, dict):
                    loc_text = f"LOCATION: {loc.get('name', 'Unknown')}\n{loc.get('description', '')}"
                    embedding = await self._generate_embeddings([loc_text])
                    if embedding:
                        chunks.append({"text": loc_text})
                        embeddings_list.append(embedding[0])
                        metadata_list.append({
                            "type": "location",
                            "name": loc.get("name", ""),
                            "importance": loc.get("importance", 5)
                        })
                        embedded_count += 1
        
        # Embed magic system
        if story_bible.magic_system:
            magic_text = f"MAGIC SYSTEM:\n{story_bible.magic_system}"
            if story_bible.magic_rules:
                magic_text += f"\n\nMagic Rules:\n" + "\n".join([f"- {r}" for r in story_bible.magic_rules[:10]])
            if story_bible.magic_limitations:
                magic_text += f"\n\nLimitations:\n" + "\n".join([f"- {l}" for l in story_bible.magic_limitations[:10]])
            
            embedding = await self._generate_embeddings([magic_text])
            if embedding:
                chunks.append({"text": magic_text})
                embeddings_list.append(embedding[0])
                metadata_list.append({"type": "magic_system", "importance": 10})
                embedded_count += 1
        
        # Embed glossary terms
        if story_bible.glossary:
            glossary = story_bible.glossary
            if isinstance(glossary, dict):
                for term, definition in glossary.items():
                    term_text = f"TERM: {term}\nDefinition: {definition}"
                    embedding = await self._generate_embeddings([term_text])
                    if embedding:
                        chunks.append({"text": term_text})
                        embeddings_list.append(embedding[0])
                        metadata_list.append({"type": "glossary", "term": term})
                        embedded_count += 1
            elif isinstance(glossary, list):
                for item in glossary:
                    if isinstance(item, dict):
                        term_text = f"TERM: {item.get('term', 'Unknown')}\nDefinition: {item.get('definition', '')}"
                        embedding = await self._generate_embeddings([term_text])
                        if embedding:
                            chunks.append({"text": term_text})
                            embeddings_list.append(embedding[0])
                            metadata_list.append({"type": "glossary", "term": item.get("term", "")})
                            embedded_count += 1
        
        # Embed central themes
        if story_bible.central_themes:
            themes_text = "CENTRAL THEMES:\n" + "\n".join([f"- {t}" for t in story_bible.central_themes])
            embedding = await self._generate_embeddings([themes_text])
            if embedding:
                chunks.append({"text": themes_text})
                embeddings_list.append(embedding[0])
                metadata_list.append({"type": "themes", "importance": 8})
                embedded_count += 1
        
        # Store all in ChromaDB
        if self.chroma_client and chunks:
            await self._store_in_chroma(
                collection_name=bible_collection,
                chunks=chunks,
                embeddings=embeddings_list,
                metadata=metadata_list
            )
        
        logger.info(f"✓ Embedded story bible for story {story_id}: {embedded_count} entries")
        return embedded_count
    
    # ==========================================================================
    # RETRIEVAL METHODS
    # ==========================================================================
    
    async def retrieve_relevant_context(
        self,
        story_id: str,
        query: str,
        top_k: int = 5,
        content_types: Optional[List[str]] = None,
        exclude_chapter_id: Optional[str] = None,
        min_importance: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant story context using semantic search
        
        This is the main RAG retrieval function that powers context-aware generation.
        """
        # Generate query embedding
        query_embedding = await self._generate_embeddings([query])
        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []
        
        # Query ChromaDB if available
        if self.chroma_client:
            try:
                collection = self.chroma_client.get_collection(f"story_{story_id}")
                
                # Build filter
                where_filter = {}
                if exclude_chapter_id:
                    where_filter["chapter_id"] = {"$ne": exclude_chapter_id}
                if content_types:
                    where_filter["content_type"] = {"$in": content_types}
                if min_importance > 0:
                    where_filter["importance"] = {"$gte": min_importance}
                
                results = collection.query(
                    query_embeddings=[query_embedding[0]],
                    n_results=top_k,
                    where=where_filter if where_filter else None
                )
                
                # Format results
                retrieved = []
                if results["documents"] and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        score = 1 - results["distances"][0][i] if results["distances"] else 0.5
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        retrieved.append({
                            "content": doc,
                            "score": score,
                            "metadata": metadata,
                            "source": "chapter"
                        })
                
                logger.debug(f"Retrieved {len(retrieved)} chunks for query")
                return retrieved
                
            except Exception as e:
                logger.warning(f"ChromaDB query failed: {e}")
        
        return []
    
    async def retrieve_character_context(
        self,
        story_id: str,
        character_ids: List[str],
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant character information based on current context"""
        query_embedding = await self._generate_embeddings([query])
        if not query_embedding:
            return []
        
        if self.chroma_client:
            try:
                collection = self.chroma_client.get_collection(f"story_{story_id}_characters")
                
                where_filter = None
                if character_ids:
                    where_filter = {"character_id": {"$in": character_ids}}
                
                results = collection.query(
                    query_embeddings=[query_embedding[0]],
                    n_results=top_k,
                    where=where_filter
                )
                
                retrieved = []
                if results["documents"] and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        score = 1 - results["distances"][0][i] if results["distances"] else 0.5
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        retrieved.append({
                            "content": doc,
                            "score": score,
                            "metadata": metadata,
                            "source": "character"
                        })
                
                return retrieved
                
            except Exception as e:
                logger.warning(f"Character retrieval failed: {e}")
        
        return []
    
    async def retrieve_story_bible_context(
        self,
        story_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant story bible entries based on current context"""
        query_embedding = await self._generate_embeddings([query])
        if not query_embedding:
            return []
        
        if self.chroma_client:
            try:
                collection = self.chroma_client.get_collection(f"story_{story_id}_bible")
                
                results = collection.query(
                    query_embeddings=[query_embedding[0]],
                    n_results=top_k
                )
                
                retrieved = []
                if results["documents"] and results["documents"][0]:
                    for i, doc in enumerate(results["documents"][0]):
                        score = 1 - results["distances"][0][i] if results["distances"] else 0.5
                        metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                        retrieved.append({
                            "content": doc,
                            "score": score,
                            "metadata": metadata,
                            "source": "bible"
                        })
                
                return retrieved
                
            except Exception as e:
                logger.warning(f"Story bible retrieval failed: {e}")
        
        return []
    
    async def retrieve_all_relevant_context(
        self,
        story_id: str,
        query: str,
        character_ids: Optional[List[str]] = None,
        exclude_chapter_id: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Comprehensive retrieval from all sources:
        - Story chapters (episodic memory)
        - Character profiles (character memory)
        - Story bible (canonical memory)
        
        Returns organized context by source type.
        """
        results = {
            "chapters": [],
            "characters": [],
            "bible": []
        }
        
        # Run all retrievals in parallel for speed
        chapter_task = self.retrieve_relevant_context(
            story_id=story_id,
            query=query,
            top_k=5,
            exclude_chapter_id=exclude_chapter_id
        )
        
        character_task = self.retrieve_character_context(
            story_id=story_id,
            character_ids=character_ids or [],
            query=query,
            top_k=3
        )
        
        bible_task = self.retrieve_story_bible_context(
            story_id=story_id,
            query=query,
            top_k=3
        )
        
        chapter_results, character_results, bible_results = await asyncio.gather(
            chapter_task, character_task, bible_task,
            return_exceptions=True
        )
        
        if isinstance(chapter_results, list):
            results["chapters"] = chapter_results
        if isinstance(character_results, list):
            results["characters"] = character_results
        if isinstance(bible_results, list):
            results["bible"] = bible_results
        
        total = len(results["chapters"]) + len(results["characters"]) + len(results["bible"])
        logger.debug(f"Retrieved total {total} context items (chapters: {len(results['chapters'])}, characters: {len(results['characters'])}, bible: {len(results['bible'])})")
        
        return results
    
    async def get_story_summary_context(
        self,
        story_id: str,
        max_chunks: int = 10
    ) -> str:
        """Get a comprehensive context summary for the entire story"""
        query = "story summary main events characters plot important moments"
        
        chunks = await self.retrieve_relevant_context(
            story_id=story_id,
            query=query,
            top_k=max_chunks
        )
        
        context_parts = []
        for chunk in chunks:
            context_parts.append(chunk["content"])
        
        return "\n\n---\n\n".join(context_parts)
    
    # ==========================================================================
    # METADATA EXTRACTION
    # ==========================================================================
    
    def _extract_chunk_metadata(
        self,
        text: str,
        known_characters: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extract rich metadata from a text chunk:
        - Characters mentioned
        - Scene type (dialogue, action, description, etc.)
        - Emotional tone
        - Importance score
        - Key events
        """
        metadata = {
            "characters": [],
            "scene_type": "unknown",
            "emotional_tone": None,
            "importance": 5,
            "events": [],
            "summary": None
        }
        
        text_lower = text.lower()
        
        # Detect characters mentioned
        if known_characters:
            for char_name in known_characters:
                if char_name.lower() in text_lower:
                    metadata["characters"].append(char_name)
        
        # Detect scene type
        for scene_type, patterns in self.SCENE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    metadata["scene_type"] = scene_type
                    break
            if metadata["scene_type"] != "unknown":
                break
        
        # Detect emotional tone
        emotional_indicators = {
            "tense": ["suddenly", "heart pounded", "danger", "threat", "fear"],
            "sad": ["tears", "grief", "loss", "mourning", "sorrow"],
            "happy": ["smiled", "laughed", "joy", "delight", "celebration"],
            "angry": ["furious", "rage", "anger", "shouted", "stormed"],
            "mysterious": ["strange", "mysterious", "unknown", "secret", "hidden"],
            "romantic": ["love", "kiss", "embrace", "heart", "tender"],
        }
        
        for tone, indicators in emotional_indicators.items():
            if any(ind in text_lower for ind in indicators):
                metadata["emotional_tone"] = tone
                break
        
        # Calculate importance score (1-10)
        importance = 5
        
        # Boost for character presence
        if len(metadata["characters"]) > 0:
            importance += 1
        if len(metadata["characters"]) > 2:
            importance += 1
        
        # Boost for revelations/twists
        if metadata["scene_type"] == "revelation":
            importance += 2
        
        # Boost for dialogue (character development)
        if metadata["scene_type"] == "dialogue":
            importance += 1
        
        # Boost for action (plot movement)
        if metadata["scene_type"] == "action":
            importance += 1
        
        # Cap at 10
        metadata["importance"] = min(importance, 10)
        
        # Extract potential events (simple heuristic)
        event_patterns = [
            r"([A-Z][a-z]+) (died|was killed|fell|discovered|revealed|married|betrayed)",
            r"(the|a|an) ([a-z]+) (exploded|collapsed|appeared|vanished)",
        ]
        for pattern in event_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:3]:  # Limit to 3 events
                metadata["events"].append(" ".join(match))
        
        return metadata
    
    # ==========================================================================
    # CHUNKING
    # ==========================================================================
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into semantic chunks for embedding
        Uses smart boundaries (paragraphs, dialogue blocks, scene breaks)
        """
        chunks = []
        
        if not text:
            return chunks
        
        # Detect scene breaks
        scene_break_pattern = r'\n\s*(?:\*\s*\*\s*\*|---+|___+|\#\#\#)\s*\n'
        
        # First split by scene breaks if present
        scenes = re.split(scene_break_pattern, text)
        
        for scene in scenes:
            if not scene.strip():
                continue
            
            # Then split by paragraphs
            paragraphs = scene.split('\n\n')
            
            current_chunk = ""
            current_start = 0
            char_position = 0
            
            for para in paragraphs:
                para_with_break = para.strip() + "\n\n"
                
                if not para_with_break.strip():
                    continue
                
                if len(current_chunk) + len(para_with_break) <= self.chunk_size:
                    current_chunk += para_with_break
                else:
                    # Save current chunk
                    if current_chunk.strip():
                        chunks.append({
                            "text": current_chunk.strip(),
                            "start": current_start,
                            "end": char_position
                        })
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > self.chunk_overlap:
                        overlap_text = current_chunk[-self.chunk_overlap:]
                    else:
                        overlap_text = current_chunk
                    
                    current_chunk = overlap_text + para_with_break
                    current_start = char_position - len(overlap_text)
                
                char_position += len(para_with_break)
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "start": current_start,
                    "end": char_position
                })
        
        return chunks
    
    # ==========================================================================
    # EMBEDDING GENERATION (OLLAMA)
    # ==========================================================================
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Ollama's embedding API
        Uses nomic-embed-text model (768 dimensions)
        """
        if not texts:
            return []
        
        embeddings = []
        
        for text in texts:
            try:
                # Clean text
                clean_text = text.strip()
                if not clean_text:
                    embeddings.append([0.0] * self.embedding_dimension)
                    continue
                
                # Call Ollama embedding endpoint
                response = await self.http_client.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": clean_text
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    
                    if embedding:
                        embeddings.append(embedding)
                    else:
                        logger.warning(f"Empty embedding returned for text: {clean_text[:50]}...")
                        embeddings.append([0.0] * self.embedding_dimension)
                else:
                    logger.error(f"Ollama embedding failed: {response.status_code} - {response.text}")
                    embeddings.append([0.0] * self.embedding_dimension)
                    
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                embeddings.append([0.0] * self.embedding_dimension)
        
        return embeddings
    
    # ==========================================================================
    # STORAGE HELPERS
    # ==========================================================================
    
    async def _store_in_chroma(
        self,
        collection_name: str,
        chunks: List[Dict],
        embeddings: List[List[float]],
        metadata: List[Dict]
    ) -> None:
        """Store embeddings in ChromaDB for fast retrieval"""
        if not self.chroma_client:
            return
        
        try:
            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(
                collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            # Generate unique IDs
            ids = [
                hashlib.md5(f"{chunk.get('text', '')}_{i}".encode()).hexdigest()[:16]
                for i, chunk in enumerate(chunks)
            ]
            
            # Add to collection (upsert)
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=[c.get("text", "") for c in chunks],
                metadatas=metadata
            )
            
            logger.debug(f"Stored {len(chunks)} chunks in ChromaDB collection: {collection_name}")
            
        except Exception as e:
            logger.error(f"ChromaDB storage failed: {e}")
    
    async def _clear_chapter_embeddings(self, db: AsyncSession, chapter_id: str) -> None:
        """Clear existing embeddings for a chapter"""
        await db.execute(
            delete(StoryEmbedding).where(StoryEmbedding.chapter_id == chapter_id)
        )
    
    async def _clear_character_embeddings(self, db: AsyncSession, character_id: str) -> None:
        """Clear existing embeddings for a character"""
        await db.execute(
            delete(CharacterEmbedding).where(CharacterEmbedding.character_id == character_id)
        )
    
    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================
    
    async def update_embeddings_for_edit(
        self,
        db: AsyncSession,
        story_id: str,
        chapter_id: str,
        old_content: str,
        new_content: str
    ) -> None:
        """Update embeddings when content is edited"""
        await self.embed_chapter(
            db=db,
            story_id=story_id,
            chapter_id=chapter_id,
            content=new_content,
            chapter_metadata={}
        )
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        a_np = np.array(a)
        b_np = np.array(b)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / (norm_a * norm_b))
