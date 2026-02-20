"""
AI-powered story extraction service.
Analyzes imported text to extract characters, plotlines, themes, and metadata.
"""

import re
import json
from typing import Dict, List, Optional
import httpx
from app.config import settings
from app.runtime_settings import get_runtime_model_name


class StoryExtractor:
    """Extract story elements from imported text using AI."""
    
    def __init__(self, ollama_url: str = settings.ollama_base_url):
        self.ollama_url = ollama_url
    
    async def extract_story_elements(self, title: str, content: str, chapters: List[Dict], max_tokens: Optional[int] = None) -> Dict:
        """
        Extract all story elements from imported content.
        
        Returns:
            {
                'metadata': {...},
                'characters': [...],
                'plotlines': [...],
                'themes': [...]
            }
        """
        # Extract metadata first
        metadata = await self._extract_metadata(title, content, chapters, max_tokens)
        
        # Extract characters
        characters = await self._extract_characters(content, chapters, max_tokens)
        
        # Extract plotlines
        plotlines = await self._extract_plotlines(content, chapters, max_tokens)
        
        # Extract themes
        themes = await self._extract_themes(content, max_tokens)
        
        return {
            'metadata': metadata,
            'characters': characters,
            'plotlines': plotlines,
            'themes': themes
        }
    
    async def _call_ollama(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Call Ollama API for text generation."""
        max_predict = max_tokens or settings.max_tokens_import_story
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": get_runtime_model_name(),
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower for more consistent extraction
                        "num_predict": max_predict
                    }
                }
            )
            result = response.json()
            return result.get('response', '')
    
    async def _extract_metadata(self, title: str, content: str, chapters: List[Dict], max_tokens: Optional[int] = None) -> Dict:
        """Extract story metadata (genre, tone, logline, etc.)."""
        
        # Use first few chapters for analysis
        sample_text = content[:5000]
        
        prompt = f"""Analyze this story excerpt and extract metadata in JSON format.

Story Title: {title}
Excerpt:
{sample_text}

Extract the following and respond ONLY with valid JSON (no other text):
{{
    "genre": "primary genre (e.g., Fantasy, Science Fiction, Mystery, Romance, Thriller)",
    "subgenres": ["list", "of", "subgenres"],
    "tone": "overall tone (e.g., Dark, Humorous, Serious, Whimsical)",
    "setting": "primary setting description",
    "time_period": "time period (e.g., Modern Day, Medieval, Future)",
    "target_audience": "target audience (e.g., Young Adult, Adult, Middle Grade)",
    "logline": "one-sentence story summary"
}}

JSON:"""
        
        response = await self._call_ollama(prompt, max_tokens)
        
        # Try to parse JSON from response
        try:
            # Extract JSON from response (might have extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                metadata = json.loads(json_match.group())
                return metadata
        except:
            pass
        
        # Fallback if parsing fails
        return {
            'genre': 'Fiction',
            'subgenres': [],
            'tone': 'Serious',
            'setting': 'Unknown',
            'time_period': 'Unknown',
            'target_audience': 'Adult',
            'logline': f'A story titled "{title}"'
        }
    
    async def _extract_characters(self, content: str, chapters: List[Dict], max_tokens: Optional[int] = None) -> List[Dict]:
        """Extract main characters from the story."""
        
        # Use first few chapters for character analysis
        sample_text = content[:10000]
        
        prompt = f"""Analyze this story excerpt and identify the main characters (up to 8 most important).

Excerpt:
{sample_text}

For each character, extract:
- Name
- Role (protagonist, antagonist, supporting)
- Brief description (age, appearance, key traits)
- Character arc (their journey/development)

Respond ONLY with valid JSON array (no other text):
[
    {{
        "name": "Character Name",
        "role": "protagonist/antagonist/supporting",
        "description": "Physical appearance and personality traits",
        "arc": "Their character development throughout the story"
    }}
]

JSON:"""
        
        response = await self._call_ollama(prompt, max_tokens)
        
        # Try to parse JSON
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                characters = json.loads(json_match.group())
                return characters
        except:
            pass
        
        return []
    
    async def _extract_plotlines(self, content: str, chapters: List[Dict], max_tokens: Optional[int] = None) -> List[Dict]:
        """Extract main plotlines/story threads."""
        
        # Build chapter summaries for analysis
        chapter_info = []
        for i, ch in enumerate(chapters[:10]):  # First 10 chapters
            preview = ch['content'][:500]
            chapter_info.append(f"Chapter {i+1}: {preview}...")
        
        chapters_text = '\n\n'.join(chapter_info)
        
        prompt = f"""Analyze this story and identify the main plotlines/story threads (up to 5).

Story Chapters:
{chapters_text}

For each plotline, extract:
- Title/name of the plotline
- Type (main_plot, subplot, romance, mystery, etc.)
- Description (what happens)
- Status (unresolved, resolved, ongoing)

Respond ONLY with valid JSON array (no other text):
[
    {{
        "title": "Plotline Title",
        "type": "main_plot/subplot/romance/mystery/character_arc",
        "description": "What happens in this plotline",
        "status": "unresolved/resolved/ongoing"
    }}
]

JSON:"""
        
        response = await self._call_ollama(prompt, max_tokens)
        
        # Try to parse JSON
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                plotlines = json.loads(json_match.group())
                return plotlines
        except:
            pass
        
        return []
    
    async def _extract_themes(self, content: str, max_tokens: Optional[int] = None) -> List[str]:
        """Extract main themes from the story."""
        
        sample_text = content[:8000]
        
        prompt = f"""Analyze this story excerpt and identify the main themes (up to 6).

Excerpt:
{sample_text}

List the major themes explored in this story (e.g., redemption, love vs duty, power and corruption, identity, family, justice).

Respond ONLY with a JSON array of theme strings (no other text):
["theme1", "theme2", "theme3"]

JSON:"""
        
        response = await self._call_ollama(prompt, max_tokens)
        
        # Try to parse JSON
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                themes = json.loads(json_match.group())
                return themes
        except:
            pass
        
        return []
