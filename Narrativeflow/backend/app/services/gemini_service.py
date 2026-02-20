"""
AI Service - Core AI integration for NarrativeFlow
Handles all communication with Ollama API
"""
import httpx
from typing import Optional, List, Dict, Any, AsyncGenerator
import asyncio
import logging
import time
import json
import re

from app.config import settings
from app.runtime_settings import get_runtime_model_name, get_runtime_vision_model_name
from app.models.generation import WritingMode, GenerationType

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Service for interacting with Ollama API (renamed for backwards compatibility)
    Handles story generation, rewriting, analysis, and more
    """
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model_name = settings.ollama_model
        self.vision_model_name = settings.ollama_model
        self.client = httpx.AsyncClient(timeout=600.0)  # 10 minute timeout for large generations
        
        # Generation settings by mode
        self.mode_settings = {
            WritingMode.AI_LEAD: {
                "temperature": settings.temperature_creative,
                "top_p": 0.95,
                "top_k": 40,
            },
            WritingMode.USER_LEAD: {
                "temperature": settings.temperature_precise,
                "top_p": 0.8,
                "top_k": 20,
            },
            WritingMode.CO_AUTHOR: {
                "temperature": settings.temperature_balanced,
                "top_p": 0.9,
                "top_k": 30,
            }
        }
        logger.info(f"Ollama API configured with model: {self.model_name} at {self.base_url}")
    
    def _get_generation_options(
        self,
        writing_mode: WritingMode,
        max_tokens: Optional[int] = None,
        temperature_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get generation options based on writing mode with pronounced differences"""
        mode_config = self.mode_settings.get(writing_mode, self.mode_settings[WritingMode.CO_AUTHOR])
        
        # Make temperature differences more pronounced
        if temperature_override is None:
            if writing_mode == WritingMode.AI_LEAD:
                temperature = 0.95  # Very creative
            elif writing_mode == WritingMode.USER_LEAD:
                temperature = 0.3   # Very conservative
            else:
                temperature = 0.7   # Balanced
        else:
            temperature = temperature_override
        
        return {
            "temperature": temperature,
            "top_p": mode_config["top_p"],
            "top_k": mode_config["top_k"],
            "num_predict": max_tokens or settings.max_tokens_per_generation,
        }
    
    async def generate_story_content(
        self,
        prompt: str,
        system_prompt: str,
        writing_mode: WritingMode,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature_override: Optional[float] = None,
        request_timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate story content based on prompt and mode
        
        Returns:
            Dict with 'content', 'tokens_used', 'generation_time_ms'
        """
        start_time = time.time()
        
        # Build full prompt with system instructions and context
        full_prompt = self._build_full_prompt(system_prompt, context, prompt)
        
        # Get generation options for mode
        generation_options = self._get_generation_options(writing_mode, max_tokens, temperature_override)
        
        try:
            # Call Ollama API
            post_kwargs = {
                "json": {
                    "model": get_runtime_model_name(),
                    "prompt": full_prompt,
                    "stream": False,
                    "options": generation_options
                }
            }
            if request_timeout is not None:
                post_kwargs["timeout"] = request_timeout
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                **post_kwargs
            )
            response.raise_for_status()
            result = response.json()
            
            generation_time = int((time.time() - start_time) * 1000)
            
            return {
                "content": result.get("response", ""),
                "tokens_used": result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
                "generation_time_ms": generation_time,
                "model": get_runtime_model_name(),
                "success": True
            }
            
        except Exception as e:
            # Include type name so callers can detect timeout vs other errors
            err_str = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error(f"Generation error: {err_str}")
            return {
                "content": "",
                "error": err_str,
                "success": False,
                "generation_time_ms": int((time.time() - start_time) * 1000)
            }

    async def generate_story_content_routed(
        self,
        user_config: Dict[str, Any],
        prompt: str,
        system_prompt: str,
        writing_mode: WritingMode,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature_override: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch generation to Ollama or an external provider based on user_config.
        user_config shape: {"provider": str, "api_key": str|None, "model": str}
        Falls back to Ollama if provider is 'ollama' or unknown.
        """
        provider = user_config.get("provider", "ollama")

        if provider == "ollama":
            return await self.generate_story_content(
                prompt=prompt,
                system_prompt=system_prompt,
                writing_mode=writing_mode,
                context=context,
                max_tokens=max_tokens,
                temperature_override=temperature_override,
                request_timeout=request_timeout,
            )

        # External provider
        from app.services.external_ai_service import generate_external
        full_prompt = self._build_full_prompt(system_prompt, context, prompt)
        generation_options = self._get_generation_options(writing_mode, max_tokens, temperature_override)
        temperature = generation_options.get("temperature", 0.7)
        tokens = max_tokens or generation_options.get("num_predict", 800)
        timeout = request_timeout or 120.0

        return await generate_external(
            provider=provider,
            api_key=user_config.get("api_key", ""),
            model=user_config.get("model", ""),
            prompt=full_prompt,
            system_prompt="",   # already embedded in full_prompt
            max_tokens=tokens,
            temperature=temperature,
            request_timeout=timeout,
        )

    async def generate_story_content_stream(
        self,
        prompt: str,
        system_prompt: str,
        writing_mode: WritingMode,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate story content with streaming response
        Yields chunks of text as they are generated
        """
        # Build full prompt with system instructions and context
        full_prompt = self._build_full_prompt(system_prompt, context, prompt)
        
        # Get generation options for mode
        generation_options = self._get_generation_options(writing_mode, max_tokens)
        
        try:
            # Call Ollama API with streaming
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": get_runtime_model_name(),
                    "prompt": full_prompt,
                    "stream": True,
                    "options": generation_options
                }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"\n[Error: {str(e)}]"
    
    async def generate_continuation_stream(
        self,
        story_context: str,
        recent_text: str,
        writing_mode: WritingMode,
        style_guide: Optional[str] = None,
        word_count_target: int = 500
    ) -> AsyncGenerator[str, None]:
        """Generate a story continuation with streaming"""
        system_prompt = self._get_continuation_system_prompt(writing_mode, style_guide)
        
        prompt = f"""Continue the following story naturally. Write approximately {word_count_target} words.

STORY CONTEXT:
{story_context}

RECENT TEXT (continue from here):
{recent_text}

Continue the story:"""
        
        async for chunk in self.generate_story_content_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=writing_mode,
            max_tokens=min(int(word_count_target * 1.3), settings.max_tokens_story_generation)
        ):
            yield chunk
    
    async def generate_continuation(
        self,
        story_context: str,
        recent_text: str,
        writing_mode: WritingMode,
        style_guide: Optional[str] = None,
        word_count_target: int = 500
    ) -> Dict[str, Any]:
        """Generate a story continuation"""
        system_prompt = self._get_continuation_system_prompt(writing_mode, style_guide)
        
        prompt = f"""Continue the following story naturally. Write approximately {word_count_target} words.

STORY CONTEXT:
{story_context}

RECENT TEXT (continue from here):
{recent_text}

Continue the story:"""
        
        return await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=writing_mode,
            max_tokens=min(int(word_count_target * 1.3), settings.max_tokens_story_generation)
        )
    
    async def rewrite_text(
        self,
        original_text: str,
        instructions: str,
        writing_mode: WritingMode,
        style_guide: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rewrite text based on instructions"""
        system_prompt = f"""You are an expert editor and rewriter. Your task is to improve text while maintaining 
        the author's voice and intent. {style_guide or ''}"""
        
        prompt = f"""Rewrite the following text according to these instructions:

INSTRUCTIONS: {instructions}

ORIGINAL TEXT:
{original_text}

REWRITTEN VERSION:"""
        
        return await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=writing_mode
        )
    
    async def generate_summary(
        self,
        content: str,
        summary_type: str = "chapter",  # chapter, story, character
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate a summary of content"""
        type_instructions = {
            "chapter": "Summarize this chapter, highlighting key events, character developments, and plot progressions.",
            "story": "Provide a comprehensive summary of this story so far, including main plot points, character arcs, and themes.",
            "character": "Summarize this character's journey, development, and current state."
        }
        
        system_prompt = """You are an expert at analyzing and summarizing narrative content. 
        Provide clear, concise summaries that capture the essential elements."""
        
        prompt = f"""{type_instructions.get(summary_type, type_instructions["chapter"])}

CONTENT TO SUMMARIZE:
{content}

SUMMARY:"""
        
        return await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=WritingMode.USER_LEAD,  # Use precise mode for summaries
            max_tokens=max_tokens or settings.max_tokens_summary
        )
    
    async def generate_story_recap(
        self,
        story_summary: str,
        chapters_summary: str,
        characters_state: str,
        plotlines_state: str
    ) -> Dict[str, Any]:
        """Generate a comprehensive story recap"""
        system_prompt = """You are an expert narrative analyst. Provide a clear, organized recap 
        of the story that helps the writer understand where things stand."""
        
        prompt = f"""Generate a comprehensive recap of this story covering:
1. What has happened (major events)
2. Current character states and locations
3. Unresolved plot threads
4. Key themes and motifs

STORY OVERVIEW:
{story_summary}

CHAPTERS SUMMARY:
{chapters_summary}

CHARACTER STATES:
{characters_state}

ACTIVE PLOTLINES:
{plotlines_state}

STORY RECAP:"""
        
        return await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=WritingMode.USER_LEAD,
            max_tokens=settings.max_tokens_recap
        )
    
    async def analyze_consistency(
        self,
        content: str,
        characters: str,
        world_rules: str,
        previous_events: str
    ) -> Dict[str, Any]:
        """Analyze content for consistency issues"""
        system_prompt = """You are an expert continuity editor. Identify any inconsistencies, 
        contradictions, or violations of established rules in the narrative."""
        
        prompt = f"""Analyze this content for consistency issues:

CONTENT TO ANALYZE:
{content}

ESTABLISHED CHARACTERS:
{characters}

WORLD RULES:
{world_rules}

PREVIOUS EVENTS:
{previous_events}

Identify any issues with:
1. Character behavior inconsistencies
2. Timeline contradictions
3. World rule violations
4. POV consistency
5. Tone drift

CONSISTENCY ANALYSIS:"""
        
        return await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=WritingMode.USER_LEAD,
            max_tokens=700  # Optimized for speed
        )
    
    async def generate_character_dialogue(
        self,
        character_profile: str,
        scene_context: str,
        dialogue_prompt: str,
        writing_mode: WritingMode
    ) -> Dict[str, Any]:
        """Generate dialogue for a specific character"""
        system_prompt = f"""You are an expert dialogue writer. Write dialogue that perfectly matches 
        the character's voice, background, and personality.
        
CHARACTER PROFILE:
{character_profile}"""
        
        prompt = f"""Write dialogue for this character in the following scene:

SCENE CONTEXT:
{scene_context}

DIALOGUE SITUATION:
{dialogue_prompt}

Write the character's dialogue (and brief action beats if needed):"""
        
        return await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=writing_mode
        )
    
    async def generate_image_prompt(
        self,
        description: str,
        image_type: str,  # character, scene, cover
        style: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured image generation prompt"""
        system_prompt = """You are an expert at creating detailed image generation prompts. 
        Create prompts that are vivid, specific, and will produce high-quality images."""
        
        style_guide = style or "cinematic, detailed, high quality, professional"
        
        prompt = f"""Create a detailed image generation prompt for the following:

TYPE: {image_type}
DESCRIPTION: {description}
STYLE: {style_guide}

Create a prompt that includes:
1. Main subject description
2. Composition and framing
3. Lighting and atmosphere
4. Style and artistic direction
5. Technical quality specifications

IMAGE PROMPT:"""
        
        return await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=WritingMode.USER_LEAD,
            max_tokens=settings.max_tokens_story_to_image_prompt
        )
    
    async def analyze_image_for_story(
        self,
        image_base64: str,
        prompt: str,
        system_prompt: str,
        writing_mode: WritingMode,
        language: str = "English",
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using a vision model and generate story content.
        Uses Ollama's vision capabilities (llava or similar model).
        Supports multiple languages.
        """
        import base64
        import time
        
        start_time = time.time()
        
        # Build full prompt
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Get generation options
        generation_options = self._get_generation_options(
            writing_mode,
            max_tokens=max_tokens or settings.max_tokens_image_to_story
        )
        
        try:
            # Check if image data is valid
            # Remove data URL prefix if present
            if image_base64.startswith('data:'):
                # Extract base64 part from data URL
                image_base64 = image_base64.split(',', 1)[1]
            
            # Call Ollama API with image
            # Note: Requires a vision-capable model like llava, bakllava, or moondream
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": get_runtime_vision_model_name(),
                    "prompt": full_prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": generation_options
                }
            )
            response.raise_for_status()
            result = response.json()
            
            generation_time = int((time.time() - start_time) * 1000)
            
            return {
                "content": result.get("response", ""),
                "image_description": "Image analyzed successfully",
                "tokens_used": result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
                "generation_time_ms": generation_time,
                "model": get_runtime_vision_model_name(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            # Fallback: If vision model fails, generate based on prompt alone
            logger.info("Falling back to text-only generation")
            return {
                "content": "",
                "error": f"Vision analysis failed: {str(e)}. Make sure you have a vision-capable model (like llava) installed in Ollama.",
                "success": False,
                "generation_time_ms": int((time.time() - start_time) * 1000)
            }
    
    async def brainstorm_ideas(
        self,
        context: str,
        brainstorm_type: str,  # plot, character, scene, dialogue
        constraints: Optional[str] = None
    ) -> Dict[str, Any]:
        """Brainstorm creative ideas"""
        system_prompt = """You are a creative writing partner. Generate multiple diverse, 
        interesting ideas that could take the story in exciting directions."""
        
        prompt = f"""Brainstorm {brainstorm_type} ideas for this story:

CONTEXT:
{context}

{f"CONSTRAINTS: {constraints}" if constraints else ""}

Generate 5 different creative options, ranging from safe to bold:"""
        
        return await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=WritingMode.AI_LEAD,  # Use creative mode for brainstorming
            max_tokens=settings.max_tokens_brainstorm
        )
    
    async def stream_generation(
        self,
        prompt: str,
        system_prompt: str,
        writing_mode: WritingMode
    ) -> AsyncGenerator[str, None]:
        """Stream generated content chunk by chunk"""
        full_prompt = self._build_full_prompt(system_prompt, None, prompt)
        generation_options = self._get_generation_options(writing_mode)
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json={
                "model": get_runtime_model_name(),
                "prompt": full_prompt,
                "stream": True,
                "options": generation_options
            },
            timeout=300.0
        ) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk and chunk["response"]:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        continue
    
    def _build_full_prompt(
        self,
        system_prompt: str,
        context: Optional[str],
        user_prompt: str
    ) -> str:
        """Build the full prompt with all components"""
        parts = [system_prompt]
        
        if context:
            parts.append(f"\n\nCONTEXT:\n{context}")
        
        parts.append(f"\n\n{user_prompt}")
        
        return "\n".join(parts)
    
    def _get_continuation_system_prompt(
        self,
        writing_mode: WritingMode,
        style_guide: Optional[str] = None
    ) -> str:
        """Get system prompt for story continuation based on mode"""
        mode_instructions = {
            WritingMode.AI_LEAD: """You are an autonomous creative writer. Take bold creative decisions, 
            introduce compelling developments, and write with confidence. The user trusts your creative vision.""",
            
            WritingMode.USER_LEAD: """You are a supportive writing assistant. Continue the story following 
            the established direction closely. Don't introduce major new elements unless essential. 
            Match the user's style exactly.""",
            
            WritingMode.CO_AUTHOR: """You are a collaborative co-author. Continue the story thoughtfully, 
            building on what's established while adding your creative input. Balance respecting the user's 
            vision with contributing fresh ideas."""
        }
        
        base_prompt = mode_instructions.get(writing_mode, mode_instructions[WritingMode.CO_AUTHOR])
        
        if style_guide:
            base_prompt += f"\n\nSTYLE GUIDE:\n{style_guide}"
        
        return base_prompt
    
    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: ~4 characters per token
        total_chars = len(prompt) + len(response)
        return total_chars // 4

    def _parse_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON extraction from model output, including truncated JSON recovery."""
        cleaned = text.strip()

        def try_load(candidate: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        def fix_and_load(candidate: str) -> Optional[Dict[str, Any]]:
            # Fix trailing commas before ] or }
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            result = try_load(fixed)
            if result is not None:
                return result
            # Try to repair truncated JSON by closing open braces/brackets
            return self._recover_truncated_json(fixed)

        # 1. Try direct parse
        direct = try_load(cleaned)
        if direct is not None:
            return direct

        # 2. Strip markdown fences that weren't caught upstream
        stripped = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        direct2 = try_load(stripped)
        if direct2 is not None:
            return direct2

        # 3. Extract first JSON object
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        return fix_and_load(match.group(0))

    def _recover_truncated_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to close open braces/brackets in truncated JSON and parse it."""
        # Remove trailing incomplete string/value by truncating at last complete value
        # Strategy: find the deepest position where we can cleanly close
        stack = []
        in_string = False
        escape_next = False
        last_safe_pos = 0

        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                stack.append(ch)
            elif ch in ('}', ']'):
                if stack:
                    stack.pop()
                if not stack:
                    last_safe_pos = i + 1  # complete top-level object

        if last_safe_pos > 0:
            # The JSON was complete up to last_safe_pos
            candidate = text[:last_safe_pos]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        if not stack:
            return None  # nothing to close

        # Truncate at last comma or colon at top level to remove partial value
        truncated = text.rstrip()
        # Remove trailing incomplete tokens (partial strings, dangling comma, etc.)
        truncated = re.sub(r',\s*$', '', truncated)  # trailing comma
        truncated = re.sub(r'"[^"]*$', '', truncated)  # unclosed string
        truncated = re.sub(r',\s*$', '', truncated)  # another trailing comma
        truncated = re.sub(r':\s*$', '', truncated)  # dangling colon
        truncated = truncated.rstrip()

        # Close remaining open brackets/braces in reverse order
        closers = {'[': ']', '{': '}'}
        closing = ''.join(closers[ch] for ch in reversed(stack))
        candidate = truncated + closing
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    
    async def generate_story_bible_simple(
        self,
        story_content: str,
        story_title: str,
        story_genre: str,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Simplified Story Bible generation with a minimal JSON schema.
        Designed for CPU-based Ollama inference (~9 tok/s); keeps prompts tiny.
        """
        # Cap content at 300 chars so total prompt stays under ~120 tokens.
        # This keeps inference time under ~30s even on pure CPU.
        snippet = story_content[:300]
        prompt = f"""Story: {snippet}
Return ONLY this JSON (no other text):
{{"world_description":"...","world_type":"{story_genre}","time_period":"...","central_themes":["..."],"quick_facts":["..."],"primary_locations":[{{"name":"...","description":"..."}}]}}"""

        result = await self.generate_story_content(
            prompt=prompt,
            system_prompt="Return ONLY valid JSON. No markdown, no explanation, no extra text.",
            writing_mode=WritingMode.USER_LEAD,
            max_tokens=min(max_tokens or 350, 350),  # 350 tokens to avoid mid-JSON truncation
            request_timeout=120.0,  # 2 min: covers ~(120-prompt-tokens + 350 output) / 9 tok/s
        )
        if result.get("success"):
            bible_data = self._parse_json_from_text(result.get("content", ""))
            if bible_data is not None:
                result["bible_data"] = bible_data
                result["parsed"] = True
            else:
                result["parsed"] = False
        return result

    async def generate_story_bible(
        self,
        story_content: str,
        story_title: str,
        story_genre: str,
        story_tone: str,
        existing_characters: Optional[str] = None,
        language: str = "English",
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Auto-generate Story Bible by analyzing story content.
        Extracts world rules, locations, terminology, themes, and more.
        Works with multiple languages.
        
        Args:
            story_content: The story text to analyze
            story_title: Title of the story
            story_genre: Genre of the story
            story_tone: Tone of the story
            existing_characters: String of known character names
            language: Language of the story content
            
        Returns:
            Dict with bible data and metadata
        """
        # Language-specific instruction
        if language != "English":
            language_note = f"\n\nNOTE: This story is written in {language}. Analyze the content in that language. Preserve all terms, names, and locations as they appear in the original {language} text. Your JSON response should be in English format, but preserve the {language} names and terms in the data fields."
        else:
            language_note = ""
            
        system_prompt = f"""You are an expert story analyst and world-building specialist working with stories in multiple languages.
Your task is to analyze story content and extract/infer all world-building elements.{language_note}

You MUST respond with valid JSON only. No other text before or after the JSON.
Be thorough - extract everything that's explicitly mentioned AND reasonably inferred."""

        prompt = f"""Analyze this {story_genre} story ("{story_title}") written in {language} and generate a comprehensive Story Bible.

STORY CONTENT:
{story_content}

{f"KNOWN CHARACTERS: {existing_characters}" if existing_characters else ""}

Extract and generate the following as a JSON object:

{{
    "world_name": "Name of the world/setting if mentioned or can be inferred, or null",
    "world_description": "Brief description of the world/setting",
    "world_type": "Type of world (e.g., 'fantasy medieval', 'cyberpunk', 'contemporary', 'space opera')",
    "time_period": "When the story takes place",
    "primary_locations": [
        {{"name": "Location name", "description": "Description", "importance": "high/medium/low"}}
    ],
    "magic_system": "Description of magic/supernatural system if any, or null",
    "magic_rules": ["Rule 1", "Rule 2"],
    "magic_limitations": ["Limitation 1", "Limitation 2"],
    "technology_level": "Technology level description",
    "societies": [
        {{"name": "Society/culture name", "description": "Description", "customs": ["custom1"]}}
    ],
    "world_rules": [
        {{"category": "physics/magic/society/technology/biology", "title": "Rule title", "description": "Rule description", "importance": 1-10}}
    ],
    "central_themes": ["Theme 1", "Theme 2"],
    "recurring_motifs": ["Motif 1", "Motif 2"],
    "glossary": [
        {{"term": "Term", "definition": "Definition"}}
    ],
    "tone_guidelines": "Specific tone/style notes for this story",
    "quick_facts": ["Important fact 1", "Important fact 2"]
}}

Analyze carefully and be comprehensive. For a {story_tone} tone {story_genre} story.
Respond with ONLY the JSON object:"""

        result = await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=WritingMode.USER_LEAD,  # Precise mode for analysis
            max_tokens=max_tokens or settings.max_tokens_story_bible,
            request_timeout=300.0  # 5 min for full prompt on CPU Ollama (~9 tok/s)
        )
        
        if result.get("success"):
            # Try to parse the JSON response
            try:
                content = result["content"].strip()
                # Handle potential markdown code blocks
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                bible_data = self._parse_json_from_text(content)
                if bible_data is None:
                    logger.warning(f"Story Bible JSON parse failed. Raw content ({len(content)} chars): {content[:500]}")
                    raise json.JSONDecodeError("Failed to parse JSON", content, 0)
                result["bible_data"] = bible_data
                result["parsed"] = True
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Story Bible JSON: {e}")
                result["parsed"] = False
                result["parse_error"] = str(e)
        
        return result
    
    async def update_story_bible_from_content(
        self,
        new_content: str,
        existing_bible: Dict[str, Any],
        story_genre: str,
        language: str = "English",
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update Story Bible incrementally based on new content.
        Adds new elements without removing existing ones.
        Works with multiple languages.
        
        Args:
            new_content: New story content to analyze
            existing_bible: Current bible data
            story_genre: Genre of the story
            language: Language of the story content
        """
        # Language-specific instruction
        if language != "English":
            language_note = f" The content is in {language} - preserve original names and terms."
        else:
            language_note = ""
            
        system_prompt = f"""You are a story analyst updating a Story Bible with new information.{language_note}
Identify NEW elements that should be added based on the new content.
Only return NEW items, not existing ones.
Respond with valid JSON only."""

        prompt = f"""Analyze this new story content (in {language}) and identify NEW Story Bible elements to add.

NEW CONTENT:
{new_content}

EXISTING STORY BIBLE:
{json.dumps(existing_bible, indent=2)}

Return ONLY NEW elements to add (not duplicates of existing):

{{
    "new_locations": [{{"name": "", "description": "", "importance": ""}}],
    "new_world_rules": [{{"category": "", "title": "", "description": "", "importance": 5}}],
    "new_glossary_terms": [{{"term": "", "definition": ""}}],
    "new_themes": [],
    "new_quick_facts": []
}}

Only include sections that have new items. Respond with JSON only:"""

        result = await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=WritingMode.USER_LEAD,
            max_tokens=max_tokens or settings.max_tokens_story_bible_update
        )
        
        if result.get("success"):
            try:
                content = result["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                updates = json.loads(content)
                result["updates"] = updates
                result["parsed"] = True
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Story Bible update JSON: {e}")
                result["parsed"] = False
        
        return result

    async def extract_characters_from_content(
        self,
        story_content: str,
        story_title: str,
        story_genre: str,
        existing_character_names: List[str] = None,
        language: str = "English",
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract characters from story content using AI analysis.
        Works with multiple languages.
        
        Args:
            story_content: The story text to analyze
            story_title: Title of the story
            story_genre: Genre of the story
            existing_character_names: List of already extracted character names
            language: Language of the story content
            
        Returns:
            Dict with extracted characters and metadata
        """
        existing_names = existing_character_names or []
        existing_str = ", ".join(existing_names) if existing_names else "None"
        
        # Language-specific instruction
        if language != "English":
            language_note = f"\nNOTE: This story is written in {language}. Analyze the content in that language and provide character names as they appear in the original text. Your response should still be in English JSON format, but preserve character names and descriptions from the {language} text."
        else:
            language_note = ""
        
        system_prompt = f"""You are an expert story analyst specializing in character identification across multiple languages.
Your task is to extract ALL characters from story content, including their traits and relationships.
Be thorough - identify main characters, supporting characters, and even minor characters mentioned.{language_note}

You MUST respond with valid JSON only. No other text before or after the JSON."""

        prompt = f"""Analyze this {story_genre} story ("{story_title}") written in {language} and extract ALL characters mentioned.

STORY CONTENT:
{story_content}

ALREADY EXISTING CHARACTERS (skip these): {existing_str}

For each NEW character found, extract:
- Their name (as appears in the story)
- Their role (protagonist, antagonist, supporting, minor, mentor, love_interest)
- Physical description (if mentioned)
- Personality traits observed from their actions/dialogue
- Their backstory (if mentioned or implied)
- Their goals/motivation (if apparent)
- Speaking style (how they talk)
- Relationships with other characters
- Any distinguishing features

Return as a JSON object:

{{
    "characters": [
        {{
            "name": "Character name",
            "full_name": "Full name if mentioned, or null",
            "role": "protagonist/antagonist/supporting/minor/mentor/love_interest",
            "age": "Age or age range if mentioned, or null",
            "gender": "Gender if mentioned or apparent, or null",
            "species": "human unless otherwise specified",
            "occupation": "Job/role if mentioned, or null",
            "physical_description": "Physical description from the text",
            "personality_summary": "Brief personality summary",
            "personality_traits": ["trait1", "trait2", "trait3"],
            "backstory": "Backstory if mentioned or implied, or null",
            "motivation": "Goals and motivations",
            "speaking_style": "How they speak (formal, casual, accent, etc.)",
            "relationships": "Relationships with other characters mentioned",
            "distinguishing_features": ["Notable feature 1", "Notable feature 2"]
        }}
    ],
    "total_found": 0
}}

Be thorough - include ALL characters, even those briefly mentioned.
Skip any characters already in the existing list.
Respond with ONLY the JSON object:"""

        result = await self.generate_story_content(
            prompt=prompt,
            system_prompt=system_prompt,
            writing_mode=WritingMode.USER_LEAD,  # Precise mode for analysis
            max_tokens=max_tokens or settings.max_tokens_character_extraction
        )
        
        if result.get("success"):
            try:
                content = result["content"].strip()
                # Handle potential markdown code blocks
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                character_data = json.loads(content)
                result["characters"] = character_data.get("characters", [])
                result["total_found"] = len(result["characters"])
                result["parsed"] = True
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse character extraction JSON: {e}")
                result["parsed"] = False
                result["parse_error"] = str(e)
        
        return result
