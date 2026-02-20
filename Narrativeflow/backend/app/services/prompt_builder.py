"""
Prompt Builder Service - Constructs optimized prompts for AI generation
Manages context injection, character data, and writing mode adaptations
"""
from typing import Optional, List, Dict, Any
from app.models.generation import WritingMode
from app.models.story import Story, StoryGenre, StoryTone
from app.models.chapter import Chapter
from app.models.character import Character, CharacterRole
from app.models.plotline import Plotline
from app.models.story_bible import StoryBible


class PromptBuilder:
    """
    Builds optimized prompts for different AI operations
    Handles context management and token optimization
    """
    
    # Token budget allocation (approximate)
    MAX_CONTEXT_TOKENS = 8000
    SYSTEM_PROMPT_BUDGET = 500
    CHARACTER_BUDGET = 1500
    PLOT_BUDGET = 1000
    WORLD_RULES_BUDGET = 800
    RECENT_CONTENT_BUDGET = 2000
    RETRIEVED_CONTEXT_BUDGET = 2000
    
    def __init__(self):
        self.genre_styles = self._load_genre_styles()
        self.tone_modifiers = self._load_tone_modifiers()
    
    def build_continuation_prompt(
        self,
        story: Story,
        chapter: Chapter,
        characters: List[Character],
        active_plotlines: List[Plotline],
        story_bible: Optional[StoryBible],
        recent_content: str,
        retrieved_context: List[str],
        writing_mode: WritingMode,
        user_direction: Optional[str] = None,
        word_target: int = 500
    ) -> Dict[str, str]:
        """
        Build a complete prompt for story continuation
        
        Returns dict with 'system_prompt', 'context', 'user_prompt'
        """
        # Build system prompt based on mode and story settings
        system_prompt = self._build_system_prompt(story, writing_mode)
        
        # Build context sections
        context_parts = []
        
        # Add story overview
        context_parts.append(self._build_story_overview(story))
        
        # Add relevant characters
        context_parts.append(self._build_character_context(characters, chapter))
        
        # Add active plotlines
        context_parts.append(self._build_plotline_context(active_plotlines))
        
        # Add world rules if available
        if story_bible:
            context_parts.append(self._build_world_rules_context(story_bible))
        
        # Add retrieved semantic context
        if retrieved_context:
            context_parts.append(self._build_retrieved_context(retrieved_context))
        
        context = "\n\n".join(filter(None, context_parts))
        
        # Build user prompt
        user_prompt = self._build_user_prompt_continuation(
            chapter=chapter,
            recent_content=recent_content,
            user_direction=user_direction,
            word_target=word_target,
            writing_mode=writing_mode
        )
        
        return {
            "system_prompt": system_prompt,
            "context": context,
            "user_prompt": user_prompt
        }
    
    def build_rewrite_prompt(
        self,
        story: Story,
        original_text: str,
        instructions: str,
        characters: List[Character],
        writing_mode: WritingMode
    ) -> Dict[str, str]:
        """Build prompt for rewriting content"""
        system_prompt = f"""You are an expert editor working on a {story.genre.value} story with a {story.tone.value} tone.

Your task is to rewrite text while:
1. Maintaining the author's voice and style
2. Preserving character voices and personalities
3. Keeping established facts and continuity
4. Following the specific instructions provided

{self._get_mode_instructions(writing_mode, "rewrite")}"""
        
        character_context = self._build_character_context(characters, None)
        
        user_prompt = f"""REWRITE INSTRUCTIONS:
{instructions}

ORIGINAL TEXT TO REWRITE:
{original_text}

Provide the rewritten version:"""
        
        return {
            "system_prompt": system_prompt,
            "context": character_context,
            "user_prompt": user_prompt
        }
    
    def build_dialogue_prompt(
        self,
        character: Character,
        scene_context: str,
        other_characters: List[Character],
        dialogue_situation: str,
        writing_mode: WritingMode
    ) -> Dict[str, str]:
        """Build prompt for generating character dialogue"""
        system_prompt = f"""You are an expert dialogue writer specializing in character-authentic speech.

Your task is to write dialogue for {character.name} that:
1. Matches their speaking style: {character.speaking_style or 'natural'}
2. Reflects their personality: {character.personality_summary or 'as established'}
3. Uses appropriate vocabulary level: {character.vocabulary_level or 'standard'}
4. Includes characteristic phrases if any: {', '.join(character.catchphrases) if character.catchphrases else 'none specified'}

{self._get_mode_instructions(writing_mode, "dialogue")}"""
        
        # Build character profile
        character_profile = self._build_detailed_character_profile(character)
        
        # Build other characters context
        others_context = ""
        if other_characters:
            others_context = "\n\nOTHER CHARACTERS IN SCENE:\n"
            for char in other_characters[:3]:  # Limit to 3
                others_context += f"- {char.name}: {char.personality_summary or char.role.value}\n"
        
        user_prompt = f"""CHARACTER PROFILE:
{character_profile}
{others_context}

SCENE CONTEXT:
{scene_context}

DIALOGUE SITUATION:
{dialogue_situation}

Write {character.name}'s dialogue (include brief action beats if needed):"""
        
        return {
            "system_prompt": system_prompt,
            "context": "",
            "user_prompt": user_prompt
        }
    
    def build_consistency_check_prompt(
        self,
        content_to_check: str,
        characters: List[Character],
        plotlines: List[Plotline],
        story_bible: Optional[StoryBible],
        previous_content: str
    ) -> Dict[str, str]:
        """Build prompt for consistency analysis"""
        system_prompt = """You are an expert continuity editor and story analyst.

Your task is to identify any inconsistencies, contradictions, or violations in the provided content.

Be thorough but fair - flag genuine issues, not stylistic choices.

Categories to check:
1. CHARACTER CONSISTENCY - Do characters behave according to their established personalities?
2. TIMELINE ISSUES - Are there temporal contradictions?
3. WORLD RULE VIOLATIONS - Does anything break established rules?
4. POV CONSISTENCY - Is the point of view maintained?
5. TONE DRIFT - Does the tone match the established style?
6. FACTUAL CONTRADICTIONS - Do details contradict earlier content?"""
        
        context_parts = []
        
        # Character context
        if characters:
            char_context = "ESTABLISHED CHARACTERS:\n"
            for char in characters:
                char_context += f"\n{char.name} ({char.role.value}):\n"
                char_context += f"  Personality: {char.personality_summary or 'Not specified'}\n"
                char_context += f"  Current state: {char.current_emotional_state or 'Not specified'}\n"
            context_parts.append(char_context)
        
        # World rules
        if story_bible:
            rules_context = "WORLD RULES:\n"
            for rule in story_bible.world_rules[:10]:  # Limit
                rules_context += f"- {rule.title}: {rule.description[:200]}\n"
            context_parts.append(rules_context)
        
        # Previous content
        if previous_content:
            context_parts.append(f"PREVIOUS CONTENT (for reference):\n{previous_content[:2000]}")
        
        user_prompt = f"""CONTENT TO ANALYZE:
{content_to_check}

Provide a detailed consistency analysis. For each issue found, specify:
1. The issue type
2. The specific problem
3. Where it occurs
4. Suggested fix

If no issues are found, confirm the content is consistent."""
        
        return {
            "system_prompt": system_prompt,
            "context": "\n\n".join(context_parts),
            "user_prompt": user_prompt
        }
    
    def build_recap_prompt(
        self,
        story: Story,
        chapters: List[Chapter],
        characters: List[Character],
        plotlines: List[Plotline]
    ) -> Dict[str, str]:
        """Build prompt for story recap generation"""
        system_prompt = """You are an expert story analyst providing a comprehensive recap.

Your recap should be:
1. Clear and well-organized
2. Focused on essential information
3. Helpful for understanding where the story currently stands
4. Highlighting unresolved threads and character states"""
        
        # Build chapter summaries
        chapters_summary = "CHAPTER SUMMARIES:\n"
        for ch in chapters:
            summary = ch.summary or f"Chapter {ch.number}: {ch.title}"
            chapters_summary += f"\nChapter {ch.number} - {ch.title}:\n{summary[:300]}\n"
        
        # Build character states
        character_states = "CHARACTER STATES:\n"
        for char in characters:
            if char.role in [CharacterRole.PROTAGONIST, CharacterRole.ANTAGONIST, CharacterRole.DEUTERAGONIST]:
                character_states += f"\n{char.name}:\n"
                character_states += f"  Status: {char.status.value}\n"
                character_states += f"  Current state: {char.current_emotional_state or 'Unknown'}\n"
                character_states += f"  Location: {char.current_location or 'Unknown'}\n"
        
        # Build plotline states
        plotline_states = "ACTIVE PLOTLINES:\n"
        for plot in plotlines:
            if plot.status.value not in ["resolved", "abandoned"]:
                plotline_states += f"\n{plot.title} ({plot.status.value}):\n"
                plotline_states += f"  {plot.description[:200] if plot.description else 'No description'}\n"
        
        user_prompt = f"""Generate a comprehensive story recap covering:

1. **WHAT HAS HAPPENED** - Major events in chronological order
2. **CURRENT STATE** - Where things stand right now
3. **CHARACTER STATUS** - Key characters and their current situations
4. **UNRESOLVED THREADS** - Plot points that need resolution
5. **KEY THEMES** - Recurring themes and motifs

STORY: {story.title}
GENRE: {story.genre.value}
TONE: {story.tone.value}

{chapters_summary}

{character_states}

{plotline_states}

COMPREHENSIVE RECAP:"""
        
        return {
            "system_prompt": system_prompt,
            "context": "",
            "user_prompt": user_prompt
        }
    
    def build_brainstorm_prompt(
        self,
        story: Story,
        brainstorm_type: str,
        current_context: str,
        specific_request: Optional[str] = None
    ) -> Dict[str, str]:
        """Build prompt for creative brainstorming"""
        type_instructions = {
            "plot": "Generate creative plot developments, twists, and story directions.",
            "character": "Generate character development ideas, arc possibilities, and relationship dynamics.",
            "scene": "Generate scene ideas, settings, and dramatic moments.",
            "dialogue": "Generate dialogue approaches, conversations, and verbal confrontations.",
            "conflict": "Generate conflict ideas, obstacles, and tension-building elements.",
            "ending": "Generate potential endings, resolutions, and climactic moments."
        }
        
        system_prompt = f"""You are a creative writing partner brainstorming for a {story.genre.value} story.

{type_instructions.get(brainstorm_type, "Generate creative ideas for the story.")}

Provide diverse options ranging from safe/expected to bold/surprising.
Each idea should be actionable and fit the established story."""
        
        user_prompt = f"""STORY: {story.title}
GENRE: {story.genre.value}
TONE: {story.tone.value}

CURRENT CONTEXT:
{current_context}

{f"SPECIFIC REQUEST: {specific_request}" if specific_request else ""}

Generate 5 creative {brainstorm_type} ideas:

1. [SAFE] - A natural, expected progression
2. [MODERATE] - An interesting development
3. [CREATIVE] - An unexpected but fitting twist
4. [BOLD] - A surprising direction
5. [WILD CARD] - An unconventional choice

For each, provide:
- The idea
- Why it works for this story
- Potential complications/opportunities"""
        
        return {
            "system_prompt": system_prompt,
            "context": "",
            "user_prompt": user_prompt
        }
    
    # Private helper methods
    
    def _build_system_prompt(self, story: Story, writing_mode: WritingMode) -> str:
        """Build the system prompt based on story settings and mode"""
        genre_style = self.genre_styles.get(story.genre, "")
        tone_modifier = self.tone_modifiers.get(story.tone, "")
        
        mode_instruction = self._get_mode_instructions(writing_mode, "continuation")
        
        # Get language instruction (default to English if not set)
        language = getattr(story, 'language', 'English') or 'English'
        language_instruction = self._get_language_instruction(language)
        
        system_prompt = f"""You are an expert creative writer working on "{story.title}".

GENRE: {story.genre.value}
{genre_style}

TONE: {story.tone.value}
{tone_modifier}

POV: {story.pov_style}
TENSE: {story.tense}
LANGUAGE: {language}
{language_instruction}

{mode_instruction}

{story.writing_style or ''}

IMPORTANT FORMATTING RULES:
- Output ONLY plain prose text. NO HTML tags, NO markdown formatting.
- Use blank lines between paragraphs for separation.
- Do NOT use <p>, </p>, <br>, or any other HTML/XML tags.
- Do NOT use markdown like **, ##, or backticks.
- Just write natural prose with line breaks for paragraphs.

Maintain absolute consistency with established characters, events, and world rules."""
        
        return system_prompt
    
    def _build_story_overview(self, story: Story) -> str:
        """Build story overview section"""
        overview = f"STORY: {story.title}\n"
        if story.logline:
            overview += f"LOGLINE: {story.logline}\n"
        if story.setting_time:
            overview += f"TIME: {story.setting_time}\n"
        if story.setting_place:
            overview += f"PLACE: {story.setting_place}\n"
        return overview
    
    def _build_character_context(
        self,
        characters: List[Character],
        current_chapter: Optional[Chapter]
    ) -> str:
        """Build character context, prioritizing relevant characters"""
        if not characters:
            return ""
        
        # Sort by importance and relevance
        sorted_chars = sorted(characters, key=lambda c: c.importance, reverse=True)
        
        context = "KEY CHARACTERS:\n"
        
        for char in sorted_chars[:8]:  # Limit to top 8
            context += f"\n{char.name}"
            if char.role:
                context += f" ({char.role.value})"
            context += ":\n"
            
            if char.personality_summary:
                context += f"  Personality: {char.personality_summary[:150]}\n"
            if char.current_emotional_state:
                context += f"  Current state: {char.current_emotional_state}\n"
            if char.speaking_style:
                context += f"  Speech: {char.speaking_style[:100]}\n"
        
        return context
    
    def _build_plotline_context(self, plotlines: List[Plotline]) -> str:
        """Build active plotlines context"""
        if not plotlines:
            return ""
        
        active = [p for p in plotlines if p.status.value not in ["resolved", "abandoned"]]
        
        if not active:
            return ""
        
        context = "ACTIVE PLOTLINES:\n"
        for plot in active[:5]:  # Limit to 5
            context += f"\n- {plot.title} ({plot.status.value}): "
            context += f"{plot.description[:150] if plot.description else 'No description'}\n"
        
        return context
    
    def _build_world_rules_context(self, story_bible: StoryBible) -> str:
        """Build world rules context"""
        context_parts = []
        
        if story_bible.world_description:
            context_parts.append(f"WORLD: {story_bible.world_description[:300]}")
        
        if story_bible.world_rules:
            rules = "KEY RULES:\n"
            for rule in story_bible.world_rules[:5]:
                rules += f"- {rule.title}: {rule.description[:100]}\n"
            context_parts.append(rules)
        
        if story_bible.magic_system:
            context_parts.append(f"MAGIC SYSTEM: {story_bible.magic_system[:200]}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _build_retrieved_context(self, retrieved: List[str]) -> str:
        """Build context from retrieved semantic chunks (RAG memory)"""
        if not retrieved:
            return ""
        
        context = "RETRIEVED MEMORY (Relevant story context for consistency):\n"
        for chunk in retrieved[:8]:  # Allow more chunks for comprehensive context
            # Chunks now come pre-labeled from ai_generation route
            # e.g., "[Previous Scene] ...", "[Character Info] ...", "[WORLD_RULE] ..."
            context += f"\n{chunk[:500]}\n"
        
        return context
    
    def _build_detailed_character_profile(self, character: Character) -> str:
        """Build detailed character profile for dialogue generation"""
        profile = f"Name: {character.name}\n"
        profile += f"Role: {character.role.value}\n"
        
        if character.age:
            profile += f"Age: {character.age}\n"
        if character.occupation:
            profile += f"Occupation: {character.occupation}\n"
        if character.personality_summary:
            profile += f"Personality: {character.personality_summary}\n"
        if character.backstory:
            profile += f"Background: {character.backstory[:300]}\n"
        if character.motivation:
            profile += f"Motivation: {character.motivation}\n"
        if character.speaking_style:
            profile += f"Speaking style: {character.speaking_style}\n"
        if character.catchphrases:
            profile += f"Catchphrases: {', '.join(character.catchphrases)}\n"
        
        return profile
    
    def _build_user_prompt_continuation(
        self,
        chapter: Chapter,
        recent_content: str,
        user_direction: Optional[str],
        word_target: int,
        writing_mode: WritingMode
    ) -> str:
        """Build the user prompt for continuation"""
        prompt = f"CHAPTER {chapter.number}: {chapter.title}\n\n"
        
        if chapter.outline:
            prompt += f"CHAPTER OUTLINE: {chapter.outline[:300]}\n\n"
        
        prompt += f"RECENT CONTENT:\n{recent_content}\n\n"
        
        if user_direction:
            prompt += f"DIRECTION: {user_direction}\n\n"
        
        mode_prompts = {
            WritingMode.AI_LEAD: f"Continue the story with approximately {word_target} words. Take creative initiative.",
            WritingMode.USER_LEAD: f"Provide a suggested continuation of approximately {word_target} words, closely following established patterns.",
            WritingMode.CO_AUTHOR: f"Continue the story with approximately {word_target} words, balancing your input with the established direction."
        }
        
        prompt += mode_prompts.get(writing_mode, mode_prompts[WritingMode.CO_AUTHOR])
        
        return prompt
    
    def _get_mode_instructions(self, mode: WritingMode, task: str) -> str:
        """Get writing mode specific instructions"""
        instructions = {
            WritingMode.AI_LEAD: {
                "continuation": """YOU ARE IN FULL CREATIVE CONTROL MODE.
- Take bold narrative risks and make dramatic choices
- Introduce new plot developments, twists, and surprises
- Create vivid, cinematic scenes with rich sensory details
- Write expansively - don't be conservative or hold back
- Push characters into challenging situations and emotional territory
- Your creativity drives the story forward""",
                "rewrite": "Rewrite creatively. Elevate the prose significantly. Add depth and flavor while keeping story beats.",
                "dialogue": "Write memorable, character-defining dialogue. Be bold with voice and characterization."
            },
            WritingMode.USER_LEAD: {
                "continuation": """YOU ARE IN ASSISTANCE MODE.
- Follow the established direction exactly - no major deviations
- Continue the immediate scene naturally without introducing new elements
- Match the existing style, tone, and pacing precisely
- Keep prose simple and direct - let the user's voice dominate
- If uncertain, choose the most conservative, predictable option
- Your role is to extend, not to innovate""",
                "rewrite": "Make ONLY the specific requested changes. Preserve every other aspect of the writing exactly as is.",
                "dialogue": "Match established speech patterns exactly. No creative embellishments."
            },
            WritingMode.CO_AUTHOR: {
                "continuation": """YOU ARE A COLLABORATIVE CO-WRITER.
- Balance following the established story with contributing fresh ideas
- Suggest natural plot progressions that honor what's been set up
- Enhance scenes with good descriptive detail, but don't overwhelm
- Stay true to character voices while adding subtle depth
- Maintain the story's tone while keeping it engaging
- Think: professional writing partner, not ghost writer or copyist""",
                "rewrite": "Improve the writing quality while maintaining the author's core voice and intent. Thoughtful enhancements only.",
                "dialogue": "Write authentic dialogue that feels true to character. Add depth without changing their essential voice."
            }
        }
        
        return instructions.get(mode, instructions[WritingMode.CO_AUTHOR]).get(task, "")
    
    def _load_genre_styles(self) -> Dict[StoryGenre, str]:
        """Load genre-specific style guidelines"""
        return {
            StoryGenre.FANTASY: "Use vivid imagery. Embrace wonder and magic. Build an immersive world.",
            StoryGenre.SCIENCE_FICTION: "Ground fantastical elements in logic. Explore ideas and consequences.",
            StoryGenre.ROMANCE: "Focus on emotional connection. Build tension and chemistry. Satisfying payoffs.",
            StoryGenre.THRILLER: "Maintain tension. Short, punchy sentences in action. Keep readers on edge.",
            StoryGenre.MYSTERY: "Plant clues fairly. Build suspense. Reward attentive readers.",
            StoryGenre.HORROR: "Build dread gradually. Use atmosphere. Make the familiar unsettling.",
            StoryGenre.LITERARY: "Prioritize prose quality. Deep character exploration. Thematic resonance.",
            StoryGenre.HISTORICAL: "Authentic period detail. Balance accuracy with storytelling.",
            StoryGenre.ADVENTURE: "Keep momentum high. Exciting setpieces. Heroic moments.",
            StoryGenre.DRAMA: "Emotional authenticity. Complex relationships. Meaningful conflict.",
        }
    
    def _load_tone_modifiers(self) -> Dict[StoryTone, str]:
        """Load tone-specific modifiers"""
        return {
            StoryTone.DARK: "Embrace shadows and moral complexity. Don't shy from difficult themes.",
            StoryTone.LIGHT: "Keep spirits up. Even conflict should have hope. Celebrate joy.",
            StoryTone.HUMOROUS: "Find comedy in situations. Witty dialogue. Don't take things too seriously.",
            StoryTone.SERIOUS: "Treat events with weight. Consequences matter. Thoughtful approach.",
            StoryTone.WHIMSICAL: "Embrace the fantastical. Playful language. Delightful surprises.",
            StoryTone.GRITTY: "Raw and real. Don't sanitize. Show the rough edges.",
            StoryTone.ROMANTIC: "Heightened emotions. Beautiful moments. Heart-forward storytelling.",
            StoryTone.SUSPENSEFUL: "Keep readers guessing. Tension in every scene. Careful reveals.",
            StoryTone.MELANCHOLIC: "Bittersweet beauty. Loss and longing. Poetic sadness.",
            StoryTone.HOPEFUL: "Light through darkness. Redemption possible. Earned optimism.",
            StoryTone.EPIC: "Grand scale. Momentous events. Sweeping narrative.",
            StoryTone.INTIMATE: "Close focus. Personal stakes. Quiet power.",
        }
    
    def _get_language_instruction(self, language: str) -> str:
        """Get language-specific writing instructions"""
        language_instructions = {
            "English": "Write in natural, fluent English prose.",
            "Japanese": "日本語で自然な文章を書いてください。日本の文学的な表現や文化的なニュアンスを適切に使用してください。",
            "Chinese": "请用自然流畅的中文写作。使用适当的中文文学表达和文化细节。",
            "Korean": "자연스럽고 유창한 한국어로 작성하세요. 한국의 문학적 표현과 문화적 뉘앙스를 적절히 사용하세요.",
            "Spanish": "Escribe en español natural y fluido. Usa expresiones literarias apropiadas.",
            "French": "Écrivez en français naturel et fluide. Utilisez des expressions littéraires appropriées.",
            "German": "Schreiben Sie in natürlichem, fließendem Deutsch. Verwenden Sie angemessene literarische Ausdrücke.",
            "Portuguese": "Escreva em português natural e fluente. Use expressões literárias apropriadas.",
            "Russian": "Пишите естественной и плавной русской прозой. Используйте соответствующие литературные выражения.",
            "Italian": "Scrivi in italiano naturale e fluente. Usa espressioni letterarie appropriate.",
            "Thai": "เขียนเป็นภาษาไทยที่เป็นธรรมชาติและคล่องแคล่ว ใช้การแสดงออกทางวรรณกรรมที่เหมาะสม",
            "Vietnamese": "Viết bằng tiếng Việt tự nhiên và trôi chảy. Sử dụng các diễn đạt văn học phù hợp.",
            "Arabic": "اكتب بالعربية الطبيعية والسلسة. استخدم التعبيرات الأدبية المناسبة.",
            "Hindi": "स्वाभाविक और प्रवाहपूर्ण हिंदी में लिखें। उपयुक्त साहित्यिक अभिव्यक्तियों का उपयोग करें।",
            "Indonesian": "Tulis dalam bahasa Indonesia yang alami dan lancar. Gunakan ekspresi sastra yang sesuai.",
            "Malay": "Tulis dalam bahasa Melayu yang semula jadi dan lancar. Gunakan ungkapan sastera yang sesuai.",
            "Telugu": "తెలుగులో సహజంగా మరియు అవ్యవధానంగా రాయండి. తగిన సాహిత్య వ్యక్తీకరణలను ఉపయోగించండి.",
            "Malayalam": "സ്വാഭാവികവും ഒഴുക്കുള്ളതുമായ മലയാളത്തിൽ എഴുതുക. ഉചിതമായ സാഹിത്യ പ്രയോഗങ്ങൾ ഉപയോഗിക്കുക.",
            "Kannada": "ನೈಸರ್ಗಿಕ ಮತ್ತು ಸರಳವಾದ ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ. ಸೂಕ್ತವಾದ ಸಾಹಿತ್ಯಿಕ ಅಭಿವ್ಯಕ್ತಿಗಳನ್ನು ಬಳಸಿ.",
            "Tamil": "இயற்கையான மற்றும் சரளமான தமிழில் எழுதுங்கள். பொருத்தமான இலக்கிய வெளிப்பாடுகளைப் பயன்படுத்துங்கள்.",
        }
        
        return language_instructions.get(
            language, 
            f"Write in natural, fluent {language} prose. Use appropriate literary expressions and cultural nuances."
        )
