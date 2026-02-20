"""
Consistency Engine - Analyzes story content for consistency issues
Checks character behavior, timeline, world rules, POV, and tone
"""
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import re

from app.models.story import Story
from app.models.chapter import Chapter
from app.models.character import Character
from app.models.plotline import Plotline
from app.models.story_bible import StoryBible, WorldRule
from app.models.generation import WritingMode
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)


class IssueType(str, Enum):
    """Types of consistency issues"""
    CHARACTER_BEHAVIOR = "character_behavior"
    CHARACTER_VOICE = "character_voice"
    TIMELINE_CONTRADICTION = "timeline_contradiction"
    WORLD_RULE_VIOLATION = "world_rule_violation"
    POV_INCONSISTENCY = "pov_inconsistency"
    TONE_DRIFT = "tone_drift"
    FACTUAL_CONTRADICTION = "factual_contradiction"
    PLOT_HOLE = "plot_hole"
    CONTINUITY_ERROR = "continuity_error"


class IssueSeverity(str, Enum):
    """Severity levels for issues"""
    LOW = "low"  # Minor, stylistic
    MEDIUM = "medium"  # Noticeable but not breaking
    HIGH = "high"  # Significant problem
    CRITICAL = "critical"  # Story-breaking issue


@dataclass
class ConsistencyIssue:
    """Represents a detected consistency issue"""
    type: IssueType
    severity: IssueSeverity
    description: str
    location: str  # Where in the text
    context: str  # Relevant context
    suggestion: Optional[str] = None
    affected_elements: List[str] = field(default_factory=list)  # Characters, plotlines, etc.


@dataclass
class ConsistencyReport:
    """Complete consistency analysis report"""
    content_analyzed: str
    issues: List[ConsistencyIssue]
    overall_score: float  # 0-100
    summary: str
    recommendations: List[str]
    
    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)
    
    @property
    def issue_count_by_type(self) -> Dict[IssueType, int]:
        counts = {}
        for issue in self.issues:
            counts[issue.type] = counts.get(issue.type, 0) + 1
        return counts


class ConsistencyEngine:
    """
    Engine for analyzing story consistency
    Combines rule-based checks with AI analysis
    """
    
    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service
    
    async def analyze_content(
        self,
        content: str,
        story: Story,
        characters: List[Character],
        plotlines: List[Plotline],
        story_bible: Optional[StoryBible],
        previous_chapters: List[Chapter],
        current_chapter: Optional[Chapter] = None
    ) -> ConsistencyReport:
        """
        Perform comprehensive consistency analysis
        
        Args:
            content: Text to analyze
            story: The story object
            characters: Story characters
            plotlines: Active plotlines
            story_bible: World-building rules
            previous_chapters: Previous chapter content/summaries
            current_chapter: Current chapter context
        
        Returns:
            ConsistencyReport with all findings
        """
        issues = []
        
        # Run rule-based checks
        issues.extend(await self._check_character_consistency(content, characters))
        issues.extend(await self._check_pov_consistency(content, story, current_chapter))
        issues.extend(await self._check_world_rules(content, story_bible))
        issues.extend(await self._check_timeline_consistency(content, previous_chapters))
        issues.extend(await self._check_tone_consistency(content, story))
        
        # Run AI-powered deep analysis
        ai_issues = await self._ai_deep_analysis(
            content=content,
            story=story,
            characters=characters,
            plotlines=plotlines,
            story_bible=story_bible,
            previous_content=self._get_previous_content_summary(previous_chapters)
        )
        issues.extend(ai_issues)
        
        # Calculate overall score
        score = self._calculate_consistency_score(issues)
        
        # Generate summary and recommendations
        summary = self._generate_summary(issues)
        recommendations = self._generate_recommendations(issues)
        
        return ConsistencyReport(
            content_analyzed=content[:500] + "..." if len(content) > 500 else content,
            issues=issues,
            overall_score=score,
            summary=summary,
            recommendations=recommendations
        )
    
    async def quick_check(
        self,
        content: str,
        characters: List[Character],
        story: Story
    ) -> List[ConsistencyIssue]:
        """
        Perform a quick consistency check (rule-based only)
        Useful for real-time checking during writing
        """
        issues = []
        
        # Quick character checks
        issues.extend(await self._check_character_names(content, characters))
        issues.extend(await self._check_pov_consistency(content, story, None))
        issues.extend(await self._check_tense_consistency(content, story))
        
        return issues
    
    async def check_new_generation(
        self,
        generated_content: str,
        context_content: str,
        story: Story,
        characters: List[Character]
    ) -> List[ConsistencyIssue]:
        """
        Check AI-generated content before accepting
        """
        issues = []
        
        # Check character consistency in generated content
        issues.extend(await self._check_character_consistency(generated_content, characters))
        
        # Check for contradictions with context
        issues.extend(await self._check_contradictions(generated_content, context_content))
        
        # Check tone match
        issues.extend(await self._check_tone_consistency(generated_content, story))
        
        return issues
    
    # Rule-based check methods
    
    async def _check_character_consistency(
        self,
        content: str,
        characters: List[Character]
    ) -> List[ConsistencyIssue]:
        """Check for character behavior and voice consistency"""
        issues = []
        
        for character in characters:
            if character.name.lower() not in content.lower():
                continue
            
            # Extract character's dialogue and actions
            char_content = self._extract_character_content(content, character.name)
            
            if not char_content:
                continue
            
            # Check speech patterns
            if character.speaking_style:
                style_issues = self._check_speaking_style(
                    char_content["dialogue"],
                    character.speaking_style,
                    character.name
                )
                issues.extend(style_issues)
            
            # Check catchphrases usage
            if character.catchphrases:
                # Could check if catchphrases are used appropriately
                pass
            
            # Check for out-of-character actions
            if character.personality_traits:
                trait_issues = self._check_personality_alignment(
                    char_content["actions"],
                    character.personality_traits,
                    character.name
                )
                issues.extend(trait_issues)
        
        return issues
    
    async def _check_pov_consistency(
        self,
        content: str,
        story: Story,
        chapter: Optional[Chapter]
    ) -> List[ConsistencyIssue]:
        """Check for POV consistency"""
        issues = []
        
        pov_style = story.pov_style
        
        # Check for POV shifts
        if pov_style == "first_person":
            # Check for sudden third-person intrusions
            third_person_patterns = [
                r'\bhe thought\b', r'\bshe thought\b',
                r'\bhe felt\b', r'\bshe felt\b',
                r'[A-Z][a-z]+ wondered\b'
            ]
            for pattern in third_person_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    issues.append(ConsistencyIssue(
                        type=IssueType.POV_INCONSISTENCY,
                        severity=IssueSeverity.MEDIUM,
                        description=f"Possible POV shift to third person detected",
                        location="Multiple locations",
                        context=f"Found: {matches[:3]}",
                        suggestion="Ensure first-person POV is maintained throughout"
                    ))
        
        elif pov_style == "third_person_limited":
            # Check for head-hopping
            # This is complex and better handled by AI analysis
            pass
        
        return issues
    
    async def _check_world_rules(
        self,
        content: str,
        story_bible: Optional[StoryBible]
    ) -> List[ConsistencyIssue]:
        """Check for world rule violations"""
        issues = []
        
        if not story_bible or not story_bible.world_rules:
            return issues
        
        for rule in story_bible.world_rules:
            if not rule.is_strict:
                continue
            
            # Check for potential violations
            # This is simplified - real implementation would be more sophisticated
            violation_keywords = self._extract_rule_keywords(rule)
            
            for keyword in violation_keywords:
                if keyword.lower() in content.lower():
                    # Potential violation - would need AI to confirm
                    issues.append(ConsistencyIssue(
                        type=IssueType.WORLD_RULE_VIOLATION,
                        severity=IssueSeverity.MEDIUM,
                        description=f"Potential violation of rule: {rule.title}",
                        location="Content contains relevant keywords",
                        context=rule.description[:200],
                        suggestion=f"Verify content aligns with: {rule.description[:100]}"
                    ))
        
        return issues
    
    async def _check_timeline_consistency(
        self,
        content: str,
        previous_chapters: List[Chapter]
    ) -> List[ConsistencyIssue]:
        """Check for timeline contradictions"""
        issues = []
        
        # Extract time references
        time_patterns = [
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+(ago|later|before|after)\b',
            r'\b(morning|afternoon|evening|night|dawn|dusk)\b',
            r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b'
        ]
        
        time_refs = []
        for pattern in time_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            time_refs.extend(matches)
        
        # Would compare with previous chapter timeline markers
        # Simplified for now
        
        return issues
    
    async def _check_tone_consistency(
        self,
        content: str,
        story: Story
    ) -> List[ConsistencyIssue]:
        """Check for tone drift from established style"""
        issues = []
        
        tone = story.tone
        
        # Tone indicators
        tone_checks = {
            "dark": {
                "expected": ["shadow", "dark", "blood", "death", "fear", "dread"],
                "unexpected": ["cheerful", "sunny", "delightful", "merry"]
            },
            "humorous": {
                "expected": ["laugh", "chuckle", "grin", "joke", "amusing"],
                "unexpected": ["despair", "anguish", "torment"]
            },
            "romantic": {
                "expected": ["heart", "love", "tender", "soft", "warm"],
                "unexpected": ["gore", "brutal", "violent"]
            }
        }
        
        if tone.value in tone_checks:
            checks = tone_checks[tone.value]
            
            # Count unexpected words
            unexpected_count = sum(
                1 for word in checks["unexpected"]
                if word.lower() in content.lower()
            )
            
            expected_count = sum(
                1 for word in checks["expected"]
                if word.lower() in content.lower()
            )
            
            if unexpected_count > expected_count and unexpected_count > 2:
                issues.append(ConsistencyIssue(
                    type=IssueType.TONE_DRIFT,
                    severity=IssueSeverity.LOW,
                    description=f"Tone may be drifting from established {tone.value} tone",
                    location="Throughout content",
                    context=f"Found {unexpected_count} tone-inconsistent words",
                    suggestion=f"Review content to ensure it maintains {tone.value} tone"
                ))
        
        return issues
    
    async def _check_tense_consistency(
        self,
        content: str,
        story: Story
    ) -> List[ConsistencyIssue]:
        """Check for tense consistency"""
        issues = []
        
        expected_tense = story.tense
        
        # Simple tense detection
        past_indicators = len(re.findall(r'\b\w+ed\b', content))
        present_indicators = len(re.findall(r'\b(is|are|am|does|do)\b', content, re.IGNORECASE))
        
        # Very rough heuristic
        if expected_tense == "past" and present_indicators > past_indicators:
            issues.append(ConsistencyIssue(
                type=IssueType.CONTINUITY_ERROR,
                severity=IssueSeverity.MEDIUM,
                description="Possible tense inconsistency - present tense detected in past tense story",
                location="Throughout content",
                context="",
                suggestion="Review verb tenses for consistency"
            ))
        
        return issues
    
    async def _check_character_names(
        self,
        content: str,
        characters: List[Character]
    ) -> List[ConsistencyIssue]:
        """Check for character name misspellings or inconsistencies"""
        issues = []
        
        # Build name variants
        name_variants = {}
        for char in characters:
            variants = [char.name]
            if char.aliases:
                variants.extend(char.aliases)
            name_variants[char.name] = variants
        
        # Check for potential misspellings
        # Simplified - would use fuzzy matching in production
        
        return issues
    
    async def _check_contradictions(
        self,
        new_content: str,
        existing_content: str
    ) -> List[ConsistencyIssue]:
        """Check for contradictions between new and existing content"""
        issues = []
        
        # This would ideally use AI to detect semantic contradictions
        # For now, just a placeholder
        
        return issues
    
    # AI-powered analysis
    
    async def _ai_deep_analysis(
        self,
        content: str,
        story: Story,
        characters: List[Character],
        plotlines: List[Plotline],
        story_bible: Optional[StoryBible],
        previous_content: str
    ) -> List[ConsistencyIssue]:
        """Use AI for deep consistency analysis"""
        issues = []
        
        # Build context for AI
        characters_context = "\n".join([
            f"- {c.name}: {c.personality_summary or c.role.value}"
            for c in characters[:10]
        ])
        
        world_rules_context = ""
        if story_bible and story_bible.world_rules:
            world_rules_context = "\n".join([
                f"- {r.title}: {r.description[:100]}"
                for r in story_bible.world_rules[:10]
            ])
        
        # Call AI service
        try:
            result = await self.gemini_service.analyze_consistency(
                content=content,
                characters=characters_context,
                world_rules=world_rules_context,
                previous_events=previous_content[:2000]
            )
            
            if result.get("success"):
                # Parse AI response into issues
                ai_issues = self._parse_ai_analysis(result["content"])
                issues.extend(ai_issues)
        except Exception as e:
            logger.error(f"AI consistency analysis failed: {e}")
        
        return issues
    
    def _parse_ai_analysis(self, ai_response: str) -> List[ConsistencyIssue]:
        """Parse AI analysis response into structured issues"""
        issues = []
        
        # Simple parsing - look for issue patterns
        # In production, would use structured output or more sophisticated parsing
        
        issue_patterns = {
            "character": IssueType.CHARACTER_BEHAVIOR,
            "timeline": IssueType.TIMELINE_CONTRADICTION,
            "world": IssueType.WORLD_RULE_VIOLATION,
            "pov": IssueType.POV_INCONSISTENCY,
            "tone": IssueType.TONE_DRIFT,
            "contradiction": IssueType.FACTUAL_CONTRADICTION
        }
        
        lines = ai_response.split("\n")
        for line in lines:
            line_lower = line.lower()
            for keyword, issue_type in issue_patterns.items():
                if keyword in line_lower and ("issue" in line_lower or "inconsisten" in line_lower or "problem" in line_lower):
                    issues.append(ConsistencyIssue(
                        type=issue_type,
                        severity=IssueSeverity.MEDIUM,  # AI would determine severity
                        description=line.strip(),
                        location="AI-identified",
                        context="",
                        suggestion=""
                    ))
                    break
        
        return issues
    
    # Helper methods
    
    def _extract_character_content(
        self,
        content: str,
        character_name: str
    ) -> Dict[str, List[str]]:
        """Extract dialogue and actions for a character"""
        result = {"dialogue": [], "actions": []}
        
        # Extract dialogue (simplified)
        dialogue_pattern = rf'{character_name}\s+said[,:]?\s*"([^"]+)"'
        result["dialogue"] = re.findall(dialogue_pattern, content, re.IGNORECASE)
        
        # Extract actions (simplified)
        action_pattern = rf'{character_name}\s+(walked|ran|looked|smiled|frowned|thought|felt|grabbed|pulled|pushed)'
        result["actions"] = re.findall(action_pattern, content, re.IGNORECASE)
        
        return result
    
    def _check_speaking_style(
        self,
        dialogue: List[str],
        expected_style: str,
        character_name: str
    ) -> List[ConsistencyIssue]:
        """Check if dialogue matches expected speaking style"""
        issues = []
        
        # This would analyze vocabulary, sentence structure, etc.
        # Simplified for now
        
        return issues
    
    def _check_personality_alignment(
        self,
        actions: List[str],
        traits: List[str],
        character_name: str
    ) -> List[ConsistencyIssue]:
        """Check if actions align with personality traits"""
        issues = []
        
        # Would check for trait/action conflicts
        # Simplified for now
        
        return issues
    
    def _extract_rule_keywords(self, rule: WorldRule) -> List[str]:
        """Extract keywords from a world rule for checking"""
        # Simple keyword extraction
        text = f"{rule.title} {rule.description}"
        words = text.split()
        # Filter to meaningful words
        keywords = [w for w in words if len(w) > 4]
        return keywords[:10]
    
    def _get_previous_content_summary(self, chapters: List[Chapter]) -> str:
        """Get summary of previous chapters"""
        summaries = []
        for ch in chapters[-3:]:  # Last 3 chapters
            if ch.summary:
                summaries.append(f"Chapter {ch.number}: {ch.summary[:200]}")
        return "\n".join(summaries)
    
    def _calculate_consistency_score(self, issues: List[ConsistencyIssue]) -> float:
        """Calculate overall consistency score"""
        if not issues:
            return 100.0
        
        # Deduct points based on issue severity
        deductions = {
            IssueSeverity.LOW: 2,
            IssueSeverity.MEDIUM: 5,
            IssueSeverity.HIGH: 15,
            IssueSeverity.CRITICAL: 30
        }
        
        total_deduction = sum(deductions.get(i.severity, 5) for i in issues)
        return max(0, 100 - total_deduction)
    
    def _generate_summary(self, issues: List[ConsistencyIssue]) -> str:
        """Generate human-readable summary"""
        if not issues:
            return "No consistency issues detected. Content appears consistent with established story elements."
        
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.type.value] = issue_counts.get(issue.type.value, 0) + 1
        
        summary_parts = [f"Found {len(issues)} consistency issues:"]
        for issue_type, count in issue_counts.items():
            summary_parts.append(f"  - {issue_type.replace('_', ' ').title()}: {count}")
        
        critical = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        if critical:
            summary_parts.append(f"\n⚠️ {len(critical)} critical issues require immediate attention.")
        
        return "\n".join(summary_parts)
    
    def _generate_recommendations(self, issues: List[ConsistencyIssue]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Group by type
        by_type = {}
        for issue in issues:
            if issue.type not in by_type:
                by_type[issue.type] = []
            by_type[issue.type].append(issue)
        
        # Generate type-specific recommendations
        if IssueType.CHARACTER_BEHAVIOR in by_type:
            recommendations.append(
                "Review character profiles and ensure actions align with established personalities."
            )
        
        if IssueType.TIMELINE_CONTRADICTION in by_type:
            recommendations.append(
                "Create a timeline document to track when events occur in your story."
            )
        
        if IssueType.WORLD_RULE_VIOLATION in by_type:
            recommendations.append(
                "Reference your story bible when writing scenes involving magic, technology, or world-specific elements."
            )
        
        if IssueType.POV_INCONSISTENCY in by_type:
            recommendations.append(
                "Review POV rules and ensure you're not revealing information the POV character wouldn't know."
            )
        
        if IssueType.TONE_DRIFT in by_type:
            recommendations.append(
                "Re-read previous chapters to recalibrate the established tone before continuing."
            )
        
        return recommendations
