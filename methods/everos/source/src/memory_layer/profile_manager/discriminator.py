"""Value discriminator for profile extraction - determines if memcell contains profile-worthy content."""

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from memory_layer.llm.llm_provider import LLMProvider
from core.observation.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DiscriminatorConfig:
    """Configuration for value discrimination.
    
    Attributes:
        min_confidence: Minimum confidence threshold (0.0-1.0)
        use_context: Whether to use previous memcells as context
        context_window: Number of previous memcells to include as context
    """
    
    min_confidence: float = 0.6
    use_context: bool = True
    context_window: int = 2


class ValueDiscriminator:
    """LLM-based discriminator to judge if a memcell contains high-value profile information.
    
    This component uses an LLM to analyze memcells and determine whether they contain
    concrete, attributable information worth extracting into user profiles.
    
    For group_chat scenario, it looks for:
    - Role/responsibility statements
    - Skill demonstrations or mentions
    - Project participation
    - Working habits and preferences
    - Personality indicators
    - Decision-making patterns
    
    For assistant scenario, it focuses on:
    - Stable personal traits
    - Enduring preferences
    - Personality dimensions
    - Decision-making style
    - Routines and habits
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[DiscriminatorConfig] = None,
        scenario: str = "group_chat"
    ):
        """Initialize value discriminator.
        
        Args:
            llm_provider: LLM provider for discrimination
            config: Discriminator configuration
            scenario: "group_chat" or "assistant"
        """
        self.llm_provider = llm_provider
        self.config = config or DiscriminatorConfig()
        self.scenario = scenario.lower()
    
    async def is_high_value(
        self,
        latest_memcell: Any,
        recent_memcells: Optional[List[Any]] = None
    ) -> Tuple[bool, float, str]:
        """Determine if the latest memcell contains high-value profile information.
        
        Args:
            latest_memcell: The memcell to evaluate
            recent_memcells: Previous memcells for context (optional)
        
        Returns:
            Tuple of (is_high_value, confidence, reason)
        """
        recent_memcells = recent_memcells or []
        
        # Build prompt based on scenario
        if self.scenario == "assistant":
            prompt = self._build_assistant_prompt(latest_memcell, recent_memcells)
        else:
            prompt = self._build_group_chat_prompt(latest_memcell, recent_memcells)
        
        try:
            response = await self.llm_provider.generate(prompt, temperature=0.0)
            is_high, conf, reason = self._parse_response(response)
            
            # Apply confidence threshold
            if is_high and conf >= self.config.min_confidence:
                return True, conf, reason
            else:
                return False, conf, reason or "Below confidence threshold"
        
        except Exception as e:
            logger.warning(f"Value discrimination failed: {e}")
            return False, 0.0, f"Discrimination error: {str(e)}"
    
    def _build_group_chat_prompt(
        self,
        latest: Any,
        recent: List[Any]
    ) -> str:
        """Build prompt for group_chat scenario."""
        context_texts = []
        if self.config.use_context and recent:
            window = recent[-self.config.context_window:]
            for i, mc in enumerate(window):
                text = self._extract_text(mc)
                if text:
                    context_texts.append(f"[Context {i+1}]\n{text}")
        
        latest_text = self._extract_text(latest)
        context_block = "\n\n".join(context_texts) if context_texts else "No context available"
        
        prompt = f"""You are a precise profile value discriminator for work/group chat scenario.

Given the latest conversation MemCell and recent context, determine if the latest MemCell contains 
new, concrete, and attributable information about user profile fields such as:

Profile Fields to Consider:
- role_responsibility: User's role, duties, responsibilities
- hard_skills: Technical skills, tools, technologies
- soft_skills: Communication, leadership, collaboration
- projects_participated: Project names, roles, contributions
- working_habit_preference: Work style, preferences, routines
- personality: Character traits, temperament
- way_of_decision_making: Decision patterns, priorities
- interests: Professional interests, areas of focus
- tendency: Behavioral tendencies, patterns

Rules for Judgment:
1. Reject small talk, vague statements, or non-attributable content
2. Prefer explicit statements (e.g., "I am responsible for X", "I have experience with Y")
3. Look for concrete evidence, not assumptions
4. Consider if the information is stable/lasting vs transient
5. Ensure the information is clearly attributable to a specific user

Context (Previous MemCells):
{context_block}

Latest MemCell to Evaluate:
{latest_text}

Respond with strict JSON only (no extra text):
{{
  "is_high_value": true/false,
  "confidence": 0.0-1.0,
  "reasons": "Brief explanation of your judgment"
}}"""
        
        return prompt
    
    def _build_assistant_prompt(
        self,
        latest: Any,
        recent: List[Any]
    ) -> str:
        """Build prompt for assistant/companion scenario."""
        context_texts = []
        if self.config.use_context and recent:
            window = recent[-self.config.context_window:]
            for i, mc in enumerate(window):
                text = self._extract_text(mc)
                if text:
                    context_texts.append(f"[Context {i+1}]\n{text}")
        
        latest_text = self._extract_text(latest)
        context_block = "\n\n".join(context_texts) if context_texts else "No context available"
        
        prompt = f"""You are a precise value discriminator for companion/assistant scenario.

Determine if the latest MemCell reveals stable personal traits or preferences worth capturing:

Profile Fields to Consider:
- personality: Enduring personality dimensions (Big Five, MBTI indicators)
- way_of_decision_making: Stable decision-making patterns
- interests: Long-term hobbies, passions, areas of interest
- tendency: Behavioral patterns, recurring preferences
- value_system: Core values, beliefs, principles
- motivation_system: What drives/motivates the user
- working_habit_preference: Routines, habits, preferences

Rules for Judgment:
1. Focus on stable, enduring traits (not transient moods or one-time events)
2. Reject casual chit-chat and vague statements
3. Look for repeated patterns or explicit self-descriptions
4. Prefer concrete examples over abstract claims
5. Ensure information is clearly attributable

Context (Previous MemCells):
{context_block}

Latest MemCell to Evaluate:
{latest_text}

Respond with strict JSON only (no extra text):
{{
  "is_high_value": true/false,
  "confidence": 0.0-1.0,
  "reasons": "Brief explanation of your judgment"
}}"""
        
        return prompt
    
    def _extract_text(self, memcell: Any) -> str:
        """Extract representative text from a memcell.
        
        Priority: episode > summary > original_data
        """
        if memcell is None:
            return ""
        
        # Try episode first
        episode = getattr(memcell, "episode", None)
        if isinstance(episode, str) and episode.strip():
            return episode.strip()
        
        # Try summary
        summary = getattr(memcell, "summary", None)
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
        
        # Fallback to compact original_data
        lines = []
        original_data = getattr(memcell, "original_data", None)
        if isinstance(original_data, list):
            for item in original_data[:5]:  # Limit to first 5 messages
                if isinstance(item, dict):
                    content = item.get("content") or item.get("summary")
                    if content:
                        text = str(content).strip()
                        if text:
                            lines.append(text)
        
        return "\n".join(lines) if lines else "Empty memcell"
    
    def _parse_response(self, response: str) -> Tuple[bool, float, str]:
        """Parse LLM response to extract judgment.
        
        Returns:
            (is_high_value, confidence, reasons)
        """
        if not response:
            return False, 0.0, "Empty response"
        
        # Try to extract JSON from code blocks first
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        
        payload: Optional[Dict[str, Any]] = None
        
        try:
            if fenced_match:
                payload = json.loads(fenced_match.group(1))
            else:
                # Try direct JSON parsing
                try:
                    payload = json.loads(response)
                except json.JSONDecodeError:
                    # Try AST literal_eval as fallback
                    parsed = ast.literal_eval(response)
                    if isinstance(parsed, dict):
                        payload = parsed
        except Exception:
            # Last resort: find first {...} in response
            obj_match = re.search(r"\{[\s\S]*?\}", response)
            if obj_match:
                try:
                    payload = json.loads(obj_match.group())
                except Exception:
                    pass
        
        if not payload:
            logger.warning(f"Failed to parse discriminator response: {response[:200]}")
            return False, 0.0, "Failed to parse response"
        
        is_high = bool(payload.get("is_high_value", False))
        conf = float(payload.get("confidence", 0.0) or 0.0)
        reasons = str(payload.get("reasons", ""))
        
        return is_high, conf, reasons

