from __future__ import annotations

import re
import time
from typing import List, Optional, Tuple

from openai import OpenAI

SUMMARY_PROMPT = """
You are a memory processor.
Read the following session text and produce a concise summary paragraph.
Focus on facts, intents, preferences, events, and entities that are useful for future retrieval.

Session:
{session_text}

Return only the summary.
""".strip()

KEYWORD_PROMPT = """
You are a memory processor.
Read the following session text and extract concise retrieval-oriented keywords.
Output keywords separated by semicolons.

Session:
{session_text}

Return only the keyword list.
""".strip()


class OpenAICompatibleLLM:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_wait_sec: float = 2.0,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_wait_sec = retry_wait_sec

    def _complete(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                return (content or "").strip()
            except Exception as err:  # pragma: no cover - network/runtime dependent
                last_err = err
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_sec)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def summarize_and_keywords(self, session_text: str) -> Tuple[str, List[str]]:
        summary = self._complete(SUMMARY_PROMPT.format(session_text=session_text))
        raw_keywords = self._complete(KEYWORD_PROMPT.format(session_text=session_text))
        keywords = self._parse_keywords(raw_keywords)
        return summary, keywords

    @staticmethod
    def _parse_keywords(raw_keywords: str) -> List[str]:
        # Support semicolon/comma/newline outputs from different models.
        chunks = re.split(r"[;,\n，；]+", raw_keywords)
        keywords: List[str] = []
        seen = set()
        for chunk in chunks:
            token = chunk.strip()
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
        return keywords

