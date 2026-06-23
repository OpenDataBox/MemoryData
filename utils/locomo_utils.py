"""Shared helpers for LoCoMo chunk metadata and retrieval parsing."""

from __future__ import annotations

import re
from typing import Iterable, List


LOCOMO_META_PATTERN = re.compile(
    r"^\[LOCOMO_META chunk_id=(?P<chunk_id>[^\s\]]+) source_ids=(?P<source_ids>[^\]]*)\]\s*$",
    flags=re.MULTILINE,
)

LOCOMO_CATEGORY_LABELS = {
    "1": "Multi-hop",
    "2": "Temporal",
    "3": "Open-domain",
    "4": "Single-hop",
    "5": "Adversarial",
}

LOCOMO_CATEGORY_SLUGS = {
    "1": "multi_hop",
    "2": "temporal",
    "3": "open_domain",
    "4": "single_hop",
    "5": "adversarial",
}


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """Return unique non-empty values while preserving first-seen order."""
    seen = set()
    output = []
    for value in values or []:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def build_locomo_storage_text(text: str, chunk_id: str, source_ids: Iterable[str]) -> str:
    """Embed chunk metadata in a compact header for retriever-side parsing."""
    normalized_ids = dedupe_preserve_order(source_ids)
    header = f"[LOCOMO_META chunk_id={chunk_id} source_ids={','.join(normalized_ids)}]"
    body = str(text or "").strip()
    return f"{header}\n{body}".strip()


def parse_locomo_metadata(text: str) -> list[dict]:
    """Extract all LoCoMo metadata headers found in text."""
    matches = []
    for match in LOCOMO_META_PATTERN.finditer(str(text or "")):
        raw_source_ids = match.group("source_ids").strip()
        matches.append(
            {
                "chunk_id": match.group("chunk_id"),
                "source_ids": dedupe_preserve_order(raw_source_ids.split(",")) if raw_source_ids else [],
            }
        )
    return matches


def parse_locomo_source_ids(text: str) -> List[str]:
    """Return the union of all LoCoMo source ids present in text."""
    source_ids = []
    for metadata in parse_locomo_metadata(text):
        source_ids.extend(metadata["source_ids"])
    return dedupe_preserve_order(source_ids)


def strip_locomo_metadata(text: str) -> str:
    """Remove LoCoMo metadata headers from text while preserving body content."""
    stripped = LOCOMO_META_PATTERN.sub("", str(text or ""))
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def get_locomo_category_label(category) -> str:
    """Return a human-readable LoCoMo category label."""
    return LOCOMO_CATEGORY_LABELS.get(str(category), f"Unknown ({category})")


def get_locomo_category_slug(category) -> str:
    """Return a filesystem-safe LoCoMo category slug."""
    return LOCOMO_CATEGORY_SLUGS.get(str(category), f"category_{category}")
