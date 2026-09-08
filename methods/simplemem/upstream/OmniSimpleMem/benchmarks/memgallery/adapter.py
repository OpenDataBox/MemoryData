"""
OmniMem adapter for the Mem-Gallery benchmark.

Implements the ExplicitMemory interface from Mem-Gallery's memengine
and uses OmniMemoryOrchestrator as the backend.

Usage:
    # In Mem-Gallery's benchmark runner, register this as a memory engine:
    from benchmarks.memgallery.adapter import OmniMemAdapter
    adapter = OmniMemAdapter(config)
    adapter.store(observation)
    context = adapter.recall(query)
"""

import os, sys, json, re, time, logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from omni_memory import OmniMemoryOrchestrator, OmniMemoryConfig

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars per token for GPT models)."""
    return len(text) // 4 + 1


_PUNCT_RE = re.compile(r'[^\w\s]')

_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'was', 'were', 'are', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'and', 'but', 'or', 'nor',
    'not', 'so', 'very', 'just', 'about', 'up', 'down', 'that', 'this',
    'these', 'those', 'it', 'its', 'he', 'she', 'they', 'them', 'his',
    'her', 'their', 'my', 'your', 'our', 'what', 'which', 'who', 'whom',
    'how', 'when', 'where', 'why', 'if', 'than', 'too', 'also', 'both',
    'each', 'all', 'any', 'some', 'such', 'no', 'more', 'most', 'other',
    'only', 'same', 'own', 'here', 'there', 'please', 'based', 'conversation',
    'did', 'does', 'user', 'assistant', 'question', 'answer', 'mention',
    'mentioned', 'discuss', 'discussed', 'talk', 'talked', 'say', 'said',
})


def _extract_key_phrases(query: str) -> List[str]:
    """Extract key noun phrases from a question for multi-BM25 search."""
    phrases = []
    quoted = re.findall(r'"([^"]+)"', query)
    for q in quoted:
        if len(q.split()) <= 5:
            phrases.append(q)
    words = query.split()
    i = 0
    while i < len(words):
        w = words[i]
        if i == 0 or w.lower() in ('did', 'does', 'do', 'is', 'was', 'what', 'which',
                                     'who', 'how', 'when', 'where', 'why', 'the', 'a',
                                     'an', 'in', 'on', 'at', 'for', 'to', 'and', 'or',
                                     'based', 'please', 'if'):
            i += 1
            continue
        if w[0].isupper() and w.lower() not in _STOP_WORDS:
            phrase_words = [w]
            j = i + 1
            while j < len(words) and words[j][0].isupper():
                phrase_words.append(words[j])
                j += 1
            phrase = _PUNCT_RE.sub('', ' '.join(phrase_words)).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
            i = j
        else:
            i += 1
    content_words = []
    for w in words:
        clean = _PUNCT_RE.sub('', w).lower()
        if clean and len(clean) >= 4 and clean not in _STOP_WORDS:
            content_words.append(clean)
    if len(content_words) >= 2:
        for k in range(len(content_words) - 1):
            bigram = f"{content_words[k]} {content_words[k+1]}"
            if bigram not in phrases:
                phrases.append(bigram)
                if len(phrases) >= 5:
                    break
    return phrases[:3]


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return _PUNCT_RE.sub(' ', text.lower()).split()


class OmniMemAdapter:
    """
    OmniMem adapter for Mem-Gallery benchmark.

    Wraps OmniMemoryOrchestrator to provide store/recall interface
    with category-aware retrieval strategies for 9 QA task types.

    Category markers ([AR], [CD], [VS], [VR], [TR], [FR], [KR], [MR], [TTL])
    in queries trigger specialized retrieval paths.
    """

    def __init__(self, data_dir: str = "./omni_memory_eval_data", config: Optional[OmniMemoryConfig] = None):
        if config is None:
            config = OmniMemoryConfig.create_default()
        self._data_dir = data_dir
        self.orchestrator = OmniMemoryOrchestrator(config=config, data_dir=data_dir)
        self.top_k: int = 20

        # BM25 index for keyword search (built lazily)
        self._bm25 = None
        self._bm25_mau_ids: List[str] = []
        self._bm25_texts: List[str] = []

        # Image catalog and BM25 (built lazily for VS/VR)
        self._image_catalog: Optional[Dict[str, Dict]] = None
        self._image_bm25 = None
        self._image_bm25_mau_ids: List[str] = []
        self._image_bm25_texts: List[List[str]] = []

    def reset(self) -> None:
        """Reset memory state."""
        config = OmniMemoryConfig.create_default()
        try:
            if self.orchestrator is not None:
                self.orchestrator.close()
        except Exception:
            pass
        self.orchestrator = OmniMemoryOrchestrator(config=config, data_dir=self._data_dir)
        self._bm25 = None
        self._bm25_mau_ids = []
        self._bm25_texts = []
        self._image_catalog = None
        self._image_bm25 = None
        self._image_bm25_mau_ids = []
        self._image_bm25_texts = []

    def store(self, observation: Any) -> None:
        """Store one dialog turn into memory."""
        if self.orchestrator is None:
            return
        text = ""
        tags = []
        if isinstance(observation, dict):
            text = observation.get("text", "") or ""
            dialogue_id = observation.get("dialogue_id")
            if dialogue_id is not None:
                tags.append(f"dialogue_id:{dialogue_id}")
                if ":" in str(dialogue_id):
                    session_part = str(dialogue_id).split(":")[0]
                    tags.append(f"session_id:{session_part}")
            timestamp = observation.get("timestamp")
            if timestamp:
                tags.append(f"timestamp:{timestamp}")
            image_info = observation.get("image")
            if isinstance(image_info, dict):
                img_id = image_info.get("img_id")
                if img_id:
                    tags.append(f"image_id:{img_id}")
        else:
            text = str(observation)
        if not text:
            return
        try:
            self.orchestrator.add_text(text, tags=tags or None)
        except Exception:
            pass

    def _build_image_catalog(self) -> None:
        """Build catalog of image-containing MAUs for VS/VR queries."""
        if self._image_catalog is not None:
            return
        self._image_catalog = {}
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        store_dir = Path(self._data_dir) / "index" / "mau_store"
        if not store_dir.exists():
            return
        for jsonl_file in sorted(store_dir.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        mau_data = json.loads(line.strip())
                        mau_id = mau_data.get("id", "")
                        if not mau_id:
                            continue
                        tags = mau_data.get("metadata", {}).get("tags", [])
                        image_ids = [t.split(":", 1)[1] for t in tags if t.startswith("image_id:")]
                        if not image_ids:
                            continue
                        details = mau_data.get("details", {})
                        full_text = details.get("full_text", "") if isinstance(details, dict) else ""
                        caption = ""
                        if "image_caption:" in full_text:
                            caption = full_text.split("image_caption:", 1)[1].strip()
                        for img_id in image_ids:
                            session_id = img_id.split(":")[0] if ":" in img_id else ""
                            self._image_catalog[mau_id] = {
                                "image_id": img_id,
                                "caption": caption,
                                "session_id": session_id,
                                "full_text": full_text,
                                "tags": tags,
                            }
                        search_text = full_text if full_text else caption
                        if search_text:
                            self._image_bm25_mau_ids.append(mau_id)
                            self._image_bm25_texts.append(_tokenize(search_text))
                    except Exception:
                        continue
        if self._image_bm25_texts:
            self._image_bm25 = BM25Okapi(self._image_bm25_texts)

    def _image_search(self, query: str, return_all: bool = False, top_k: int = 30) -> List[Dict]:
        """Search image catalog using BM25."""
        self._build_image_catalog()
        if not self._image_bm25 or not self._image_bm25_mau_ids:
            return []
        tokenized_query = _tokenize(query)
        scores = self._image_bm25.get_scores(tokenized_query)
        if return_all:
            indices = [i for i in range(len(scores)) if scores[i] > 0]
            indices.sort(key=lambda i: scores[i], reverse=True)
        else:
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            indices = [i for i in indices if scores[i] > 0]
        results = []
        for i in indices:
            mau_id = self._image_bm25_mau_ids[i]
            entry = self._image_catalog.get(mau_id, {})
            if entry:
                results.append({**entry, "mau_id": mau_id, "score": scores[i]})
        return results

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all stored MAUs."""
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        mau_store = getattr(self.orchestrator, 'mau_store', None)
        if not mau_store:
            return
        all_maus = mau_store.get_all() if hasattr(mau_store, 'get_all') else []
        if not all_maus:
            store_dir = Path(self._data_dir) / "index" / "mau_store"
            if store_dir.exists():
                for jsonl_file in sorted(store_dir.glob("*.jsonl")):
                    with open(jsonl_file) as f:
                        for line in f:
                            try:
                                mau_data = json.loads(line.strip())
                                mau_id = mau_data.get("id", "")
                                details = mau_data.get("details", {})
                                text = ""
                                if isinstance(details, dict):
                                    text = details.get("full_text", "")
                                if not text:
                                    text = mau_data.get("summary", "")
                                if text and mau_id:
                                    self._bm25_mau_ids.append(mau_id)
                                    self._bm25_texts.append(_tokenize(text))
                            except Exception:
                                continue
        if self._bm25_texts:
            self._bm25 = BM25Okapi(self._bm25_texts)

    def _bm25_search(self, query: str, top_k: int = 10) -> List[str]:
        """Search using BM25 and return MAU IDs."""
        if self._bm25 is None:
            self._build_bm25_index()
        if self._bm25 is None or not self._bm25_mau_ids:
            return []
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._bm25_mau_ids[i] for i in top_indices if scores[i] > 0]

    def _get_dynamic_top_k(self, query: str, category: Optional[str]) -> int:
        """Estimate optimal top_k based on query complexity and category."""
        _CAT_TOP_K = {
            "AR": 10, "CD": 15, "VS": 15, "VR": 15, "TR": 20,
            "FR": 30, "KR": 20, "MR": 25, "TTL": 15,
        }
        base_k = _CAT_TOP_K.get(category, self.top_k)
        q_lower = query.lower()
        if any(kw in q_lower for kw in ("list all", "please list", "what are all", "name all", "what were all")):
            base_k = max(base_k, 35)
        if "how many" in q_lower:
            base_k = max(base_k, 30)
        proper_nouns = [w for w in query.split() if w[0:1].isupper() and w.lower() not in _STOP_WORDS]
        if len(proper_nouns) >= 3:
            base_k = max(base_k, 25)
        return base_k

    def recall(self, query: Any) -> str:
        """
        Recall relevant memories. Supports category markers [VS], [VR], etc.
        Returns metadata-enriched context for the LLM.
        """
        if self.orchestrator is None:
            return ""
        query_text = query if isinstance(query, str) else str(query)
        if not query_text:
            return ""

        # Extract category marker
        category = None
        cat_match = re.match(r'^\[([A-Z]+)\]\s*', query_text)
        if cat_match:
            category = cat_match.group(1)
            query_text = query_text[cat_match.end():]

        # VS/VR: use image-specific retrieval
        if category in ("VS", "VR"):
            return self._recall_visual(query_text, category)

        # Category-aware top_k
        effective_top_k = self._get_dynamic_top_k(query_text, category)

        # FR list questions need broader retrieval
        is_list_question = False
        if category == "FR":
            q_lower = query_text.lower()
            if any(kw in q_lower for kw in ("list all", "please list", "what are all", "name all", "what were all", "how many")):
                is_list_question = True
                effective_top_k = max(effective_top_k, 40)

        try:
            result = self.orchestrator.query(query_text, top_k=effective_top_k, auto_expand=False)
        except Exception:
            return ""

        items = getattr(result, "items", []) or []

        # BM25 augmentation (set-union merge)
        bm25_extra = 15 if (category in ("FR",) and is_list_question) or category == "KR" else 10
        try:
            existing_ids = {item.get("id") for item in items}
            bm25_ids = self._bm25_search(query_text, top_k=effective_top_k)
            new_bm25_ids = [mid for mid in bm25_ids if mid not in existing_ids]
            if new_bm25_ids:
                bm25_expanded = self.orchestrator.expand(new_bm25_ids[:bm25_extra])
                for bm25_item in bm25_expanded.items:
                    items.append(bm25_item)
        except Exception:
            pass

        if not items:
            return ""

        # TR/CD: chronological sort
        if category in ("TR", "CD"):
            def _sort_key(item):
                tags = item.get("tags") or []
                if not tags:
                    meta = item.get("metadata")
                    if isinstance(meta, dict):
                        tags = meta.get("tags") or []
                session = ""
                turn = ""
                for tag in tags:
                    if tag.startswith("session_id:"):
                        session = tag[11:]
                    elif tag.startswith("dialogue_id:"):
                        turn = tag[12:]
                s_num = 0
                t_num = 0
                try:
                    s_num = int(session.lstrip("D")) if session.startswith("D") else int(session)
                except (ValueError, AttributeError):
                    pass
                try:
                    t_num = int(turn)
                except (ValueError, AttributeError):
                    pass
                return (s_num, t_num)
            items.sort(key=_sort_key)

        formatted = self._format_items(items)

        # TTL: also search image catalog for matching images
        if category == "TTL" and "image_caption:" in query_text.lower():
            try:
                caption_part = query_text.lower().split("image_caption:", 1)[1].strip()
                image_results = self._image_search(caption_part, return_all=False, top_k=10)
                if image_results:
                    img_lines = ["\n\n=== Related Images from Conversation ==="]
                    seen_img_ids = set()
                    for idx, img in enumerate(image_results, start=1):
                        img_id = img.get("image_id", "")
                        if img_id in seen_img_ids:
                            continue
                        seen_img_ids.add(img_id)
                        cap = img.get("caption", "")
                        session = img.get("session_id", "")
                        full_text = img.get("full_text", "")
                        entry = f"[IMG-{idx}] {img_id}"
                        if session:
                            entry += f" (SESSION:{session})"
                        entry += f"\nCaption: {cap}"
                        if full_text and full_text != cap:
                            entry += f"\nConversation context: {full_text[:500]}"
                        img_lines.append(entry)
                    formatted += "\n".join(img_lines)
            except Exception:
                pass

        return formatted

    def _recall_visual(self, query_text: str, category: str) -> str:
        """Retrieval for VS (Visual Search) and VR (Visual Reasoning)."""
        lines = []

        # Standard FAISS + BM25 retrieval
        try:
            result = self.orchestrator.query(query_text, top_k=15, auto_expand=False)
            items = getattr(result, "items", []) or []
        except Exception:
            items = []

        try:
            existing_ids = {item.get("id") for item in items}
            bm25_ids = self._bm25_search(query_text, top_k=15)
            new_bm25_ids = [mid for mid in bm25_ids if mid not in existing_ids]
            if new_bm25_ids:
                bm25_expanded = self.orchestrator.expand(new_bm25_ids[:10])
                for bm25_item in bm25_expanded.items:
                    items.append(bm25_item)
        except Exception:
            pass

        if items:
            standard_text = self._format_items(items)
            if standard_text:
                lines.append("=== Conversation Context ===")
                lines.append(standard_text)

        # Image-specific catalog search
        image_results = self._image_search(query_text, return_all=(category == "VR"), top_k=30)
        if image_results:
            lines.append("")
            lines.append("=== Image Catalog ===")
            seen_img_ids = set()
            for idx, img in enumerate(image_results, start=1):
                img_id = img.get("image_id", "")
                if img_id in seen_img_ids:
                    continue
                seen_img_ids.add(img_id)
                caption = img.get("caption", "")
                session = img.get("session_id", "")
                full_text = img.get("full_text", "")
                entry = f"[IMG-{idx}] {img_id}"
                if session:
                    entry += f" (SESSION:{session})"
                if caption:
                    entry += f"\nCaption: {caption}"
                if full_text and full_text != caption:
                    entry += f"\nContext: {full_text[:300]}"
                lines.append(entry)

        if not lines:
            return ""
        return "\n\n".join(lines)

    def _format_items(self, items: List[Dict[str, Any]]) -> str:
        """Format retrieved items with metadata headers."""
        try:
            mau_ids = [item["id"] for item in items if item.get("id") and not item.get("details")]
            if mau_ids:
                expanded = self.orchestrator.expand(mau_ids)
                expanded_map = {item["id"]: item for item in expanded.items}
                for i, item in enumerate(items):
                    if item.get("id") in expanded_map:
                        items[i] = {**item, **expanded_map[item["id"]]}
        except Exception:
            pass

        lines = []
        for idx, item in enumerate(items, start=1):
            details = item.get("details")
            if isinstance(details, dict):
                text_content = details.get("full_text", "")
            elif isinstance(details, str):
                text_content = details
            else:
                text_content = ""
            if not text_content:
                text_content = item.get("summary") or item.get("text") or ""
            if not text_content:
                continue

            tags = item.get("tags") or []
            if not tags:
                meta = item.get("metadata")
                if isinstance(meta, dict):
                    tags = meta.get("tags") or []
            meta_parts = []
            for tag in tags:
                if tag.startswith("timestamp:"):
                    meta_parts.append(f"DATE:{tag[10:]}")
                elif tag.startswith("session_id:"):
                    meta_parts.append(f"SESSION:{tag[11:]}")
                elif tag.startswith("dialogue_id:"):
                    meta_parts.append(f"TURN:{tag[12:]}")
                elif tag.startswith("image_id:"):
                    meta_parts.append(f"IMG:{tag[9:]}")

            ts = item.get("timestamp")
            if ts and not any(p.startswith("DATE:") for p in meta_parts):
                try:
                    from datetime import datetime
                    meta_parts.insert(0, f"DATE:{datetime.fromtimestamp(ts).strftime('%Y-%m-%d')}")
                except (ValueError, TypeError, OSError):
                    pass

            header = f"[{idx}]"
            if meta_parts:
                header += " " + " | ".join(meta_parts)
            lines.append(f"{header}\n{text_content}")

        return "\n\n".join(lines)
