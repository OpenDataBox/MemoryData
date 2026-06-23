import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

from openai import AsyncOpenAI
from utils.locomo_utils import strip_locomo_metadata


ANSWER_SYSTEM_PROMPT = """You are a helpful expert assistant answering user questions from retrieved conversation memories.
Answer briefly and precisely using only the provided context. If the context is insufficient, abstain.

When interpreting memories, use the timestamp to determine when an event happened, not when someone talked about it.
"""

_SPEAKER_ALIASES = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System",
    "customer": "Customer",
    "recommender": "Recommender",
    "recommender system": "Recommender System",
    "speaker a": "Speaker A",
    "speaker b": "Speaker B",
    "human": "Human",
    "bot": "Bot",
    "agent": "Agent",
}
_SPEAKER_PATTERN = "|".join(
    sorted((re.escape(label) for label in _SPEAKER_ALIASES), key=len, reverse=True)
)
_BRACKETED_TURN_RE = re.compile(
    rf"^\s*[<\[](?P<speaker>{_SPEAKER_PATTERN})[>\]]\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)
_COLON_TURN_RE = re.compile(
    rf"^\s*(?P<speaker>{_SPEAKER_PATTERN})\s*[:：-]\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)


def get_retrieval_query(query: str) -> str:
    start_marker = "These are the events"
    end_markers = [
        "Your task is to",
        "Below is a list of possible subsequent events:",
    ]
    end_indices = [idx for marker in end_markers if (idx := query.find(marker)) != -1]
    if end_indices:
        end_idx = min(end_indices)
        start_idx = query.rfind(start_marker, 0, end_idx)
        if start_idx != -1:
            return query[start_idx:end_idx].strip()

    match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
    if match:
        return "".join(match.groups())

    match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
    if match:
        return "".join(match.groups())

    return query


def _should_disable_qwen3_thinking(model_name):
    return "qwen3" in str(model_name or "").strip().lower()


def _extract_openai_message_text(message):
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str) and item.strip():
                text_parts.append(item.strip())
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if isinstance(text_value, str) and text_value.strip():
                    text_parts.append(text_value.strip())
        if text_parts:
            return "\n".join(text_parts)
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def _is_multiple_choice_question(question: str) -> bool:
    text = str(question or "")
    return all(option in text for option in ("\nA.", "\nB.", "\nC.", "\nD."))


def _match_dialogue_turn(line: str):
    for pattern in (_BRACKETED_TURN_RE, _COLON_TURN_RE):
        match = pattern.match(line)
        if match:
            return match
    return None


def extract_dialogue_episode_bodies(text: str):
    turns = []
    current_speaker = None
    current_lines = []

    def flush_current():
        if not current_speaker:
            return
        turn_text = " ".join(part.strip() for part in current_lines if part.strip()).strip()
        if turn_text:
            turns.append(f"\n{current_speaker}: {turn_text}")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _match_dialogue_turn(line)
        if match:
            flush_current()
            current_speaker = _SPEAKER_ALIASES[match.group("speaker").strip().lower()]
            current_lines = [match.group("text").strip()]
            continue

        if current_speaker:
            current_lines.append(line)

    flush_current()
    return turns if len(turns) >= 2 else []


def split_text_episode_bodies(text: str, max_chars: int):
    raw_text = (text or "").strip()
    if not raw_text or len(raw_text) <= max_chars:
        return [raw_text] if raw_text else []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", raw_text) if part.strip()]
    units = paragraphs if len(paragraphs) > 1 else [
        part.strip() for part in re.split(r"(?<=[.!?])\s+", raw_text) if part.strip()
    ]
    if not units:
        units = [raw_text]

    episodes = []
    current = ""

    def flush_current():
        nonlocal current
        if current.strip():
            episodes.append(current.strip())
        current = ""

    for unit in units:
        if len(unit) > max_chars:
            flush_current()
            start = 0
            while start < len(unit):
                episodes.append(unit[start : start + max_chars].strip())
                start += max_chars
            continue

        separator = "\n\n" if paragraphs and len(paragraphs) > 1 else " "
        candidate = unit if not current else f"{current}{separator}{unit}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush_current()
            current = unit

    flush_current()
    return [episode for episode in episodes if episode]


def extract_retrieved_facts(results):
    retrieved = {"Edges": [], "Nodes": [], "Episodes": [], "Communities": []}

    try:
        for edge in getattr(results, "edges", [])[:5]:
            fact = f"{getattr(edge, 'name', 'Edge')}: {getattr(edge, 'fact', '')}"
            time_info = []
            if getattr(edge, "expired_at", None):
                time_info.append(f" (Expired at: {edge.expired_at})")
            if getattr(edge, "valid_at", None):
                time_info.append(f" (Valid from: {edge.valid_at})")
            if getattr(edge, "invalid_at", None):
                time_info.append(f" (Valid until: {edge.invalid_at})")
            retrieved["Edges"].append(fact + "".join(time_info))
    except Exception:
        pass

    try:
        for node in getattr(results, "nodes", [])[:5]:
            retrieved["Nodes"].append(
                f"{getattr(node, 'name', 'Node')}: {getattr(node, 'summary', '')}"
            )
    except Exception:
        pass

    try:
        for episode in getattr(results, "episodes", []):
            retrieved["Episodes"].append(
                f"{getattr(episode, 'source_description', 'Episode')}: {strip_locomo_metadata(getattr(episode, 'content', ''))}"
            )
    except Exception:
        pass

    try:
        for community in getattr(results, "communities", [])[:3]:
            retrieved["Communities"].append(
                f"{getattr(community, 'name', 'Community')}: {getattr(community, 'summary', '')}"
            )
    except Exception:
        pass

    return retrieved


def build_answer_context(retrieved):
    sections = []
    for section_name in ("Edges", "Nodes", "Episodes", "Communities"):
        values = retrieved.get(section_name, [])
        if values:
            sections.append(f"[{section_name}]\n" + "\n".join(values))
    return "\n\n".join(sections)


class GraphitiLocalMemory:
    def __init__(
        self,
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_model,
        llm_small_model,
        llm_api_key,
        llm_base_url,
        llm_temperature,
        llm_max_tokens,
        episode_max_chars,
        embedding_model_name,
        embedding_api_key,
        embedding_base_url,
        embedding_dim,
        answer_model,
        answer_api_key,
        answer_base_url,
        answer_temperature=0.0,
        answer_max_tokens=200,
    ):
        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.driver import GraphDriver
            from graphiti_core.driver.neo4j_driver import Neo4jDriver
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
            from graphiti_core.nodes import EpisodeType
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:
            raise RuntimeError(
                "zep_local requires 'graphiti-core' and its Neo4j dependencies. "
                "Install the updated requirements before running this method."
            ) from exc

        class _GraphitiNeo4jDriver(Neo4jDriver):
            """Neo4j driver without Graphiti's background index-build task."""

            def __init__(self, uri: str, user: str | None, password: str | None, database: str = "neo4j"):
                GraphDriver.__init__(self)
                self.client = AsyncGraphDatabase.driver(
                    uri=uri,
                    auth=(user or "", password or ""),
                )
                self._database = database
                self.aoss_client = None

            async def build_indices_and_constraints(self, delete_existing: bool = False):
                if delete_existing:
                    await self.delete_all_indexes()

                index_queries = get_range_indices(self.provider) + get_fulltext_indices(self.provider)
                for query in index_queries:
                    await self.execute_query(query)

        class _RobustOpenAIGenericClient(OpenAIGenericClient):
            """OpenAI-compatible Graphiti client with more tolerant JSON parsing."""

            _DEBUG_LOG_PATH = Path("/tmp/zep_local_graphiti_json_failures.log")

            @classmethod
            def _dump_invalid_json(
                cls,
                *,
                stage: str,
                response_model=None,
                payload: str = "",
                error: Exception | None = None,
            ):
                try:
                    entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "stage": stage,
                        "response_model": getattr(response_model, "__name__", None),
                        "error": repr(error) if error is not None else None,
                        "payload": payload,
                    }
                    with cls._DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            @staticmethod
            def _normalize_relation_type(value):
                relation = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "RELATES_TO")).strip("_")
                relation = re.sub(r"_+", "_", relation)
                return relation.upper() or "RELATES_TO"

            @staticmethod
            def _coerce_int(value):
                if isinstance(value, bool):
                    return None
                if isinstance(value, int):
                    return value
                if isinstance(value, float):
                    return int(value)
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.isdigit():
                        return int(stripped)
                return None

            @classmethod
            def _normalize_entity_items(cls, items):
                normalized = []
                seen = set()
                for item in items or []:
                    if isinstance(item, str):
                        name = item.strip()
                        lowered = name.lower()
                        if name and lowered not in seen:
                            seen.add(lowered)
                            normalized.append({"name": name, "entity_type_id": 0})
                        continue
                    if not isinstance(item, dict):
                        continue
                    name = (
                        item.get("name")
                        or item.get("entity")
                        or item.get("text")
                        or item.get("label")
                        or item.get("entity_name")
                    )
                    if not isinstance(name, str) or not name.strip():
                        continue
                    entity_type_id = cls._coerce_int(
                        item.get("entity_type_id")
                        or item.get("type_id")
                        or item.get("entityTypeId")
                        or 0
                    )
                    lowered = name.strip().lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    normalized.append(
                        {
                            "name": name.strip(),
                            "entity_type_id": 0 if entity_type_id is None else entity_type_id,
                        }
                    )
                    if len(normalized) >= 16:
                        break
                return {"extracted_entities": normalized}

            @classmethod
            def _normalize_edge_items(cls, items):
                normalized = []
                seen = set()
                for item in items or []:
                    if not isinstance(item, dict):
                        continue
                    source_entity_id = cls._coerce_int(
                        item.get("source_entity_id")
                        or item.get("source_id")
                        or item.get("source")
                        or item.get("source_node_id")
                    )
                    target_entity_id = cls._coerce_int(
                        item.get("target_entity_id")
                        or item.get("target_id")
                        or item.get("target")
                        or item.get("target_node_id")
                    )
                    fact = (
                        item.get("fact")
                        or item.get("description")
                        or item.get("sentence")
                        or item.get("text")
                    )
                    if (
                        source_entity_id is None
                        or target_entity_id is None
                        or source_entity_id == target_entity_id
                        or not isinstance(fact, str)
                        or not fact.strip()
                    ):
                        continue
                    relation_type = cls._normalize_relation_type(
                        item.get("relation_type")
                        or item.get("predicate")
                        or item.get("relation")
                        or item.get("edge_type")
                        or item.get("type")
                    )
                    dedupe_key = (
                        relation_type,
                        source_entity_id,
                        target_entity_id,
                        fact.strip().lower(),
                    )
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    normalized.append(
                        {
                            "relation_type": relation_type,
                            "source_entity_id": source_entity_id,
                            "target_entity_id": target_entity_id,
                            "fact": fact.strip(),
                            "valid_at": item.get("valid_at") or item.get("start_time"),
                            "invalid_at": item.get("invalid_at") or item.get("end_time"),
                        }
                    )
                    if len(normalized) >= 24:
                        break
                return {"edges": normalized}

            @classmethod
            def _normalize_string_list(cls, parsed, key, alternative_keys):
                if isinstance(parsed, list):
                    values = [str(item).strip() for item in parsed if str(item).strip()]
                    return {key: values}
                if isinstance(parsed, dict):
                    if isinstance(parsed.get(key), list):
                        values = [str(item).strip() for item in parsed[key] if str(item).strip()]
                        return {key: values}
                    for candidate_key in alternative_keys:
                        if isinstance(parsed.get(candidate_key), list):
                            values = [
                                str(item).strip() for item in parsed[candidate_key] if str(item).strip()
                            ]
                            return {key: values}
                return {key: []}

            @classmethod
            def _normalize_node_resolutions(cls, parsed):
                if isinstance(parsed, dict):
                    items = parsed.get("entity_resolutions", [])
                elif isinstance(parsed, list):
                    items = parsed
                else:
                    items = []

                normalized = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_id = cls._coerce_int(item.get("id"))
                    if item_id is None:
                        continue
                    duplicates = []
                    for duplicate in item.get("duplicates", []):
                        duplicate_idx = cls._coerce_int(duplicate)
                        if duplicate_idx is not None:
                            duplicates.append(duplicate_idx)
                    duplicate_idx = cls._coerce_int(item.get("duplicate_idx"))
                    normalized.append(
                        {
                            "id": item_id,
                            "name": str(item.get("name") or "").strip(),
                            "duplicate_idx": -1 if duplicate_idx is None else duplicate_idx,
                            "duplicates": duplicates,
                        }
                    )
                return {"entity_resolutions": normalized}

            @classmethod
            def _normalize_payload_for_schema(cls, parsed, response_model):
                if response_model is None:
                    return parsed

                model_name = getattr(response_model, "__name__", "")
                if model_name == "ExtractedEntities":
                    if isinstance(parsed, dict) and "extracted_entities" in parsed:
                        return cls._normalize_entity_items(parsed.get("extracted_entities"))
                    if isinstance(parsed, dict):
                        for candidate_key in (
                            "entities",
                            "entity_nodes",
                            "items",
                            "results",
                            "extractedNodes",
                        ):
                            if candidate_key in parsed:
                                return cls._normalize_entity_items(parsed.get(candidate_key))
                        if any(key in parsed for key in ("name", "entity", "text", "label")):
                            return cls._normalize_entity_items([parsed])
                    return cls._normalize_entity_items(parsed if isinstance(parsed, list) else [])

                if model_name == "ExtractedEdges":
                    if isinstance(parsed, dict) and "edges" in parsed:
                        return cls._normalize_edge_items(parsed.get("edges"))
                    if isinstance(parsed, dict):
                        for candidate_key in (
                            "facts",
                            "relationships",
                            "relations",
                            "triples",
                            "items",
                            "extracted_edges",
                        ):
                            if candidate_key in parsed:
                                return cls._normalize_edge_items(parsed.get(candidate_key))
                        if any(
                            key in parsed
                            for key in (
                                "relation_type",
                                "predicate",
                                "relation",
                                "source_entity_id",
                                "target_entity_id",
                                "fact",
                            )
                        ):
                            return cls._normalize_edge_items([parsed])
                    return cls._normalize_edge_items(parsed if isinstance(parsed, list) else [])

                if model_name == "MissingFacts":
                    return cls._normalize_string_list(
                        parsed,
                        "missing_facts",
                        ("facts", "missed_facts", "extracted_facts", "items"),
                    )

                if model_name == "MissedEntities":
                    return cls._normalize_string_list(
                        parsed,
                        "missed_entities",
                        ("entities", "missing_entities", "items"),
                    )

                if model_name == "NodeResolutions":
                    return cls._normalize_node_resolutions(parsed)

                return parsed

            @classmethod
            def _recover_summary_text(cls, payload: str):
                text = (payload or "").strip()
                match = re.search(r'"summary"\s*:\s*"(.*)$', text, re.DOTALL)
                if not match:
                    return "Summary unavailable."
                summary = match.group(1)
                summary = re.sub(r'\\(["\\/bfnrt])', r"\1", summary)
                summary = summary.replace("\\n", " ").replace("\\t", " ")
                summary = re.sub(r"\s+", " ", summary).strip()
                if not summary:
                    return "Summary unavailable."
                return summary[:1200].strip()

            @classmethod
            def _fallback_payload_for_schema(cls, response_model, payload: str = ""):
                model_name = getattr(response_model, "__name__", "")
                if model_name == "ExtractedEntities":
                    return {"extracted_entities": []}
                if model_name == "ExtractedEdges":
                    return {"edges": []}
                if model_name == "MissingFacts":
                    return {"missing_facts": []}
                if model_name == "MissedEntities":
                    return {"missed_entities": []}
                if model_name == "EntitySummary":
                    return {"summary": cls._recover_summary_text(payload)}
                if model_name == "NodeResolutions":
                    return {"entity_resolutions": []}
                return None

            @staticmethod
            def _parse_json_payload(payload: str):
                def close_json_fragment(fragment: str):
                    candidate = re.sub(r",\s*$", "", fragment.strip())
                    stack = []
                    in_string = False
                    escape = False
                    opener_to_closer = {"{": "}", "[": "]"}
                    closer_to_opener = {"}": "{", "]": "["}

                    for char in candidate:
                        if in_string:
                            if escape:
                                escape = False
                            elif char == "\\":
                                escape = True
                            elif char == '"':
                                in_string = False
                            continue

                        if char == '"':
                            in_string = True
                        elif char in opener_to_closer:
                            stack.append(char)
                        elif char in closer_to_opener and stack and stack[-1] == closer_to_opener[char]:
                            stack.pop()

                    if in_string:
                        candidate += '"'
                    while stack:
                        candidate += opener_to_closer[stack.pop()]
                    return re.sub(r",\s*([}\]])", r"\1", candidate)

                cleaned = payload.strip()
                fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
                if fence_match:
                    cleaned = fence_match.group(1).strip()

                candidates = [cleaned]
                for opener, closer in (("{", "}"), ("[", "]")):
                    start = cleaned.find(opener)
                    end = cleaned.rfind(closer)
                    if start != -1 and end != -1 and end > start:
                        candidates.append(cleaned[start : end + 1])

                decoder = json.JSONDecoder()
                last_error = None
                seen = set()

                for candidate in candidates:
                    for variant in (
                        candidate,
                        re.sub(r",\s*([}\]])", r"\1", candidate),
                    ):
                        if not variant or variant in seen:
                            continue
                        seen.add(variant)
                        try:
                            return json.loads(variant)
                        except json.JSONDecodeError as exc:
                            last_error = exc

                        try:
                            parsed, _ = decoder.raw_decode(variant)
                            return parsed
                        except json.JSONDecodeError as exc:
                            last_error = exc

                        for parser in ("dirtyjson", "demjson3"):
                            try:
                                if parser == "dirtyjson":
                                    import dirtyjson

                                    return dirtyjson.loads(variant)
                                import demjson3

                                return demjson3.decode(variant)
                            except Exception as exc:
                                last_error = exc
                        balanced_variant = close_json_fragment(variant)
                        if balanced_variant != variant:
                            try:
                                return json.loads(balanced_variant)
                            except json.JSONDecodeError as exc:
                                last_error = exc
                            try:
                                parsed, _ = decoder.raw_decode(balanced_variant)
                                return parsed
                            except json.JSONDecodeError as exc:
                                last_error = exc
                            for parser in ("dirtyjson", "demjson3"):
                                try:
                                    if parser == "dirtyjson":
                                        import dirtyjson

                                        return dirtyjson.loads(balanced_variant)
                                    import demjson3

                                    return demjson3.decode(balanced_variant)
                                except Exception as exc:
                                    last_error = exc

                        for closer in ("}", "]"):
                            end = variant.rfind(closer)
                            while end != -1:
                                truncated = re.sub(r",\s*([}\]])", r"\1", variant[: end + 1])
                                try:
                                    return json.loads(truncated)
                                except json.JSONDecodeError as exc:
                                    last_error = exc
                                try:
                                    parsed, _ = decoder.raw_decode(truncated)
                                    return parsed
                                except json.JSONDecodeError as exc:
                                    last_error = exc
                                for parser in ("dirtyjson", "demjson3"):
                                    try:
                                        if parser == "dirtyjson":
                                            import dirtyjson

                                            return dirtyjson.loads(truncated)
                                        import demjson3

                                        return demjson3.decode(truncated)
                                    except Exception as exc:
                                        last_error = exc
                                balanced = close_json_fragment(truncated)
                                try:
                                    return json.loads(balanced)
                                except json.JSONDecodeError as exc:
                                    last_error = exc
                                try:
                                    parsed, _ = decoder.raw_decode(balanced)
                                    return parsed
                                except json.JSONDecodeError as exc:
                                    last_error = exc
                                for parser in ("dirtyjson", "demjson3"):
                                    try:
                                        if parser == "dirtyjson":
                                            import dirtyjson

                                            return dirtyjson.loads(balanced)
                                        import demjson3

                                        return demjson3.decode(balanced)
                                    except Exception as exc:
                                        last_error = exc
                                end = variant.rfind(closer, 0, end)

                if last_error is not None:
                    raise last_error
                raise json.JSONDecodeError("Empty JSON payload", payload, 0)

            async def _generate_response(
                self,
                messages,
                response_model=None,
                max_tokens=8192,
                model_size=None,
            ):
                openai_messages = []
                for message in messages:
                    content = self._clean_input(message.content)
                    if message.role == "user":
                        openai_messages.append({"role": "user", "content": content})
                    elif message.role == "system":
                        openai_messages.append({"role": "system", "content": content})

                response_format = {"type": "json_object"}
                if response_model is not None:
                    schema_json = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
                    model_name = getattr(response_model, "__name__", "")
                    extra_constraints = ""
                    if model_name == "ExtractedEntities":
                        extra_constraints = (
                            " Keep only the most salient unique entities. "
                            "Do not emit duplicates or pronouns. Prefer no more than 16 entities."
                        )
                    elif model_name == "ExtractedEdges":
                        extra_constraints = (
                            " Keep only the most salient unique facts. "
                            "Do not emit duplicates. Prefer no more than 24 edges."
                        )
                    openai_messages.insert(
                        0,
                        {
                            "role": "system",
                            "content": (
                                "Return only one valid JSON object matching this schema exactly. "
                                f"Schema: {schema_json}.{extra_constraints}"
                            ),
                        },
                    )

                request_kwargs = {
                    "model": self.model or "gpt-4.1-mini",
                    "messages": openai_messages,
                    "temperature": self.temperature,
                    "max_tokens": min(max_tokens or self.max_tokens, self.max_tokens),
                    "response_format": response_format,
                }
                if _should_disable_qwen3_thinking(self.model):
                    request_kwargs["extra_body"] = {
                        "enable_thinking": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }

                last_exception = None
                response = None
                for attempt in range(3):
                    try:
                        response = await self.client.chat.completions.create(**request_kwargs)
                        break
                    except Exception as exc:
                        last_exception = exc
                        status_code = getattr(exc, "status_code", None)
                        if status_code and int(status_code) >= 500 and attempt < 2:
                            await asyncio.sleep(1 + attempt)
                            continue
                        raise
                if response is None:
                    raise last_exception
                result = _extract_openai_message_text(response.choices[0].message)
                parsed = None
                try:
                    parsed = self._parse_json_payload(result)
                    parsed = self._normalize_payload_for_schema(parsed, response_model)
                    if response_model is None:
                        return parsed
                    response_model.model_validate(parsed)
                    return parsed
                except Exception as exc:
                    self._dump_invalid_json(
                        stage="initial_parse_failed",
                        response_model=response_model,
                        payload=result,
                        error=exc,
                    )
                    repair_prompt = (
                        "Repair or rewrite the JSON below so it becomes a valid JSON object that matches "
                        "the target schema exactly. Return only the corrected JSON object."
                    )
                    if response_model is not None:
                        repair_prompt += (
                            "\n\nTarget schema:\n"
                            f"{json.dumps(response_model.model_json_schema(), ensure_ascii=False)}"
                        )

                    repair_source = (
                        json.dumps(parsed, ensure_ascii=False)
                        if parsed is not None
                        else result
                    )
                    repair_kwargs = {
                        "model": self.model or "gpt-4.1-mini",
                        "messages": [
                            {"role": "system", "content": repair_prompt},
                            {"role": "user", "content": repair_source},
                        ],
                        "temperature": 0.0,
                        "max_tokens": self.max_tokens,
                        "response_format": {"type": "json_object"},
                    }
                    if _should_disable_qwen3_thinking(self.model):
                        repair_kwargs["extra_body"] = {
                            "enable_thinking": False,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                    repair_response = await self.client.chat.completions.create(**repair_kwargs)
                    repaired = _extract_openai_message_text(repair_response.choices[0].message)
                    try:
                        repaired_parsed = self._parse_json_payload(repaired)
                        repaired_parsed = self._normalize_payload_for_schema(
                            repaired_parsed, response_model
                        )
                        if response_model is not None:
                            response_model.model_validate(repaired_parsed)
                        return repaired_parsed
                    except Exception as repair_exc:
                        self._dump_invalid_json(
                            stage="repair_parse_failed",
                            response_model=response_model,
                            payload=repaired,
                            error=repair_exc,
                        )
                        fallback = self._fallback_payload_for_schema(response_model, repaired or result)
                        if fallback is not None:
                            response_model.model_validate(fallback)
                            return fallback
                        raise

        llm_config = LLMConfig(
            api_key=llm_api_key,
            model=llm_model,
            small_model=llm_small_model or llm_model,
            base_url=llm_base_url,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
        )
        llm_client = _RobustOpenAIGenericClient(config=llm_config, max_tokens=llm_max_tokens)
        self._episode_message_type = EpisodeType.message
        self._episode_text_type = EpisodeType.text
        self.graphiti = Graphiti(
            llm_client=llm_client,
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    embedding_model=embedding_model_name,
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                    embedding_dim=embedding_dim,
                )
            ),
            cross_encoder=OpenAIRerankerClient(config=llm_config),
            graph_driver=_GraphitiNeo4jDriver(
                neo4j_uri,
                neo4j_user,
                neo4j_password,
            ),
        )
        self.answer_client = AsyncOpenAI(
            api_key=answer_api_key,
            base_url=answer_base_url,
        )
        self.answer_model = answer_model
        self.answer_temperature = answer_temperature
        self.answer_max_tokens = answer_max_tokens
        self.episode_max_chars = max(512, int(episode_max_chars or 3000))
        # The benchmark pipeline currently feeds plain text chunks rather than the
        # source repo's dialogue/session objects, so we do not have reliable
        # per-chunk event timestamps. Use a deterministic monotonic clock instead
        # of wall-clock "now" to preserve insertion order without injecting false
        # real-world dates into temporal reasoning.
        self._synthetic_epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
        self._namespace_offsets = {}
        self._loop = asyncio.new_event_loop()
        self._run_sync(self.graphiti.driver.build_indices_and_constraints())

    def extract_episode_bodies_from_chunk(self, content):
        raw_content = (content or "").strip()
        if not raw_content:
            return []

        dialogue_turns = extract_dialogue_episode_bodies(raw_content)
        if dialogue_turns:
            return dialogue_turns
        return split_text_episode_bodies(raw_content, self.episode_max_chars)

    def _get_episode_type(self, content):
        return (
            self._episode_message_type
            if extract_dialogue_episode_bodies(content or "")
            else self._episode_text_type
        )

    async def add_memory(
        self,
        namespace,
        content,
        name,
        source_description,
        reference_time=None,
    ):
        if reference_time is None:
            offset = self._namespace_offsets.get(namespace, 0)
            reference_time = self._synthetic_epoch + timedelta(seconds=offset)
            self._namespace_offsets[namespace] = offset + 1
        await self.graphiti.add_episode(
            name=name,
            episode_body=content,
            source=self._get_episode_type(content),
            source_description=source_description,
            reference_time=reference_time,
            group_id=namespace,
        )

    async def build_communities(self, namespace):
        await self.graphiti.build_communities(group_ids=[namespace])

    async def search(self, question, namespace):
        retrieval_query = get_retrieval_query(question)
        return await self.graphiti.search_(retrieval_query, group_ids=[namespace])

    async def answer_query(self, question, namespace, results=None):
        if results is None:
            results = await self.search(question, namespace)
        retrieved = extract_retrieved_facts(results)
        context = build_answer_context(retrieved)

        if not context.strip():
            return "Insufficient context to answer.", ""

        system_prompt = ANSWER_SYSTEM_PROMPT
        user_suffix = "Answer briefly and directly based only on the context."
        if _is_multiple_choice_question(question):
            system_prompt += "\nIf the question is multiple-choice, reply with exactly one uppercase letter: A, B, C, or D. Do not explain your answer."
            user_suffix = "Reply with exactly one uppercase letter: A, B, C, or D."

        request_kwargs = {
            "model": self.answer_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "# CONTEXT\n"
                        f"{context}\n\n"
                        "# QUESTION\n"
                        f"{question}\n\n"
                        f"{user_suffix}"
                    ),
                },
            ],
            "max_tokens": self.answer_max_tokens,
            "temperature": self.answer_temperature,
        }
        if _should_disable_qwen3_thinking(self.answer_model):
            request_kwargs["extra_body"] = {
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        last_exception = None
        response = None
        for attempt in range(3):
            try:
                response = await self.answer_client.chat.completions.create(**request_kwargs)
                break
            except Exception as exc:
                last_exception = exc
                status_code = getattr(exc, "status_code", None)
                if status_code and int(status_code) >= 500 and attempt < 2:
                    await asyncio.sleep(1 + attempt)
                    continue
                raise
        if response is None:
            raise last_exception

        answer = _extract_openai_message_text(response.choices[0].message) or "No response generated"
        return answer.strip(), context

    async def close(self):
        await self.graphiti.close()

    def _run_sync(self, coroutine):
        return self._loop.run_until_complete(coroutine)

    def add_memory_sync(self, **kwargs):
        return self._run_sync(self.add_memory(**kwargs))

    def build_communities_sync(self, namespace):
        return self._run_sync(self.build_communities(namespace))

    def answer_query_sync(self, question, namespace, results=None):
        return self._run_sync(self.answer_query(question, namespace, results=results))

    def search_sync(self, question, namespace):
        return self._run_sync(self.search(question, namespace))

    def close_sync(self):
        try:
            return self._run_sync(self.close())
        finally:
            self._loop.close()
