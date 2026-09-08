# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingParameterType=false, reportUnusedCallResult=false, reportUnusedImport=false, reportDeprecated=false
import json
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Optional, Type, cast
from uuid import uuid4

from .types import (
    ConsolidationRun,
    CrossObservation,
    EventKind,
    MemoryLink,
    ObservationType,
    RedactionLevel,
    SessionEvent,
    SessionRecord,
    SessionStatus,
    SessionSummary,
)


logger = logging.getLogger(__name__)


class SQLiteStorage:
    """Manages SQLite database for cross-session memory."""

    db_path: Path
    conn: sqlite3.Connection

    def __init__(self, db_path: str = "~/.simplemem-cross/cross_memory.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._run_migrations()

    def __enter__(self) -> "SQLiteStorage":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            logger.exception("Failed to close SQLite connection")

    def _configure_connection(self) -> None:
        try:
            _ = self.conn.execute("PRAGMA journal_mode=WAL")
            _ = self.conn.execute("PRAGMA foreign_keys=ON")
            _ = self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            logger.exception("Failed to configure SQLite connection")
            raise

    def _run_migrations(self) -> None:
        schema_statements = [
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                content_session_id TEXT UNIQUE NOT NULL,
                memory_session_id TEXT UNIQUE NOT NULL,
                project TEXT NOT NULL,
                user_prompt TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT CHECK(status IN ('active', 'completed', 'failed')) DEFAULT 'active',
                metadata_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS session_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                kind TEXT CHECK(kind IN ('message', 'tool_use', 'file_change', 'note', 'system')) NOT NULL,
                title TEXT,
                payload_json TEXT,
                redaction_level TEXT DEFAULT 'none',
                FOREIGN KEY(memory_session_id) REFERENCES sessions(memory_session_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS observations (
                obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                type TEXT CHECK(type IN ('decision', 'bugfix', 'feature', 'refactor', 'discovery', 'change')) NOT NULL,
                title TEXT NOT NULL,
                subtitle TEXT,
                facts_json TEXT,
                narrative TEXT,
                concepts_json TEXT,
                files_json TEXT,
                vector_ref TEXT,
                FOREIGN KEY(memory_session_id) REFERENCES sessions(memory_session_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS session_summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                request TEXT,
                investigated TEXT,
                learned TEXT,
                completed TEXT,
                next_steps TEXT,
                vector_ref TEXT,
                FOREIGN KEY(memory_session_id) REFERENCES sessions(memory_session_id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_links (
                link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_entry_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                score REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS consolidation_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                timestamp TEXT NOT NULL,
                policy_json TEXT,
                stats_json TEXT
            )
            """,
        ]
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_content_id ON sessions(content_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_memory_id ON sessions(memory_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)",
            "CREATE INDEX IF NOT EXISTS idx_events_session ON session_events(memory_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_events_kind ON session_events(kind)",
            "CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(memory_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(type)",
            "CREATE INDEX IF NOT EXISTS idx_summaries_session ON session_summaries(memory_session_id)",
            "CREATE INDEX IF NOT EXISTS idx_links_entry ON memory_links(memory_entry_id)",
            "CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_kind, source_id)",
        ]
        try:
            cursor = self.conn.cursor()
            for statement in schema_statements:
                _ = cursor.execute(statement)
            for statement in index_statements:
                _ = cursor.execute(statement)
            self.conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to run SQLite migrations")
            self.conn.rollback()
            raise

    def create_session(
        self,
        tenant_id: str,
        content_session_id: str,
        project: str,
        user_prompt: Optional[str] = None,
        metadata: Optional[dict[str, object]] = None,
    ) -> SessionRecord:
        memory_session_id = str(uuid4())
        started_at = self._now_iso()
        metadata_json = json.dumps(metadata) if metadata is not None else None
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO sessions (
                    tenant_id, content_session_id, memory_session_id, project,
                    user_prompt, started_at, status, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    content_session_id,
                    memory_session_id,
                    project,
                    user_prompt,
                    started_at,
                    "active",
                    metadata_json,
                ),
            )
            self.conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to create session")
            self.conn.rollback()
            raise
        session = self.get_session_by_content_id(content_session_id)
        if session is None:
            raise RuntimeError("Failed to retrieve session after insert")
        return session

    def get_session_by_content_id(
        self, content_session_id: str
    ) -> Optional[SessionRecord]:
        return self._fetch_session(
            "SELECT * FROM sessions WHERE content_session_id = ?",
            (content_session_id,),
        )

    def get_session_by_memory_id(
        self, memory_session_id: str
    ) -> Optional[SessionRecord]:
        return self._fetch_session(
            "SELECT * FROM sessions WHERE memory_session_id = ?",
            (memory_session_id,),
        )

    def get_session_by_id(self, session_id: int) -> Optional[SessionRecord]:
        return self._fetch_session("SELECT * FROM sessions WHERE id = ?", (session_id,))

    def update_session_status(
        self,
        memory_session_id: str,
        status: SessionStatus,
        ended_at: Optional[str] = None,
    ) -> None:
        status_value = self._enum_to_value(status)
        if ended_at is None and status_value in {"completed", "failed"}:
            ended_at = self._now_iso()
        try:
            _ = self.conn.execute(
                """
                UPDATE sessions
                SET status = ?, ended_at = ?
                WHERE memory_session_id = ?
                """,
                (status_value, ended_at, memory_session_id),
            )
            self.conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to update session status")
            self.conn.rollback()
            raise

    def list_sessions(
        self,
        tenant_id: Optional[str] = None,
        project: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRecord]:
        where_clauses = []
        params: list[object] = []
        if tenant_id:
            where_clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if project:
            where_clauses.append("project = ?")
            params.append(project)
        if status:
            status_value = self._enum_to_value(status)
            where_clauses.append("status = ?")
            params.append(status_value)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = (
            f"SELECT * FROM sessions {where_sql} "
            "ORDER BY started_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])
        try:
            cursor = self.conn.execute(query, params)
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_session(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to list sessions")
            raise

    def add_event(
        self,
        memory_session_id: str,
        kind: EventKind,
        title: Optional[str] = None,
        payload_json: Optional[dict[str, object]] = None,
        redaction_level: Optional[RedactionLevel] = None,
    ) -> int:
        timestamp = self._now_iso()
        kind_value = self._enum_to_value(kind)
        redaction_value = self._enum_to_value(redaction_level, default="none")
        payload_text = json.dumps(payload_json) if payload_json is not None else None
        try:
            cursor = self.conn.execute(
                """
                INSERT INTO session_events (
                    memory_session_id, timestamp, kind, title, payload_json, redaction_level
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_session_id,
                    timestamp,
                    kind_value,
                    title,
                    payload_text,
                    redaction_value,
                ),
            )
            self.conn.commit()
            return self._lastrowid(cursor, "Failed to create session event")
        except sqlite3.Error:
            logger.exception("Failed to add session event")
            self.conn.rollback()
            raise

    def get_events_for_session(
        self,
        memory_session_id: str,
        kinds: Optional[Sequence[EventKind]] = None,
    ) -> list[SessionEvent]:
        params: list[object] = [memory_session_id]
        kind_clause = ""
        if kinds:
            placeholders = ",".join(["?"] * len(kinds))
            kind_clause = f" AND kind IN ({placeholders})"
            params.extend([k.value if hasattr(k, "value") else str(k) for k in kinds])
        query = (
            "SELECT * FROM session_events WHERE memory_session_id = ?"
            f"{kind_clause} ORDER BY timestamp ASC"
        )
        try:
            cursor = self.conn.execute(query, params)
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_event(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch session events")
            raise

    def store_observation(
        self,
        memory_session_id: str,
        type: ObservationType,
        title: str,
        subtitle: Optional[str] = None,
        facts_json: Optional[dict[str, object]] = None,
        narrative: Optional[str] = None,
        concepts_json: Optional[Iterable[str]] = None,
        files_json: Optional[Iterable[str]] = None,
        vector_ref: Optional[str] = None,
    ) -> int:
        timestamp = self._now_iso()
        type_value = type.value if hasattr(type, "value") else str(type)
        try:
            cursor = self.conn.execute(
                """
                INSERT INTO observations (
                    memory_session_id, timestamp, type, title, subtitle, facts_json,
                    narrative, concepts_json, files_json, vector_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_session_id,
                    timestamp,
                    type_value,
                    title,
                    subtitle,
                    json.dumps(facts_json) if facts_json is not None else None,
                    narrative,
                    json.dumps(list(concepts_json))
                    if concepts_json is not None
                    else None,
                    json.dumps(list(files_json)) if files_json is not None else None,
                    vector_ref,
                ),
            )
            self.conn.commit()
            return self._lastrowid(cursor, "Failed to store observation")
        except sqlite3.Error:
            logger.exception("Failed to store observation")
            self.conn.rollback()
            raise

    def get_observations_for_session(
        self, memory_session_id: str
    ) -> list[CrossObservation]:
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM observations
                WHERE memory_session_id = ?
                ORDER BY timestamp ASC
                """,
                (memory_session_id,),
            )
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_observation(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch observations for session")
            raise

    def get_recent_observations(
        self,
        project: str,
        limit: int = 50,
        types: Optional[Sequence[ObservationType]] = None,
    ) -> list[CrossObservation]:
        params: list[object] = [project]
        type_clause = ""
        if types:
            placeholders = ",".join(["?"] * len(types))
            type_clause = f" AND observations.type IN ({placeholders})"
            params.extend([t.value if hasattr(t, "value") else str(t) for t in types])
        query = (
            """
            SELECT observations.* FROM observations
            JOIN sessions ON sessions.memory_session_id = observations.memory_session_id
            WHERE sessions.project = ?
            """
            f"{type_clause} ORDER BY observations.timestamp DESC LIMIT ?"
        )
        params.append(limit)
        try:
            cursor = self.conn.execute(query, params)
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_observation(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch recent observations")
            raise

    def get_observations_by_ids(self, obs_ids: list[int]) -> list[CrossObservation]:
        if not obs_ids:
            return []
        placeholders = ",".join(["?"] * len(obs_ids))
        query = f"SELECT * FROM observations WHERE obs_id IN ({placeholders})"
        try:
            cursor = self.conn.execute(query, obs_ids)
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_observation(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch observations by ids")
            raise

    def store_summary(
        self,
        memory_session_id: str,
        request: Optional[str] = None,
        investigated: Optional[str] = None,
        learned: Optional[str] = None,
        completed: Optional[str] = None,
        next_steps: Optional[str] = None,
        vector_ref: Optional[str] = None,
    ) -> int:
        timestamp = self._now_iso()
        try:
            cursor = self.conn.execute(
                """
                INSERT INTO session_summaries (
                    memory_session_id, timestamp, request, investigated,
                    learned, completed, next_steps, vector_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_session_id,
                    timestamp,
                    request,
                    investigated,
                    learned,
                    completed,
                    next_steps,
                    vector_ref,
                ),
            )
            self.conn.commit()
            return self._lastrowid(cursor, "Failed to store session summary")
        except sqlite3.Error:
            logger.exception("Failed to store session summary")
            self.conn.rollback()
            raise

    def get_summary_for_session(
        self, memory_session_id: str
    ) -> Optional[SessionSummary]:
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM session_summaries
                WHERE memory_session_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (memory_session_id,),
            )
            row = cast(Optional[sqlite3.Row], cursor.fetchone())
            return self._row_to_summary(row) if row else None
        except sqlite3.Error:
            logger.exception("Failed to fetch session summary")
            raise

    def get_recent_summaries(
        self, project: str, limit: int = 10
    ) -> list[SessionSummary]:
        try:
            cursor = self.conn.execute(
                """
                SELECT session_summaries.* FROM session_summaries
                JOIN sessions ON sessions.memory_session_id = session_summaries.memory_session_id
                WHERE sessions.project = ?
                ORDER BY session_summaries.timestamp DESC
                LIMIT ?
                """,
                (project, limit),
            )
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_summary(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch recent summaries")
            raise

    def create_link(
        self,
        memory_entry_id: str,
        source_kind: str,
        source_id: int,
        score: float = 0.0,
    ) -> int:
        timestamp = self._now_iso()
        try:
            cursor = self.conn.execute(
                """
                INSERT INTO memory_links (
                    memory_entry_id, source_kind, source_id, score, timestamp
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (memory_entry_id, source_kind, source_id, score, timestamp),
            )
            self.conn.commit()
            return self._lastrowid(cursor, "Failed to create memory link")
        except sqlite3.Error:
            logger.exception("Failed to create memory link")
            self.conn.rollback()
            raise

    def get_links_for_entry(self, memory_entry_id: str) -> list[MemoryLink]:
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM memory_links
                WHERE memory_entry_id = ?
                ORDER BY score DESC, timestamp DESC
                """,
                (memory_entry_id,),
            )
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_link(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch links for entry")
            raise

    def get_links_for_source(
        self, source_kind: str, source_id: int
    ) -> list[MemoryLink]:
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM memory_links
                WHERE source_kind = ? AND source_id = ?
                ORDER BY score DESC, timestamp DESC
                """,
                (source_kind, source_id),
            )
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_link(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch links for source")
            raise

    def record_consolidation_run(
        self,
        tenant_id: str,
        policy_json: Optional[dict[str, object]] = None,
        stats_json: Optional[dict[str, object]] = None,
    ) -> int:
        timestamp = self._now_iso()
        try:
            cursor = self.conn.execute(
                """
                INSERT INTO consolidation_runs (
                    tenant_id, timestamp, policy_json, stats_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    timestamp,
                    json.dumps(policy_json) if policy_json is not None else None,
                    json.dumps(stats_json) if stats_json is not None else None,
                ),
            )
            self.conn.commit()
            return self._lastrowid(cursor, "Failed to record consolidation run")
        except sqlite3.Error:
            logger.exception("Failed to record consolidation run")
            self.conn.rollback()
            raise

    def get_recent_consolidation_runs(
        self,
        tenant_id: str,
        limit: int = 10,
    ) -> list[ConsolidationRun]:
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM consolidation_runs
                WHERE tenant_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (tenant_id, limit),
            )
            rows = cast(list[sqlite3.Row], cursor.fetchall())
            return [self._row_to_consolidation_run(row) for row in rows]
        except sqlite3.Error:
            logger.exception("Failed to fetch consolidation runs")
            raise

    def get_stats(
        self, tenant_id: Optional[str] = None, project: Optional[str] = None
    ) -> dict[str, int]:
        stats: dict[str, int] = {}
        stats["sessions"] = self._count_sessions(tenant_id, project)
        stats["events"] = self._count_events(tenant_id, project)
        stats["observations"] = self._count_observations(tenant_id, project)
        stats["summaries"] = self._count_summaries(tenant_id, project)
        return stats

    def _fetch_session(
        self, query: str, params: Sequence[object]
    ) -> Optional[SessionRecord]:
        try:
            cursor = self.conn.execute(query, params)
            row = cast(Optional[sqlite3.Row], cursor.fetchone())
            return self._row_to_session(row) if row else None
        except sqlite3.Error:
            logger.exception("Failed to fetch session")
            raise

    def _row_to_session(self, row: sqlite3.Row) -> SessionRecord:
        data = cast(dict[str, object], dict(row))
        data["status"] = self._coerce_enum(SessionStatus, data.get("status"))
        return self._build_model(SessionRecord, data)

    def _row_to_event(self, row: sqlite3.Row) -> SessionEvent:
        data = cast(dict[str, object], dict(row))
        data["kind"] = self._coerce_enum(EventKind, data.get("kind"))
        data["redaction_level"] = self._coerce_enum(
            RedactionLevel, data.get("redaction_level")
        )
        return self._build_model(SessionEvent, data)

    def _row_to_observation(self, row: sqlite3.Row) -> CrossObservation:
        data = cast(dict[str, object], dict(row))
        data["type"] = self._coerce_enum(ObservationType, data.get("type"))
        return self._build_model(CrossObservation, data)

    def _row_to_summary(self, row: sqlite3.Row) -> SessionSummary:
        data = cast(dict[str, object], dict(row))
        return self._build_model(SessionSummary, data)

    def _row_to_link(self, row: sqlite3.Row) -> MemoryLink:
        data = cast(dict[str, object], dict(row))
        return self._build_model(MemoryLink, data)

    def _row_to_consolidation_run(self, row: sqlite3.Row) -> ConsolidationRun:
        data = cast(dict[str, object], dict(row))
        return self._build_model(ConsolidationRun, data)

    def _count_sessions(self, tenant_id: Optional[str], project: Optional[str]) -> int:
        where_clauses = []
        params: list[object] = []
        if tenant_id:
            where_clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if project:
            where_clauses.append("project = ?")
            params.append(project)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = f"SELECT COUNT(*) FROM sessions {where_sql}"
        return self._fetch_count(query, params)

    def _count_events(self, tenant_id: Optional[str], project: Optional[str]) -> int:
        return self._count_joined(
            "session_events",
            "session_events.memory_session_id = sessions.memory_session_id",
            tenant_id,
            project,
        )

    def _count_observations(
        self, tenant_id: Optional[str], project: Optional[str]
    ) -> int:
        return self._count_joined(
            "observations",
            "observations.memory_session_id = sessions.memory_session_id",
            tenant_id,
            project,
        )

    def _count_summaries(self, tenant_id: Optional[str], project: Optional[str]) -> int:
        return self._count_joined(
            "session_summaries",
            "session_summaries.memory_session_id = sessions.memory_session_id",
            tenant_id,
            project,
        )

    def _count_joined(
        self,
        table: str,
        join_condition: str,
        tenant_id: Optional[str],
        project: Optional[str],
    ) -> int:
        where_clauses = []
        params: list[object] = []
        if tenant_id:
            where_clauses.append("sessions.tenant_id = ?")
            params.append(tenant_id)
        if project:
            where_clauses.append("sessions.project = ?")
            params.append(project)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        query = (
            f"SELECT COUNT(*) FROM {table} "
            f"JOIN sessions ON {join_condition} {where_sql}"
        )
        return self._fetch_count(query, params)

    def _fetch_count(self, query: str, params: Sequence[object]) -> int:
        try:
            cursor = self.conn.execute(query, params)
            row = cast(Optional[sqlite3.Row], cursor.fetchone())
            if row is None:
                return 0
            return int(cast(int, row[0]))
        except sqlite3.Error:
            logger.exception("Failed to fetch count")
            raise

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _loads_json(payload: object) -> object:
        if payload is None:
            return None
        if not isinstance(payload, str):
            return payload
        try:
            return cast(object, json.loads(payload))
        except json.JSONDecodeError:
            logger.exception("Failed to decode JSON payload")
            return None

    @staticmethod
    def _build_model(model_cls, data: dict[str, object]):
        if hasattr(model_cls, "model_fields"):
            allowed = set(model_cls.model_fields.keys())
        elif hasattr(model_cls, "__fields__"):
            allowed = set(model_cls.__fields__.keys())
        elif hasattr(model_cls, "__dataclass_fields__"):
            allowed = set(model_cls.__dataclass_fields__.keys())
        elif hasattr(model_cls, "__annotations__"):
            allowed = set(model_cls.__annotations__.keys())
        else:
            return model_cls(**data)
        filtered = {key: value for key, value in data.items() if key in allowed}
        return model_cls(**filtered)

    @staticmethod
    def _coerce_enum(enum_cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, str):
            try:
                return enum_cls(value)
            except Exception:
                return value
        return value

    @staticmethod
    def _enum_to_value(value: object, default: Optional[str] = None) -> Optional[str]:
        if value is None:
            return default
        value_attr = cast(object, getattr(value, "value", None))
        if value_attr is not None:
            return str(value_attr)
        return str(value)

    @staticmethod
    def _lastrowid(cursor: sqlite3.Cursor, error_message: str) -> int:
        lastrowid = cursor.lastrowid
        if lastrowid is None:
            raise RuntimeError(error_message)
        return int(lastrowid)
