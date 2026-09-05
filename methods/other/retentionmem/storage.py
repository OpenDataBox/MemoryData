"""Transactional local storage with an append-only source archive."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from .types import Event, MemoryUnit, SourceSpan


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT, 'raw events are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT, 'raw events are immutable'); END;
            CREATE TABLE IF NOT EXISTS units(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS costs(seq INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS embeddings(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
        """)

    def meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, canonical(value)))

    def observe(self, event: Event) -> bool:
        payload = canonical(event.to_dict())
        old = self.db.execute("SELECT payload FROM events WHERE id=?", (event.id,)).fetchone()
        if old:
            if old[0] != payload:
                raise ValueError(f"Event {event.id} already exists with different content")
            return False
        # The same transaction publishes the raw event and its immediately searchable unit.
        text = f"[{event.timestamp or 'time unspecified'} | {event.actor} | {event.kind}]\n{event.text}"
        unit = MemoryUnit("event:" + event.id, text, (event.id,), event.source_ids or (event.id,))
        with self.db:
            self.db.execute("INSERT INTO events(id,payload) VALUES (?,?)", (event.id, payload))
            self.db.execute("INSERT INTO units VALUES (?,?)", (unit.id, canonical(asdict(unit))))
            pending = self.meta("pending", []) + [unit.id]
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('pending',?)", (canonical(pending),))
        return True

    def events(self, ids=None) -> list[Event]:
        if ids is not None:
            wanted = set(ids)
            if not wanted:
                return []
            marks = ",".join("?" for _ in wanted)
            rows = self.db.execute(f"SELECT payload FROM events WHERE id IN ({marks}) ORDER BY seq", tuple(wanted))
        else:
            rows = self.db.execute("SELECT payload FROM events ORDER BY seq")
        return [Event(**json.loads(row[0])) for row in rows]

    def units(self) -> list[MemoryUnit]:
        result = []
        for row in self.db.execute("SELECT payload FROM units ORDER BY rowid"):
            value = json.loads(row[0])
            value["event_ids"] = tuple(value["event_ids"])
            value["source_ids"] = tuple(value["source_ids"])
            value["residual_spans"] = tuple(SourceSpan(**s) for s in value["residual_spans"])
            result.append(MemoryUnit(**value))
        return result

    def commit(self, region: list[MemoryUnit], replacement: MemoryUnit, audit: dict):
        """Reject stale proposals; publish a complete successor and audit atomically."""
        with self.db:
            for unit in region:
                row = self.db.execute("SELECT payload FROM units WHERE id=?", (unit.id,)).fetchone()
                if not row or row[0] != canonical(asdict(unit)):
                    raise RuntimeError("Maintenance region changed while the proposal was being evaluated")
            for unit in region:
                self.db.execute("DELETE FROM units WHERE id=?", (unit.id,))
            self.db.execute("INSERT INTO units VALUES (?,?)", (replacement.id, canonical(asdict(replacement))))
            self.db.execute("INSERT INTO audit(payload) VALUES (?)", (canonical(audit),))
            self._consume_pending([u.id for u in region])

    def _consume_pending(self, ids):
        consumed = set(ids)
        pending = [p for p in self.meta("pending", []) if p not in consumed]
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('pending',?)", (canonical(pending),))

    def audit(self, value, consumed=()):
        with self.db:
            self.db.execute("INSERT INTO audit(payload) VALUES (?)", (canonical(value),))
            self._consume_pending(consumed)

    def audits(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM audit ORDER BY seq")]

    def cost(self, value):
        with self.db:
            self.db.execute("INSERT INTO costs(payload) VALUES (?)", (canonical(value),))

    def costs(self):
        rows = [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM costs ORDER BY seq")]
        keys = ("calls", "input_tokens", "output_tokens", "seconds", "cache_hits", "missing_usage_calls")
        stages = {}
        for row in rows:
            part = stages.setdefault(row["stage"], {k: 0 for k in keys})
            for key in keys:
                part[key] += row.get(key, 0)
            if row.get("usage_reported") is False and "missing_usage_calls" not in row:
                part["missing_usage_calls"] += row.get("calls", 0)
        totals = {k: sum(v[k] for v in stages.values()) for k in keys if k != "seconds"}
        # Nested service durations overlap maintenance/retrieval wall time. Never add them.
        totals["service_seconds"] = sum(r.get("seconds", 0) for r in rows if r.get("kind") in {"llm", "embedding"})
        totals["wall_seconds"] = sum(r.get("seconds", 0) for r in rows if r["stage"].endswith("_wall") or r["stage"] == "write")
        totals["policy_seconds"] = sum(r.get("seconds", 0) for r in rows if r["stage"] == "policy")
        return {"by_stage": stages, **totals,
                "usd": None, "pricing": "not_configured"}

    def storage_stats(self):
        raw_bytes = sum(len(r[0].encode("utf-8")) for r in self.db.execute("SELECT payload FROM events"))
        active_bytes = sum(len(r[0].encode("utf-8")) for r in self.db.execute("SELECT payload FROM units"))
        return {"raw_bytes": raw_bytes, "active_bytes": active_bytes,
                "database_bytes": self.path.stat().st_size, "raw_events": len(self.events()),
                "active_units": len(self.units())}

    def close(self):
        self.db.close()
