<div align="center">

# 🧠 SimpleMem-Cross

### Persistent Cross-Conversation Memory for LLM Agents

<p>
<b>Your agents remember everything. Across every conversation.</b><br/>
<small>Context, decisions, and learnings persist automatically — no manual re-injection needed.</small>
</p>

<br/>

<p>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-2EA44F?style=flat&labelColor=555" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-127%20passed-brightgreen?style=flat&labelColor=555" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10-3775A9?style=flat&labelColor=555&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/async-first-14B8A6?style=flat&labelColor=555" alt="Async"></a>
  <br/>
  <a href="#-http-api"><img src="https://img.shields.io/badge/REST_API-FastAPI-009688?style=flat&labelColor=555&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="#-mcp-integration"><img src="https://img.shields.io/badge/MCP-8_tools-5865F2?style=flat&labelColor=555" alt="MCP Tools"></a>
  <a href="#"><img src="https://img.shields.io/badge/storage-SQLite_+_LanceDB-FF6B6B?style=flat&labelColor=555" alt="Storage"></a>
</p>

<br/>

[![Performance](https://img.shields.io/badge/🏆_LoCoMo_Benchmark-64%25_faster_than_Claude--Mem-FFD700?style=for-the-badge&labelColor=FFD700&color=FF6B6B)](../)

<br/>

[Features](#-key-features) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Architecture](#-architecture) • [Configuration](#%EF%B8%8F-configuration)

</div>

---

## 🏆 Performance

<div align="center">

| System | LoCoMo Score | vs SimpleMem |
|:-------|:------------:|:------------:|
| **SimpleMem** | **48** | — |
| Claude-Mem | 29.3 | **+64%** |

</div>

> **SimpleMem-Cross** extends [SimpleMem](https://github.com/aiming-lab/SimpleMem) with persistent cross-conversation memory. The original SimpleMem code is preserved **byte-identical** — all new functionality resides in this `cross/` package using composition, not modification.

---

## ✨ Key Features

<table>
<tr>
<td width="50%" valign="top">

### 🔄 Session Lifecycle
Full session management with **start → record → stop → end** lifecycle. Every event is tracked, timestamped, and persisted.

### 🎯 Automatic Context Injection
Token-budgeted context from previous sessions is **injected automatically** at session start. No manual prompt engineering.

### 📝 Smart Event Collection
Record messages, tool uses, and file changes with **3-tier automatic redaction** for secrets and sensitive data.

</td>
<td width="50%" valign="top">

### 🔍 Observation Extraction
Heuristic extraction of **decisions, discoveries, and learnings** from conversations. Your agent learns from experience.

### 🔗 Provenance Tracking
Every memory entry **links back to source evidence**. Know exactly where each piece of context originated.

### 🧹 Memory Consolidation
Automatic **decay, merge, and prune** of old memories. Quality over quantity, maintained automatically.

</td>
</tr>
</table>

---

## 🚀 Quick Start

```python
import asyncio
from cross.orchestrator import create_orchestrator

async def main():
    # 🔧 Create the orchestrator for your project
    orch = create_orchestrator(project="my-project")

    # 🚀 Start a new session — context from previous sessions is injected automatically
    result = await orch.start_session(
        content_session_id="session-001",
        user_prompt="Continue building the REST API authentication",
    )
    memory_session_id = result["memory_session_id"]
    print(result["context"])  # 📚 Relevant context from previous sessions

    # 📝 Record events during the session
    await orch.record_message(memory_session_id, "User asked about JWT auth")
    await orch.record_tool_use(
        memory_session_id,
        tool_name="read_file",
        tool_input="auth/jwt.py",
        tool_output="class JWTHandler: ...",
    )
    await orch.record_message(memory_session_id, "Implemented token refresh logic", role="assistant")

    # ✅ Finalize — extracts observations, generates summary, stores memory entries
    report = await orch.stop_session(memory_session_id)
    print(f"Stored {report.entries_stored} memory entries, {report.observations_count} observations")

    # 🧹 Cleanup
    await orch.end_session(memory_session_id)
    orch.close()

asyncio.run(main())
```

---

## 📦 Installation

SimpleMem-Cross uses the same dependencies as SimpleMem, plus standard library `sqlite3`:

```bash
pip install -r requirements.txt
```

> **Note**: No additional packages required. LanceDB and Pydantic are already in the SimpleMem dependency tree.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           Agent Frameworks (Claude Code / Cursor / custom)      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────────┐
│  Hook/Lifecycle     │               │    HTTP/MCP API         │
│  Adapter            │               │    (FastAPI)            │
│  ─────────────────  │               │  ─────────────────────  │
│  SessionStart       │               │  POST /sessions/start   │
│  UserMessage        │               │  POST /sessions/{id}/*  │
│  ToolUse            │               │  POST /search           │
│  Stop / End         │               │  GET  /stats            │
└─────────┬───────────┘               └───────────┬─────────────┘
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
          ┌───────────────────────────────────────┐
          │         CrossMemOrchestrator          │
          │         ═══════════════════           │
          │  • Facade for all memory operations   │
          │  • Multi-tenant isolation             │
          │  • Async-first design                 │
          └───────────────────┬───────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
┌─────────────┐       ┌─────────────┐         ┌─────────────────┐
│   Session   │       │   Context   │         │  Consolidation  │
│   Manager   │       │   Injector  │         │     Worker      │
│ ─────────── │       │ ─────────── │         │ ─────────────── │
│ SQLite DB   │       │ Token-      │         │ Decay / Merge   │
│ • sessions  │       │ budgeted    │         │ Prune old       │
│ • events    │       │ context     │         │ entries         │
│ • summaries │       │ bundle      │         │                 │
└──────┬──────┘       └──────┬──────┘         └────────┬────────┘
       │                     │                         │
       └─────────────────────┼─────────────────────────┘
                             ▼
          ┌───────────────────────────────────────┐
          │    Cross-Session Vector Store         │
          │           (LanceDB)                   │
          │    ═══════════════════════════        │
          │  • Semantic search (1024-d vectors)   │
          │  • Keyword matching (BM25-style)      │
          │  • Structured metadata filtering      │
          │  • Provenance fields for tracing      │
          └───────────────────────────────────────┘
                             │
                             ▼
          ┌───────────────────────────────────────┐
          │  Reuses SimpleMem 3-Stage Pipeline    │
          │  (Composition, not modification)      │
          │  ─────────────────────────────────    │
          │  MemoryBuilder → HybridRetriever →    │
          │  AnswerGenerator                      │
          └───────────────────────────────────────┘
```

### 🎯 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Composition over modification** | Original SimpleMem is wrapped, never edited |
| **SQLite for session timeline** | Sessions, events, observations, summaries |
| **LanceDB for vectors** | Cross-session memory entries with provenance |
| **Hook-based lifecycle** | `SessionStart → UserMessage/ToolUse → Stop → End` |
| **Progressive disclosure** | Token-budgeted context injection at session start |
| **Provenance tracking** | Every vector links back to its source evidence |

---

## 📚 Module Reference

| Module | Lines | Description |
|:-------|------:|:------------|
| `types.py` | 227 | 📋 Pydantic models — enums, records, ContextBundle, FinalizationReport |
| `storage_sqlite.py` | 805 | 🗄️ SQLite backend — 6 tables (sessions, events, observations, summaries) |
| `storage_lancedb.py` | 542 | 🔍 LanceDB vector store — semantic/keyword/structured search |
| `hooks.py` | 401 | 🪝 Abstract `SessionHooks` with 5 async lifecycle methods |
| `collectors.py` | 413 | 📝 `RedactionFilter` (3-tier regex), thread-safe `EventCollector` |
| `session_manager.py` | 755 | 🔄 Full lifecycle orchestration — start/record/finalize/end |
| `context_injector.py` | 385 | 💉 Token-budgeted `ContextBundle` builder and renderer |
| `orchestrator.py` | 530 | 🎭 Top-level facade `CrossMemOrchestrator` and factory |
| `api_http.py` | 556 | 🌐 FastAPI router — 8 REST endpoints with Pydantic models |
| `api_mcp.py` | 620 | 🔌 `MCPToolRegistry` — 8 MCP tool definitions with JSON Schema |
| `consolidation.py` | 390 | 🧹 `ConsolidationWorker` — decay/merge/prune pipeline |

---

## 📖 API Reference

### CrossMemOrchestrator

<table>
<tr><th>Method</th><th>Parameters</th><th>Returns</th><th>Description</th></tr>
<tr>
  <td><code>start_session</code></td>
  <td><code>content_session_id, user_prompt?</code></td>
  <td><code>dict</code></td>
  <td>Start session with context injection</td>
</tr>
<tr>
  <td><code>record_message</code></td>
  <td><code>memory_session_id, content, role?</code></td>
  <td><code>None</code></td>
  <td>Record a chat message event</td>
</tr>
<tr>
  <td><code>record_tool_use</code></td>
  <td><code>memory_session_id, tool_name, tool_input, tool_output</code></td>
  <td><code>None</code></td>
  <td>Record a tool invocation</td>
</tr>
<tr>
  <td><code>stop_session</code></td>
  <td><code>memory_session_id</code></td>
  <td><code>FinalizationReport</code></td>
  <td>Finalize: extract observations, generate summary</td>
</tr>
<tr>
  <td><code>end_session</code></td>
  <td><code>memory_session_id</code></td>
  <td><code>None</code></td>
  <td>Mark session completed, cleanup</td>
</tr>
<tr>
  <td><code>search</code></td>
  <td><code>query, top_k?</code></td>
  <td><code>list[CrossMemoryEntry]</code></td>
  <td>Semantic search across all sessions</td>
</tr>
<tr>
  <td><code>get_context_for_prompt</code></td>
  <td><code>user_prompt?</code></td>
  <td><code>str</code></td>
  <td>Build and render context for system prompt</td>
</tr>
<tr>
  <td><code>get_stats</code></td>
  <td>—</td>
  <td><code>dict</code></td>
  <td>Storage statistics</td>
</tr>
<tr>
  <td><code>close</code></td>
  <td>—</td>
  <td><code>None</code></td>
  <td>Close SQLite connection</td>
</tr>
</table>

---

## 🌐 HTTP API

<details open>
<summary><b>REST Endpoints</b></summary>

| Method | Path | Description |
|:-------|:-----|:------------|
| `POST` | `/cross/sessions/start` | 🚀 Start a new cross-session |
| `POST` | `/cross/sessions/{id}/message` | 💬 Record a message event |
| `POST` | `/cross/sessions/{id}/tool-use` | 🔧 Record a tool-use event |
| `POST` | `/cross/sessions/{id}/stop` | ✅ Finalize session memory |
| `POST` | `/cross/sessions/{id}/end` | 🏁 End and cleanup session |
| `POST` | `/cross/search` | 🔍 Search cross-session memories |
| `GET` | `/cross/stats` | 📊 Get memory system statistics |
| `GET` | `/cross/health` | 💚 Health check with uptime |

</details>

### Running the HTTP Server

```python
from cross.api_http import create_app

app = create_app(project="my-project")

# Run with uvicorn
# uvicorn cross.api_http:app --host 0.0.0.0 --port 8000
```

Or mount on an existing FastAPI app:

```python
from cross.api_http import create_cross_router
from cross.orchestrator import create_orchestrator

orch = create_orchestrator(project="my-project")
router = create_cross_router(orch)
app.include_router(router, prefix="/cross")
```

---

## 🔌 MCP Integration

<details open>
<summary><b>Available MCP Tools</b></summary>

| Tool Name | Description |
|:----------|:------------|
| `cross_session_start` | 🚀 Start a new cross-session memory session |
| `cross_session_message` | 💬 Record a user/assistant message |
| `cross_session_tool_use` | 🔧 Record a tool invocation |
| `cross_session_stop` | ✅ Finalize and persist session memory |
| `cross_session_end` | 🏁 End session and cleanup |
| `cross_session_search` | 🔍 Search across all session memories |
| `cross_session_context` | 📚 Get context bundle for system prompt |
| `cross_session_stats` | 📊 Get memory system statistics |

</details>

### MCP Setup

```python
from cross.api_mcp import create_mcp_tools
from cross.orchestrator import create_orchestrator

orch = create_orchestrator(project="my-project")
tools = create_mcp_tools(orch)

# Get tool definitions for MCP server registration
definitions = tools.get_tool_definitions()

# Dispatch a tool call
result = await tools.call_tool("cross_session_start", {
    "tenant_id": "default",
    "content_session_id": "ses-1",
    "project": "my-project",
    "user_prompt": "Help me debug the auth module",
})
```

---

## ⚙️ Configuration

### 📁 Default Paths

| Setting | Default | Description |
|:--------|:--------|:------------|
| SQLite DB | `~/.simplemem-cross/cross_memory.db` | Session metadata, events, observations |
| LanceDB | `~/.simplemem-cross/lancedb_cross` | Vector storage for memory entries |
| Max context tokens | `2000` | Token budget for context injection |

### 🔧 Custom Configuration

```python
orch = create_orchestrator(
    project="my-project",
    tenant_id="team-alpha",
    db_path="/custom/path/memory.db",
    lancedb_path="/custom/path/lancedb",
    max_context_tokens=3000,
)
```

### 👥 Multi-Tenant Support

Pass `tenant_id` to isolate memory across tenants. Each tenant's memories are stored and retrieved independently.

---

## 🧹 Consolidation

The consolidation worker maintains memory quality over time:

```python
from cross.consolidation import ConsolidationWorker, ConsolidationPolicy

policy = ConsolidationPolicy(
    max_age_days=90,                    # ⏰ Decay entries older than 90 days
    decay_factor=0.9,                   # 📉 Multiply importance by 0.9 per period
    merge_similarity_threshold=0.95,   # 🔗 Merge near-duplicates
    min_importance=0.05,               # 🗑️ Prune below this threshold
)

worker = ConsolidationWorker(sqlite_storage, vector_store, policy)
result = worker.run(tenant_id="default")

print(f"📉 Decayed: {result.decayed_count}")
print(f"🔗 Merged: {result.merged_count}")
print(f"🗑️ Pruned: {result.pruned_count}")
```

---

## 🧪 Testing

```bash
# 🧪 Run all cross-session tests
pytest cross/tests/ -v

# 📋 Run specific test module
pytest cross/tests/test_types.py -v
pytest cross/tests/test_storage.py -v
pytest cross/tests/test_e2e.py -v
```

> **Note**: Tests use real SQLite (temp databases) and mock LanceDB. No external services, API keys, or GPU required.

---

## 📏 Constraints

| Constraint | Reason |
|:-----------|:-------|
| ✅ **Original SimpleMem is byte-identical** | Published research paper; never modified |
| ✅ **All code in English** | No Chinese in code, comments, docstrings, or strings |
| ✅ **Python-only** | Matches SimpleMem's tech stack |
| ✅ **Composition pattern** | SimpleMem is wrapped via duck typing, never subclassed |

---

<div align="center">

<br/>

**[⬆ Back to Top](#-simplemem-cross)**

<br/>

Made with ❤️ by the SimpleMem Team

</div>
