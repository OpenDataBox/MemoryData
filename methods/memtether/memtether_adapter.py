# -*- coding: utf-8 -*-
"""
memtether_adapter.py — MemTether adapter for MemoryData benchmark suite.

MemTether is a cross-client AI memory hub where heterogeneous AI agents share
the same physical SQLite database via file-level pointers (junction/symlink).

GitHub: https://github.com/MemTether/MemTether
PyPI:   https://pypi.org/project/memtether/

This adapter wraps MemTether's gateway.py (write) and memsearch.py (retrieve)
to expose the interface MemoryData expects:
  - memorize(context_chunks)  → gateway.py remember
  - retrieve(query, top_k)    → memsearch.search_hybrid
  - answer via LLM (standard OpenAI-compatible call, same as other adapters)
"""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import json
from typing import List, Dict, Union


class MemTetherMemory:
    """Wrapper around MemTether CLI for MemoryData benchmark integration."""

    def __init__(self, agent_config: dict, agent_save_folder: str):
        """
        Initialize MemTether instance for this benchmark run.

        Uses an isolated temporary SQLite database (not the production one)
        to avoid cross-contamination between benchmark runs.
        """
        self.agent_config = agent_config
        self.agent_save_folder = agent_save_folder
        self.source = agent_config.get("memtether_source", "memorydata_bench")
        self.retrieve_num = agent_config.get("retrieve_num", 12)

        # Locate MemTether installation
        # Priority: explicit config → pip-installed → local memory_hub
        self._mt_dir = self._find_memtether_dir()
        if not self._mt_dir:
            raise RuntimeError(
                "MemTether not found. Install with: pip install memtether "
                "or set memtether_dir in config YAML."
            )

        self._gateway = os.path.join(self._mt_dir, "gateway.py")
        self._memsearch = os.path.join(self._mt_dir, "memsearch.py")

        # Find the correct Python interpreter (needs .venv-memory for semantic)
        self._py = self._find_python()

        # Create isolated DB in agent_save_folder
        self._db_dir = os.path.join(agent_save_folder, "memtether_db")
        os.makedirs(self._db_dir, exist_ok=True)
        self._db_path = os.path.join(self._db_dir, "memory.db")

        # Set environment for isolated MemTether instance
        self._env = os.environ.copy()
        self._env["MEM_DB"] = self._db_path
        self._env["MEM_PROJ_PATH"] = os.path.join(self._db_dir, "MEMORY.md")
        self._env["MEM_EMBED_BACKEND"] = agent_config.get(
            "memtether_embedding_backend", "local"
        )
        self._env["MEM_DEFAULT_SOURCE"] = self.source
        self._env["PYTHONIOENCODING"] = "utf-8"

        # Initialize the isolated DB
        self._run_gateway(["init"])
        print(f"[MemTether] Initialized isolated DB: {self._db_path}")

    def _find_memtether_dir(self) -> str:
        """Locate MemTether source directory."""
        # 1. Explicit config
        explicit = self.agent_config.get("memtether_dir")
        if explicit and os.path.isdir(explicit):
            return explicit
        # 2. Local memory_hub (development machine)
        local = r"E:\RUANJIAN\memory_hub"
        if os.path.isdir(local):
            return local
        # 3. pip-installed memtether
        try:
            import memtether
            return os.path.dirname(os.path.abspath(memtether.__file__))
        except ImportError:
            pass
        return None

    def _find_python(self) -> str:
        """Find the correct Python interpreter for MemTether."""
        # Prefer .venv-memory (has chromadb/onnxruntime for semantic retrieval)
        venv_py = os.path.join(self._mt_dir, ".venv-memory", "Scripts", "python.exe")
        if os.path.exists(venv_py):
            return venv_py
        # Fallback to current interpreter
        return sys.executable

    def _run_gateway(self, args: list, timeout: int = 120) -> dict:
        """Run a gateway.py subcommand and return parsed JSON output."""
        cmd = [self._py, self._gateway] + args
        try:
            r = subprocess.run(
                cmd, env=self._env, cwd=self._mt_dir,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout
            )
            if r.returncode != 0:
                return {"ok": False, "error": (r.stderr or r.stdout or "")[-200:]}
            # Try to parse JSON from stdout
            stdout = (r.stdout or "").strip()
            if stdout:
                try:
                    return json.loads(stdout.splitlines()[-1])
                except json.JSONDecodeError:
                    return {"ok": True, "raw": stdout[:200]}
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}

    def memorize(self, text: str, tags: str = "") -> dict:
        """Store a memory entry via gateway.py remember."""
        return self._run_gateway([
            "remember", text,
            "--type", "fact",
            "--source", self.source,
            "--tags", tags,
        ])

    def memorize_batch(self, texts: List[str]) -> List[dict]:
        """Store multiple memories."""
        results = []
        for t in texts:
            results.append(self.memorize(t))
        return results

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """Retrieve relevant memories via memsearch.search_hybrid."""
        k = top_k or self.retrieve_num
        r = self._run_gateway(["search", query, "--limit", str(k)], timeout=60)
        if not r.get("ok"):
            return []
        # gateway search returns {"results": [{"content": ..., "score": ...}]}
        results = r.get("results", [])
        return [item.get("content", "") for item in results if item.get("content")]

    def stats(self) -> dict:
        """Get MemTether instance statistics."""
        return self._run_gateway(["stats"])

    def cleanup(self):
        """Remove the isolated benchmark database."""
        if os.path.isdir(self._db_dir):
            shutil.rmtree(self._db_dir, ignore_errors=True)