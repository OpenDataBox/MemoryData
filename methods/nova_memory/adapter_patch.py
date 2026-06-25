"""adapter_patch.py — inject NovaMemoryAgent into MemoryData/utils/agent.py.

Idempotent. Backs up utils/agent.py before patching.

Usage (from MemoryData root):
    python methods/nova_memory/adapter_patch.py
"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD_ROOT = HERE.parent.parent
AGENT_PY = MD_ROOT / "utils" / "agent.py"

if not AGENT_PY.exists():
    print(f"ERROR: {AGENT_PY} not found. Run from MemoryData root.")
    sys.exit(1)


HOOK_CODE = '''
    # ================== Nova Memory agent ==================
    # Injected by methods/nova_memory/adapter_patch.py
    def _is_nova_agent(self):
        return 'nova' in self.agent_name.lower()

    def _initialize_nova_agent(self, agent_config, dataset_config):
        """Initialize Nova lexical baseline agent."""
        import sys as _sys
        _src = str(_MD_ROOT) + "/methods/nova_memory/source".replace("/", __import__("os").sep)
        if _src not in _sys.path:
            _sys.path.insert(0, _src)
        from nova_agent import NovaMemoryAgent

        api_key = self.api_key or self._resolve_llm_api_key(["OPENAI_API_KEY"])
        base_url = self._resolve_base_url() if hasattr(self, "_resolve_base_url") else None

        self._nova_agent = NovaMemoryAgent(
            model=self.model,
            retrieve_num=agent_config.get('retrieve_num', 5),
            api_key=api_key,
            base_url=base_url,
            agent_save_to_folder=self.agent_save_to_folder,
            chunk_size=agent_config.get('agent_chunk_size', 4096),
        )
        self._nova_agent.load()
        self.retrieve_num = agent_config.get('retrieve_num', 5)
        self.context = ''

    def send_message(self, text, memorizing=False, **kwargs):
        """MemoryData calls this. If Nova, delegate to NovaMemoryAgent."""
        if getattr(self, '_nova_agent', None) is not None:
            return self._nova_agent.send_message(text, memorizing=memorizing)
        raise NotImplementedError(
            "send_message reached but no _nova_agent. Set agent_name containing 'nova'."
        )

'''


def already_patched(src: str) -> bool:
    return "_is_nova_agent" in src and "NovaMemoryAgent" in src


def find_agentwrapper_end(src: str) -> int:
    """Find byte offset where AgentWrapper class ends (next top-level def/class)."""
    m = re.search(r"^class AgentWrapper\b", src, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find 'class AgentWrapper'.")
    after_class_start = m.end()
    # Find next top-level def/class AFTER class AgentWrapper:
    # 'def ' or 'class ' at column 0, NOT inside the class body
    # Simplest: look for first '\nclass ' or '\ndef ' at start of line
    next_top = re.search(r"^(class |def )", src[after_class_start:], re.MULTILINE)
    if next_top:
        return after_class_start + next_top.start()
    return len(src)


def patch(src: str) -> str:
    # 1) Dispatch branch
    if "elif self._is_nova_agent" not in src:
        pattern = r"(elif self\._is_agent_type\(\"memagent\"\):\s*\n\s*self\._initialize_memagent\(agent_config\))"
        replacement = (
            r"\1\n"
            r"        elif self._is_nova_agent():\n"
            r"            self._initialize_nova_agent(agent_config, dataset_config)"
        )
        new_src, n = re.subn(pattern, replacement, src, count=1)
        if n == 0:
            raise RuntimeError(
                "Could not find memagent dispatch branch. MemoryData version mismatch?"
            )
        src = new_src

    # 2) Hook methods at END of AgentWrapper class
    if not already_patched(src):
        end = find_agentwrapper_end(src)
        hook = HOOK_CODE.replace("_MD_ROOT", repr(str(MD_ROOT)))
        src = src[:end] + hook + src[end:]

    return src


def main():
    src = AGENT_PY.read_text(encoding="utf-8")

    if already_patched(src):
        print(f"[OK] {AGENT_PY} already patched (idempotent).")
        return

    bak = AGENT_PY.with_suffix(
        f".py.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(AGENT_PY, bak)
    print(f"[BAK] {bak}")

    new_src = patch(src)
    AGENT_PY.write_text(new_src, encoding="utf-8")
    print(f"[OK]  Patched {AGENT_PY}")
    print("      + dispatch branch (elif nova in _initialize_agent_by_type)")
    print("      + _is_nova_agent / _initialize_nova_agent / send_message at end of AgentWrapper")
    print()
    print("Run a benchmark:")
    print("  python main.py --agent_config config/sequential_nova_memory.yaml \\")
    print("                  --dataset_config <benchmark yaml>")


if __name__ == "__main__":
    main()