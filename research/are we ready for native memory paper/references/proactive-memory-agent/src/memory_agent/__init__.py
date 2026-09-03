"""MemoryAgent — Reflective Memory for Terminal-Bench."""

import sys
from pathlib import Path

_external = Path(__file__).resolve().parent.parent.parent / "external"

# Add harbor to sys.path so its modules can be imported
_harbor_src = str(_external / "harbor" / "src")
if _harbor_src not in sys.path:
    sys.path.insert(0, _harbor_src)
