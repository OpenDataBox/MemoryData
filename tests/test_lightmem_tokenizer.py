import sys
from pathlib import Path


LIGHTMEM_SOURCE = Path(__file__).parents[1] / "methods" / "lightmem" / "source"
sys.path.insert(0, str(LIGHTMEM_SOURCE))

from lightmem.memory.utils import resolve_tokenizer


def test_resolve_tokenizer_accepts_tokenizer_protocol_without_transformers():
    class Tokenizer:
        def encode(self, text):
            return text.split()

    tokenizer = Tokenizer()

    assert resolve_tokenizer(tokenizer) is tokenizer
    assert "transformers" not in sys.modules
