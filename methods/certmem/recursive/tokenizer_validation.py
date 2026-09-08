"""CPU token recount using the exact exported reader tokenizer assets."""
import hashlib
import json
from functools import lru_cache
from tokenizers import Tokenizer


def load_counter(root):
    metadata=json.loads((root/'reference_metadata.json').read_text(encoding='utf-8'))
    for name,value in metadata['file_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value
    tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
    @lru_cache(maxsize=100000)
    def count(text):
        return len(tokenizer.encode(text,add_special_tokens=False).ids)
    return count,metadata
