"""Check the actual GPU stack and tokenizer before the expensive writer run."""
import json
from pathlib import Path
import torch
import vllm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

root = Path('/workspace/.hf_home/hub')
def snapshot(name):
    base = root / ('models--' + name.replace('/', '--'))
    return base / 'snapshots' / (base/'refs/main').read_text().strip()

embed = SentenceTransformer(str(snapshot('Qwen/Qwen3-Embedding-0.6B')),device='cuda')
vectors = embed.encode(['Alice visited Paris.', 'Bob works in London.'],normalize_embeddings=True)
tok = AutoTokenizer.from_pretrained(str(snapshot('Qwen/Qwen3-8B')))
prompt = tok.apply_chat_template([{'role':'user','content':'Reply hello.'}],tokenize=False,
                                add_generation_prompt=True,enable_thinking=False)
assert vectors.shape[0] == 2 and vectors.shape[1] > 0
assert '</think>' in prompt
result = {'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__, 'cuda':torch.version.cuda,
          'vllm':vllm.__version__, 'embedding_shape':list(vectors.shape),
          'gpu_memory_allocated_mb':torch.cuda.memory_allocated()/1024**2,
          'thinking_disabled_template':True}
Path('gpu_smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
