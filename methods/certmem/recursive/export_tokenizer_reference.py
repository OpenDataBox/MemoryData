"""Export the public frozen tokenizer for independent text-token recounts."""
import hashlib
from pathlib import Path
import tarfile
from transformers import AutoTokenizer
import refine as core
from run_evidence_utility import read

meta=read('environment.json')['models']['Qwen/Qwen3-8B']
out=Path('tokenizer_reference')
out.mkdir(exist_ok=True)
tokenizer=AutoTokenizer.from_pretrained(meta['path'])
tokenizer.save_pretrained(out)
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='reference_metadata.json'}
core.save(out/'reference_metadata.json',{'model_meta':meta,'source':'Frozen public Qwen/Qwen3-8B tokenizer, no model weights or credentials.',
                                      'file_sha256':files})
with tarfile.open('tokenizer_reference.tgz','w:gz') as archive:
    archive.add(out,arcname='tokenizer_reference')
print('TOKENIZER_REFERENCE_EXPORTED '+str(Path('tokenizer_reference.tgz').stat().st_size))
