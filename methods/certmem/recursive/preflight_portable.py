"""CPU-only exact reconstruction gate; safe while the single GPU is occupied."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

from transformers import AutoTokenizer
import refine as core
from portable_parent import backend,augment
from run_evidence_utility import read


def main():
    environment=read('environment.json')
    tokenizer=AutoTokenizer.from_pretrained(environment['models']['Qwen/Qwen3-8B']['path'])
    rt=SimpleNamespace(ntok=lru_cache(maxsize=100000)(lambda text:len(tokenizer.encode(text,add_special_tokens=False))))
    manifest=read('runs/pilot100_v3/manifest.json')
    cids={r['conv_id'] for r in manifest['records'] if r['split']=='dev'}
    samples=read('data/locomo10.json')
    assert core.digest(samples)==manifest['dataset_sha256']
    sessions={str(s['sample_id']):core.session_data(s) for s in samples if str(s['sample_id']) in cids}
    del samples
    plans={name:read(Path('runs/continuous_v3/plans')/f'{name}.json') for name in
           ('r06_calendar_month','r12_filter_current_best','r40_fused_four_turn')}
    rows=defaultdict(list)
    for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json'):
        r=read(p)
        rows[r['conv_id']].append(r)
    for cid in sorted(cids):
        seed=read(Path('runs/pilot100_v3/memories/r03_dialogue_residual')/f'{cid}.json')
        base=backend(rt,sessions[cid],seed,plans)
        assert base==read(Path('runs/continuous_v3/memories/r40_fused_four_turn')/f'{cid}.json'),cid
        memory,_,_=augment(rt,sessions[cid],base,rows[cid])
        assert memory==read(Path('runs/parent_evidence_v1/memories/s_parent_single_2000')/f'{cid}.json'),cid
        print('PORTABLE_EXACT_MEMORY_GATE '+cid,flush=True)
    print('ALL_7_PARENTS_RECONSTRUCTED_EXACTLY',flush=True)


if __name__=='__main__':
    main()
