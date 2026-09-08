"""Frozen source-only construction, then controlled dev-only evaluation."""
import argparse
from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import time

import refine as core
from evaluate_indexed import evaluate
from budgeted_evidence import construct,storage_cost
from evidence_fidelity import verify,VERIFIER
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/locomo10.json')
    ap.add_argument('--environment',default='environment.json')
    ap.add_argument('--out',default='runs/budgeted_evidence_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    out=Path(args.out)
    fit_run=Path('runs/evidence_utility_fit_v1')
    parent_run=Path('runs/continuous_v3')
    assert read(fit_run/'status.json')['phase']=='fit_complete'
    fit_protocol=read(fit_run/'protocol.json')
    for name,value in fit_protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    manifest=read('runs/pilot100_v3/manifest.json')
    records=[r for r in manifest['records'] if r['split']=='dev']
    cids={r['conv_id'] for r in records}
    samples=read(args.data)
    assert core.digest(samples)==manifest['dataset_sha256']
    byid={str(s['sample_id']):s for s in samples if str(s['sample_id']) in cids}
    del samples
    sessions={cid:core.session_data(sample) for cid,sample in byid.items()}
    corpus=read('research_data/source_sessions.json')
    assert corpus['sessions_by_id']==sessions
    assert not cids.intersection(manifest['holdout_convs'])
    fit_rows=[read(p) for p in (fit_run/'items').glob('*.json')]
    assert len(fit_rows)==len(fit_protocol['selected_ids'])
    assert {r['id'] for r in fit_rows}==set(fit_protocol['selected_ids'])
    assert all(r['split']=='probe_fit' and r['conv_id'] in cids for r in fit_rows)
    byconv=defaultdict(list)
    for row in fit_rows:
        byconv[row['conv_id']].append(row)
    environment=read(args.environment)
    reference_history=read(parent_run/'history.json')
    reference_name='r40_fused_four_turn'
    reference=next(r['dev'] for r in reference_history['rounds'] if r['name']==reference_name)
    recipes=[{'name':'b00_raw','mode':'raw','factor':1.0}]
    for factor in (1.25,1.5):
        for mode in ('full','single','joint','single_verified','joint_verified'):
            recipes.append({'name':f'b_{mode}_{round(factor*100)}','mode':mode,'factor':factor})
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('run_budgeted_refinement.py','budgeted_evidence.py','evidence_fidelity.py',
                         'evidence_utility.py','run_evidence_utility.py','evaluate_indexed.py','refine.py')}
    protocol={'args':vars(args),'source_sha256':source,'environment':environment,'records':records,
              'fit_protocol_sha256':core.digest(fit_protocol),'fit_result_digest':core.digest(sorted(fit_rows,key=lambda r:r['id'])),
              'recipes':recipes,'reader':core.READER,'verifier':VERIFIER,'excluded_holdout_conversations':manifest['holdout_convs'],
              'storage':'Shared per-conversation ceiling 1.25x or 1.5x raw text tokens; raw residual retained; distinct key and payload both counted. Actual spend need not be equal.',
              'candidate_pool':'Only already-generated source controls (single, pair, full proposals); identical pool for verified/unverified families, verifier filters candidates.',
              'constructor':'No benchmark questions/answers; source-fit likelihood utility with optional source-only semantic/behavior check; quantized multiple-choice budget DP.',
              'limitations':'Same-model self-verification, incomplete source-probe coverage, development selection bias. Not a factuality or generalization guarantee.'}
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    for path in (parent_run/'cache').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.npy'):
            target=out/'cache'/path.relative_to(parent_run/'cache')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                os.link(path,target)
    status={'pid':os.getpid(),'started_at':time.time()}
    core.save(out/'status.json',{**status,'phase':'model_load','heartbeat_at':time.time()})
    rt=core.Runtime(args,environment)
    # Reproduction precedes any new benchmark result. Original cache and legacy
    # evaluator keep this exactly comparable to the prior development experiment.
    reference_mem={cid:read(parent_run/'memories'/reference_name/f'{cid}.json') for cid in sorted(cids)}
    replay=evaluate(rt,reference_mem,records,byid,args.budget,'reference_replay',out)
    assert abs(core.summarize(replay)['official_f1']-reference['official_f1'])<1e-12
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':reference['official_f1']})
    core.save(out/'status.json',{**status,'phase':'source_fidelity_checks','heartbeat_at':time.time()})
    verification=read(out/'source_verification.json') if (out/'source_verification.json').exists() else verify(rt,sessions,fit_rows)
    core.save(out/'source_verification.json',verification)
    history=read(out/'history.json') if (out/'history.json').exists() else {'rounds':[],'best_f1':reference['official_f1'],'winner':reference_name}
    done={r['name'] for r in history['rounds']}
    for recipe in recipes:
        if recipe['name'] in done:
            continue
        name=recipe['name']
        core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':source,'committed_at':time.time()})
        memories={}
        for cid in sorted(cids):
            core.save(out/'status.json',{**status,'phase':'construction','round':name,'conversation':cid,'heartbeat_at':time.time()})
            baseline=core.raw_units(sessions[cid])
            basecost=sum(storage_cost(unit,rt.ntok) for unit in baseline)
            cap=math.floor(basecost*recipe['factor'])
            if recipe['mode']=='raw':
                memory,details=baseline,{'budget':cap,'storage_tokens':basecost}
            else:
                memory,details=construct(sessions[cid],byconv[cid],rt.ntok,recipe['mode'],cap,
                                         retain_raw=True,quantum=8,verification=verification,realized_only=True)
            assert sum(storage_cost(u,rt.ntok) for u in memory)<=cap
            core.save(out/'memories'/name/f'{cid}.json',memory)
            core.save(out/'construction'/name/f'{cid}.json',details)
            memories[cid]=memory
        core.save(out/'status.json',{**status,'phase':'dev_evaluation','round':name,'heartbeat_at':time.time()})
        rows=evaluate(rt,memories,records,byid,args.budget,name,out)
        summary=core.summarize(rows)
        accepted=summary['official_f1']>history['best_f1']+0.001
        history['rounds'].append({'name':name,'recipe':recipe,'dev':summary,'accepted':accepted})
        if accepted:
            history['winner'],history['best_f1']=name,summary['official_f1']
        core.save(out/'history.json',history)
        print('BUDGETED_RESULT '+str({'name':name,'f1':summary['official_f1'],'tokens':summary['memory_tokens'],'accepted':accepted}),flush=True)
    core.save(out/'status.json',{**status,'phase':'controlled_batch_complete','winner':history['winner'],
                                'best_f1':history['best_f1'],'heartbeat_at':time.time(),
                                'next':'Inspect methodological ablations and continue user refinement goal; no new-question audit has been opened.'})


if __name__=='__main__':
    main()
