"""Source-derived verified cue routing on the strongest frozen memory backend."""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import time

import refine as core
from evaluate_indexed import evaluate
from parent_evidence import make_options,verify_parent_options,construct
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/parent_evidence_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    out=Path(args.out)
    previous=Path('runs/budgeted_evidence_v1')
    assert read(previous/'status.json')['phase']=='controlled_batch_complete'
    previous_protocol=read(previous/'protocol.json')
    for name,value in previous_protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    manifest=read('runs/pilot100_v3/manifest.json')
    records=[r for r in manifest['records'] if r['split']=='dev']
    cids={r['conv_id'] for r in records}
    samples=read('data/locomo10.json')
    assert core.digest(samples)==manifest['dataset_sha256']
    byid={str(sample['sample_id']):sample for sample in samples if str(sample['sample_id']) in cids}
    del samples
    sessions={cid:core.session_data(sample) for cid,sample in byid.items()}
    assert not cids.intersection(manifest['holdout_convs'])
    corpus=read('research_data/source_sessions.json')
    assert corpus['sessions_by_id']==sessions
    rows=[read(p) for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json')]
    assert core.digest(sorted(rows,key=lambda r:r['id']))==previous_protocol['fit_result_digest']
    assert all(r['split']=='probe_fit' and r['conv_id'] in cids for r in rows)
    byconv=defaultdict(list)
    for row in rows:
        byconv[row['conv_id']].append(row)
    raw_verification=read(previous/'source_verification.json')
    reference_name='r40_fused_four_turn'
    reference_run=Path('runs/continuous_v3')
    baseline={cid:read(reference_run/'memories'/reference_name/f'{cid}.json') for cid in sorted(cids)}
    expected=read(previous/'reference_gate.json')['expected_f1']
    families=('raw_single','raw_joint','raw_joint_verified','parent_single','parent_joint',
              'parent_single_verified','parent_joint_verified')
    recipes=[{'name':f's_{family}_{extra}','family':family,'extra_budget':extra}
             for extra in (2000,4000) for family in families]
    environment=read('environment.json')
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('run_parent_refinement.py','parent_evidence.py','budgeted_evidence.py',
                         'evidence_fidelity.py','evidence_utility.py','run_evidence_utility.py','evaluate_indexed.py','refine.py')}
    protocol={'args':vars(args),'source_sha256':source,'environment':environment,'records':records,
              'excluded_holdout_conversations':manifest['holdout_convs'],'reader':core.READER,
              'previous_protocol_sha256':core.digest(previous_protocol),'parent_memory_sha256':core.digest(baseline),
              'recipes':recipes,'reference':{'name':reference_name,'expected_f1':expected},
              'construction':'Question-blind source-fit cues; original subsets or whole parent units covering their joint source IDs; source-only behavioral verification.',
              'budget':'Same 2000 or 4000 additional text tokens per conversation; actual spend reported; index and payload both counted.',
              'controls':'Strong backend already includes prior calendar/filter heuristics, frozen unchanged. New construction adds no date/keyword rules. Original-subset controls isolate the parent mapping.',
              'limitations':'Same-model verifier; likelihood proxy from original subsets may not equal utility after parent expansion; new actual generations and verification are measured. Development selection bias, not a generalization claim.'}
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    for path in (previous/'cache').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.npy'):
            target=out/'cache'/path.relative_to(previous/'cache')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                os.link(path,target)
    status={'pid':os.getpid(),'started_at':time.time()}
    core.save(out/'status.json',{**status,'phase':'model_load','heartbeat_at':time.time()})
    rt=core.Runtime(args,environment)
    replay=evaluate(rt,baseline,records,byid,args.budget,'reference_replay',out)
    assert abs(core.summarize(replay)['official_f1']-expected)<1e-12
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':expected})
    options_by_id={}
    checks_by_id={}
    for cid in sorted(cids):
        core.save(out/'status.json',{**status,'phase':'parent_evidence_verification','conversation':cid,'heartbeat_at':time.time()})
        path=out/'options'/f'{cid}.json'
        if path.exists():
            data=read(path)
            groups,mapped=data['groups'],data['mapped']
        else:
            groups,mapped=make_options(rt,sessions[cid],baseline[cid],byconv[cid])
            core.save(path,{'groups':groups,'mapped':mapped})
        check_path=out/'verification'/f'{cid}.json'
        checks=read(check_path) if check_path.exists() else verify_parent_options(rt,sessions[cid],byconv[cid],mapped)
        core.save(check_path,checks)
        options_by_id[cid]=(groups,mapped)
        checks_by_id[cid]=checks
    history=read(out/'history.json') if (out/'history.json').exists() else {'rounds':[],'best_f1':expected,'winner':reference_name}
    done={r['name'] for r in history['rounds']}
    for recipe in recipes:
        name=recipe['name']
        if name in done:
            continue
        core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':source,'committed_at':time.time()})
        memories={}
        for cid in sorted(cids):
            core.save(out/'status.json',{**status,'phase':'construction','round':name,'conversation':cid,'heartbeat_at':time.time()})
            groups,mapped=options_by_id[cid]
            memory,details=construct(baseline[cid],groups,mapped,checks_by_id[cid],raw_verification,
                                     rt.ntok,recipe['family'],recipe['extra_budget'])
            core.save(out/'memories'/name/f'{cid}.json',memory)
            core.save(out/'construction'/name/f'{cid}.json',details)
            memories[cid]=memory
        core.save(out/'status.json',{**status,'phase':'dev_evaluation','round':name,'heartbeat_at':time.time()})
        evaluated=evaluate(rt,memories,records,byid,args.budget,name,out)
        summary=core.summarize(evaluated)
        accepted=summary['official_f1']>history['best_f1']+0.001
        history['rounds'].append({'name':name,'recipe':recipe,'dev':summary,'accepted':accepted})
        if accepted:
            history['winner'],history['best_f1']=name,summary['official_f1']
        core.save(out/'history.json',history)
        print('PARENT_RESULT '+str({'name':name,'f1':summary['official_f1'],'memory_tokens':summary['memory_tokens'],'accepted':accepted}),flush=True)
    core.save(out/'status.json',{**status,'phase':'controlled_batch_complete','winner':history['winner'],
                                'best_f1':history['best_f1'],'heartbeat_at':time.time(),
                                'next':'Continue methodological refinement; goal active, no new-question audit used.'})


if __name__=='__main__':
    main()
