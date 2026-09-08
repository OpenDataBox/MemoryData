"""Predeclared evidence identity consolidation controls on three fixed parents."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import refine as core
from evaluate_indexed import evaluate
from identity_consolidation import consolidate
from budgeted_evidence import storage_cost
from retrieval_contract import assess,qualified,compare_contract
from crossview_probes import assess_view,require_fit
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/identity_refinement_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    out=Path(args.out)
    previous=Path('runs/crossview_refinement_v1')
    assert read(previous/'status.json')['phase']=='controlled_batch_complete'
    old_protocol=read(previous/'protocol.json')
    for name,value in old_protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    manifest=read('runs/pilot100_v3/manifest.json')
    records=[r for r in manifest['records'] if r['split']=='dev']
    cids={r['conv_id'] for r in records}
    samples=read('data/locomo10.json')
    assert core.digest(samples)==manifest['dataset_sha256']
    byid={str(s['sample_id']):s for s in samples if str(s['sample_id']) in cids}
    del samples
    sessions={cid:core.session_data(s) for cid,s in byid.items()}
    assert sessions==read('research_data/source_sessions.json')['sessions_by_id']
    assert not cids.intersection(manifest['holdout_convs'])
    source_rows=sorted(qualified([read(p) for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json')]),key=lambda r:r['id'])
    fit_views=read(previous/'query_views/fit_views.json')
    require_fit(fit_views)
    # Previously exposed audit B is not used in this construction or selection.
    parents_spec={'best':('parent_evidence_v1','s_parent_single_2000'),
                  'q0':('retrieval_contract_v1','t_source_selected'),
                  'view':('crossview_refinement_v1','v_crossview_selected')}
    parents={label:{cid:read(Path('runs')/run/'memories'/name/f'{cid}.json') for cid in sorted(cids)}
             for label,(run,name) in parents_spec.items()}
    expected=read(Path('runs/parent_evidence_v1/history.json'))['best_f1']
    recipes=[{'name':f'w_{label}_{mode}','parent':label,'family':'exact_evidence_identity_consolidation','mode':mode}
             for label in parents_spec for mode in ('payload','union','cues')]
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('run_identity_refinement.py','identity_consolidation.py','budgeted_evidence.py',
                         'retrieval_contract.py','crossview_probes.py','evidence_fidelity.py','evidence_utility.py',
                         'run_evidence_utility.py','evaluate_indexed.py','refine.py')}
    protocol={'args':vars(args),'source_sha256':source,'environment':read('environment.json'),
              'previous_protocol_sha256':core.digest(old_protocol),'parent_specs':parents_spec,
              'parent_memory_sha256':{k:core.digest(v) for k,v in parents.items()},
              'source_probe_digest':core.digest(source_rows),'fit_view_digest':core.digest(fit_views),
              'records':records,'excluded_holdout_conversations':manifest['holdout_convs'],
              'reader':core.READER,'recipes':recipes,'reference':{'name':'s_parent_single_2000','expected_f1':expected},
              'construction':'Exact payload+source-ID+session equivalence classes; coalesce to one read unit. Three key ablations: payload, all original keys union, question cues only. No new generated strings or QA-specific conditions.',
              'selection':'Before new dev evaluation, compare all candidates to best on both original source-fit queries and fit view A. Require no observed regressions on either set and at least one repair; maximize total repairs then minimize storage then name. If none, keep best.',
              'limits':'Pure construction changes, frozen retriever and packer. Fixed three parents were selected or diagnosed in earlier development, including exposed view audit B. B is not reused as a sealed audit. No fresh benchmark audit100 or source session audit used. All nine dev ablations are reported after the source choice lock; any dev winner is a development result.'}
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
    def state(phase,**fields):
        core.save(out/'status.json',{**status,'phase':phase,'heartbeat_at':time.time(),**fields})
    state('model_load')
    rt=core.Runtime(args,protocol['environment'])
    replay=evaluate(rt,parents['best'],records,byid,args.budget,'reference_replay',out)
    assert abs(core.summarize(replay)['official_f1']-expected)<1e-12
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':expected})
    state('source_baseline_assessment')
    base_q0=assess(rt,parents['best'],sessions,source_rows,'source_baseline',out)
    base_a=assess_view(rt,parents['best'],sessions,fit_views,'fit_baseline',out)
    core.save(out/'source_baseline_contract.json',{'q0':base_q0,'fit_a':base_a})
    candidates={}
    feasible=[]
    for recipe in recipes:
        name=recipe['name']
        state('construction',round=name)
        core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':source,'committed_at':time.time()})
        memories={}
        for cid in sorted(cids):
            memory,details=consolidate(parents[recipe['parent']][cid],rt.ntok,recipe['mode'])
            cap=sum(storage_cost(u,rt.ntok) for u in parents['best'][cid])+4000
            assert details['storage_tokens']<=cap
            details['storage_cap']=cap
            core.save(out/'memories'/name/f'{cid}.json',memory)
            core.save(out/'construction'/name/f'{cid}.json',details)
            memories[cid]=memory
        state('source_counterfactual_evaluation',round=name)
        q0=assess(rt,memories,sessions,source_rows,name,out)
        a=assess_view(rt,memories,sessions,fit_views,name,out)
        dq,da=compare_contract(base_q0,q0),compare_contract(base_a,a)
        repairs=len(dq['repairs'])+len(da['repairs'])
        cost=sum(sum(storage_cost(u,rt.ntok) for u in units) for units in memories.values())
        result={'name':name,'q0':dq,'fit_a':da,'total_repairs':repairs,'total_storage':cost,
                'feasible':not dq['regressions'] and not da['regressions'] and repairs>0}
        core.save(out/'source_trials'/f'{name}.json',{'result':result,'q0_contract':q0,'fit_a_contract':a})
        if result['feasible']:
            feasible.append(result)
        candidates[name]=memories
        print('IDENTITY_SOURCE '+str(result),flush=True)
    chosen=max(feasible,key=lambda r:(r['total_repairs'],-r['total_storage'],r['name'])) if feasible else None
    core.save(out/'source_selection_locked.json',{'locked_at':time.time(),'selected':chosen['name'] if chosen else 's_parent_single_2000',
              'result':chosen,'memory_sha256':core.digest(candidates[chosen['name']] if chosen else parents['best']),
              'selection_used_audit_views':False,'selection_used_new_dev_outcomes':False})
    history={'rounds':[],'best_f1':expected,'winner':'s_parent_single_2000'}
    for recipe in recipes:
        name=recipe['name']
        state('dev_evaluation',round=name)
        values=evaluate(rt,candidates[name],records,byid,args.budget,name,out)
        summary=core.summarize(values)
        accepted=summary['official_f1']>history['best_f1']+0.001
        history['rounds'].append({'name':name,'recipe':recipe,'dev':summary,'accepted':accepted})
        if accepted:
            history['winner'],history['best_f1']=name,summary['official_f1']
        core.save(out/'history.json',history)
        print('IDENTITY_DEV '+str({'name':name,'f1':summary['official_f1'],'accepted':accepted}),flush=True)
    state('controlled_batch_complete',winner=history['winner'],best_f1=history['best_f1'],
          next='Continue refinement; separate source-selected result from dev-selected best; goal active.')


if __name__=='__main__':
    main()
