"""Recursively refine using source-only retrieval feedback; dev is evaluated last."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import refine as core
from evaluate_indexed import evaluate
from budgeted_evidence import storage_cost
from retrieval_contract import qualified,assess,compare_contract,propose
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/retrieval_contract_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    out=Path(args.out)
    parent=Path('runs/parent_evidence_v1')
    assert read(parent/'status.json')['phase']=='controlled_batch_complete'
    parent_protocol=read(parent/'protocol.json')
    for name,value in parent_protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    parent_history=read(parent/'history.json')
    winner=parent_history['winner']
    expected=parent_history['best_f1']
    manifest=read('runs/pilot100_v3/manifest.json')
    records=[r for r in manifest['records'] if r['split']=='dev']
    cids={r['conv_id'] for r in records}
    samples=read('data/locomo10.json')
    assert core.digest(samples)==manifest['dataset_sha256']
    byid={str(s['sample_id']):s for s in samples if str(s['sample_id']) in cids}
    del samples
    sessions={cid:core.session_data(sample) for cid,sample in byid.items()}
    assert sessions==read('research_data/source_sessions.json')['sessions_by_id']
    assert not cids.intersection(manifest['holdout_convs'])
    fit=qualified([read(p) for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json')])
    baseline={cid:read(parent/'memories'/winner/f'{cid}.json') for cid in sorted(cids)}
    options={}
    checks={}
    for cid in sorted(cids):
        data=read(parent/'options'/f'{cid}.json')
        options[cid]=[[data['mapped'][o['id']] for o in group if o['id'] in data['mapped']] for group in data['groups']]
        checks[cid]=read(parent/'verification'/f'{cid}.json')
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('run_retrieval_contract.py','retrieval_contract.py','budgeted_evidence.py','evidence_fidelity.py',
                         'evidence_utility.py','run_evidence_utility.py','evaluate_indexed.py','refine.py')}
    recipe={'name':'t_source_selected','family':'retrieval_feedback_no_observed_source_regression',
            'global_extra_budget':4000,'max_source_rounds':3,'source_step_budgets':[1000,2000],
            'source_families':['single','joint','joint_verified']}
    protocol={'args':vars(args),'source_sha256':source,'environment':read('environment.json'),
              'parent_protocol_sha256':core.digest(parent_protocol),'parent_memory_sha256':core.digest(baseline),
              'source_probe_digest':core.digest(sorted(fit,key=lambda r:r['id'])),'records':records,
              'excluded_holdout_conversations':manifest['holdout_convs'],'reader':core.READER,'recipes':[recipe],
              'reference':{'name':winner,'expected_f1':expected},
              'selection':'Within up to 3 source-only rounds, propose cues for failed source contracts; assess actual retrieval+reader; reject any observed regression; maximize repairs then minimize storage. Freeze source selection before new dev evaluation.',
              'limits':'No-regression is only on observed source-fit behavior judged by the same model, not a guarantee on unseen questions. Full original memory remains; additional storage capped at 4000 tokens per conversation.'}
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    for path in (parent/'cache').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.npy'):
            target=out/'cache'/path.relative_to(parent/'cache')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                os.link(path,target)
    status={'pid':os.getpid(),'started_at':time.time()}
    def state(phase,**fields):
        core.save(out/'status.json',{**status,'phase':phase,'heartbeat_at':time.time(),**fields})
    state('model_load')
    rt=core.Runtime(args,protocol['environment'])
    replay=evaluate(rt,baseline,records,byid,args.budget,'reference_replay',out)
    assert abs(core.summarize(replay)['official_f1']-expected)<1e-12
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':expected})
    state('source_baseline_assessment')
    current=baseline
    assessment=assess(rt,current,sessions,fit,'source_baseline',out)
    core.save(out/'source_baseline_contract.json',assessment)
    caps={cid:sum(storage_cost(u,rt.ntok) for u in baseline[cid])+4000 for cid in cids}
    source_history=[]
    for iteration in range(3):
        candidates=[]
        for family in ('single','joint','joint_verified'):
            for budget in (1000,2000):
                name=f'iter{iteration}_{family}_{budget}'
                state('source_counterfactual_evaluation',iteration=iteration,candidate=name)
                memory={}
                for cid in sorted(cids):
                    memory[cid],_=propose(current[cid],options[cid],checks[cid],assessment,rt.ntok,
                                          family,budget,caps[cid])
                if core.digest(memory)==core.digest(current):
                    continue
                trial=assess(rt,memory,sessions,fit,name,out)
                comparison=compare_contract(assessment,trial)
                cost=sum(sum(storage_cost(u,rt.ntok) for u in units) for units in memory.values())
                result={'name':name,'family':family,'step_budget':budget,'total_storage':cost,**comparison}
                core.save(out/'source_trials'/f'{name}.json',{'result':result,'contract':trial})
                print('SOURCE_COUNTERFACTUAL '+str({'name':name,'repairs':len(comparison['repairs']),
                       'regressions':len(comparison['regressions']),'feasible':comparison['feasible']}),flush=True)
                if comparison['feasible']:
                    candidates.append((result,memory,trial))
        if not candidates:
            source_history.append({'iteration':iteration,'accepted':False,'reason':'No observed source-safe repair within the frozen proposal family and storage cap'})
            break
        chosen,memory,trial=max(candidates,key=lambda x:(len(x[0]['repairs']),-x[0]['total_storage'],x[0]['name']))
        source_history.append({'iteration':iteration,'accepted':True,**chosen})
        current,assessment=memory,trial
        core.save(out/'source_history.json',source_history)
    core.save(out/'source_history.json',source_history)
    core.save(out/'source_selection_locked.json',{'locked_at':time.time(),'source_history':source_history,
               'memory_sha256':core.digest(current),'source_preserved':sum(v['preserved'] for v in assessment.values()),
               'source_probes':len(fit),'selection_used_benchmark_answers':False})
    name=recipe['name']
    core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':source,'committed_at':time.time()})
    for cid in sorted(cids):
        core.save(out/'memories'/name/f'{cid}.json',current[cid])
        cost=sum(storage_cost(u,rt.ntok) for u in current[cid])
        core.save(out/'construction'/name/f'{cid}.json',{'storage_tokens':cost,'storage_cap':caps[cid]})
    state('dev_evaluation',round=name)
    evaluated=evaluate(rt,current,records,byid,args.budget,name,out)
    summary=core.summarize(evaluated)
    accepted=summary['official_f1']>expected+0.001
    history={'rounds':[{'name':name,'recipe':recipe,'dev':summary,'accepted':accepted}],
             'best_f1':summary['official_f1'] if accepted else expected,'winner':name if accepted else winner}
    core.save(out/'history.json',history)
    state('controlled_batch_complete',winner=history['winner'],best_f1=history['best_f1'],
          next='Continue the active refinement goal; do not use the sealed new-question audit for tuning.')


if __name__=='__main__':
    main()
