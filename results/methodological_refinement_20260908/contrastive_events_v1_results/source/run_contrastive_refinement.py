"""Query-independent event distinctions, with locked two-view source choices."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import refine as core
from evaluate_indexed import evaluate
from contrastive_events import proposal_pairs,compile_candidates,construct,WRITER,VERIFIER,SCHEMA
from retrieval_contract import assess,qualified,compare_contract
from crossview_probes import assess_view,require_fit
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/contrastive_events_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    out=Path(args.out)
    previous=Path('runs/identity_refinement_v1')
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
    baseline={cid:read(Path('runs/parent_evidence_v1/memories/s_parent_single_2000')/f'{cid}.json') for cid in sorted(cids)}
    parent={cid:read(Path('runs/continuous_v3/memories/r40_fused_four_turn')/f'{cid}.json') for cid in sorted(cids)}
    fit=read('runs/crossview_refinement_v1/query_views/fit_views.json')
    require_fit(fit)
    ids={r['id'] for r in fit}
    q0=sorted([r for r in qualified([read(p) for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json')])
               if r['id'] in ids],key=lambda r:r['id'])
    assert {r['id'] for r in q0}==ids
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('run_contrastive_refinement.py','contrastive_events.py','retrieval_contract.py',
                         'crossview_probes.py','budgeted_evidence.py','evidence_fidelity.py','evidence_utility.py',
                         'run_evidence_utility.py','evaluate_indexed.py','refine.py')}
    precommit=read('contrastive_precommit.json')
    assert precommit['source_sha256']==source
    recipes=[{'name':f'x_{mode}_{budget}','family':'source_event_distinction','mode':mode,'extra_budget':budget}
             for budget in (2000,4000) for mode in ('literal','contrastive')]
    expected=.5927437370002835
    protocol={'args':vars(args),'source_sha256':source,'environment':read('environment.json'),
              'precommit':precommit,
              'previous_protocol_sha256':core.digest(old_protocol),'parent_memory_sha256':core.digest(parent),
              'baseline_memory_sha256':core.digest(baseline),'records':records,'excluded_holdout_conversations':manifest['holdout_convs'],
              'reader':core.READER,'recipes':recipes,'reference':{'name':'s_parent_single_2000','expected_f1':expected},
              'source_q0_digest':core.digest(q0),'source_fit_view_digest':core.digest(fit),'paired_source_fact_count':len(q0),
              'writer':WRITER,'verifier':VERIFIER,'schema':SCHEMA,
              'graph':{'maximum_pairs_per_conversation':64,'maximum_degree':2,
                       'proposal':'Descending positive embedding cosine between r40 units in different sessions with disjoint source IDs; deterministic tie order, degree limit; no QA input.'},
              'construction':'Writer and verifier receive original source turns with session dates, not generated backend facts or benchmark questions. Keep supported distinct-event keys. Under a common key+payload cost, greedy similarity/cost matching with no repeated source across new pairs selects identical evidence pair IDs for literal/contrastive controls.',
              'source_selection':'Matched 75 facts in Q0 and A. Main choice requires positive net preserved-answer count in BOTH views; maximize the smaller net gain, then total net gain, then smaller storage, then name. Individual regressions allowed and reported. Also lock a strict no-individual-regression choice as a diagnostic control. If none qualify, retain baseline. Both choices before any new dev result.',
              'limits':'No novelty claim from contrastive notes or event graphs alone. Same-model support judge. Structural hyperparameters are fixed generic assumptions. Existing baseline rules unchanged. No audit100 contents, predictions, scores or caches are inputs, and exposed B is not used.'}
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    for p in (previous/'cache').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.npy'):
            target=out/'cache'/p.relative_to(previous/'cache')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                os.link(p,target)
    status={'pid':os.getpid(),'started_at':time.time()}
    def state(phase,**fields):
        core.save(out/'status.json',{**status,'phase':phase,'heartbeat_at':time.time(),**fields})
    state('model_load')
    rt=core.Runtime(args,protocol['environment'])
    replay=evaluate(rt,baseline,records,byid,2048,'reference_replay',out)
    assert abs(core.summarize(replay)['official_f1']-expected)<1e-12
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':expected})
    graphs={}
    for cid in sorted(cids):
        state('source_event_graph',conversation=cid)
        path=out/'event_graph'/f'{cid}.json'
        if path.exists():
            graph=read(path)
        else:
            vectors=rt.encode([u['text'] for u in parent[cid]])
            edges=proposal_pairs(parent[cid],vectors,limit=64,max_degree=2)
            graph=compile_candidates(rt,sessions[cid],parent[cid],edges)
            core.save(path,graph)
        graphs[cid]=graph
        print('EVENT_GRAPH '+str({'conversation':cid,'proposed':len(graph['proposal_edges']),
              'distinct':len(graph['candidates']),'supported':sum(c['accepted'] for c in graph['candidates'])}),flush=True)
    # New candidate generation has completed without any probe question input.
    core.save(out/'event_graph_locked.json',{'locked_at':time.time(),'graphs_digest':core.digest(graphs),
              'generation_used_probe_questions':False,'generation_used_benchmark_qa':False})
    state('source_baseline_assessment')
    baseq=assess(rt,baseline,sessions,q0,'source_baseline',out)
    basea=assess_view(rt,baseline,sessions,fit,'fit_baseline',out)
    core.save(out/'source_baseline_contract.json',{'q0':baseq,'fit_a':basea})
    memories={}
    choices=[]
    strict=[]
    paired_ids={}
    for recipe in recipes:
        name=recipe['name']
        state('construction',round=name)
        core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':source,'committed_at':time.time()})
        memory={}
        total_storage=0
        for cid in sorted(cids):
            memory[cid],details=construct(baseline[cid],graphs[cid]['candidates'],rt.ntok,recipe['mode'],recipe['extra_budget'])
            key=(cid,recipe['extra_budget'])
            if key in paired_ids:
                assert paired_ids[key]==details['selected_ids']
            paired_ids[key]=details['selected_ids']
            core.save(out/'memories'/name/f'{cid}.json',memory[cid])
            core.save(out/'construction'/name/f'{cid}.json',details)
            total_storage+=details['storage_tokens']
        memories[name]=memory
        state('source_counterfactual_evaluation',round=name)
        cq=assess(rt,memory,sessions,q0,name,out)
        ca=assess_view(rt,memory,sessions,fit,name,out)
        dq,da=compare_contract(baseq,cq),compare_contract(basea,ca)
        netq,neta=[len(d['repairs'])-len(d['regressions']) for d in (dq,da)]
        result={'name':name,'q0':dq,'fit_a':da,'net_q0':netq,'net_a':neta,'worst_view_gain':min(netq,neta),
                'total_storage':total_storage,'worst_view_feasible':netq>0 and neta>0,
                'strict_feasible':not dq['regressions'] and not da['regressions'] and netq+neta>0}
        core.save(out/'source_trials'/f'{name}.json',{'result':result,'q0_contract':cq,'fit_a_contract':ca})
        if result['worst_view_feasible']:
            choices.append(result)
        if result['strict_feasible']:
            strict.append(result)
        print('EVENT_SOURCE '+str(result),flush=True)
    ranking=lambda r:(r['worst_view_gain'],r['net_q0']+r['net_a'],-r['total_storage'],r['name'])
    chosen=max(choices,key=ranking) if choices else None
    strict_choice=max(strict,key=ranking) if strict else None
    core.save(out/'source_selection_locked.json',{'locked_at':time.time(),
              'selected':chosen['name'] if chosen else 's_parent_single_2000','result':chosen,
              'strict_selected':strict_choice['name'] if strict_choice else 's_parent_single_2000','strict_result':strict_choice,
              'memory_sha256':core.digest(memories[chosen['name']] if chosen else baseline),
              'selection_used_audit_views':False,'selection_used_new_dev_outcomes':False})
    history={'rounds':[],'best_f1':expected,'winner':'s_parent_single_2000'}
    for recipe in recipes:
        name=recipe['name']
        state('dev_evaluation',round=name)
        values=evaluate(rt,memories[name],records,byid,2048,name,out)
        summary=core.summarize(values)
        accepted=summary['official_f1']>history['best_f1']+.001
        history['rounds'].append({'name':name,'recipe':recipe,'dev':summary,'accepted':accepted})
        if accepted:
            history['winner'],history['best_f1']=name,summary['official_f1']
        core.save(out/'history.json',history)
        print('EVENT_DEV '+str({'name':name,'f1':summary['official_f1'],'accepted':accepted}),flush=True)
    state('controlled_batch_complete',winner=history['winner'],best_f1=history['best_f1'],
          next='Continue methodological refinement on source/dev; do not use the separate frozen audit100 for tuning.')


if __name__=='__main__':
    main()
