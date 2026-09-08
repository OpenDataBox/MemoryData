"""Source-only recursive selection with disjoint query wording and a final view audit."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import refine as core
from evaluate_indexed import evaluate
from budgeted_evidence import storage_cost
from retrieval_contract import qualified,compare_contract,propose
from crossview_probes import prepare,assess_view,require_fit,fingerprint,SCHEMA,WRITER,QUESTION_CHECK
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/crossview_refinement_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    out=Path(args.out)
    parent=Path('runs/retrieval_contract_v1')
    assert read(parent/'status.json')['phase']=='controlled_batch_complete'
    parent_protocol=read(parent/'protocol.json')
    for name,value in parent_protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    strong=Path('runs/parent_evidence_v1')
    strong_history=read(strong/'history.json')
    winner,expected=strong_history['winner'],strong_history['best_f1']
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
    original=sorted(qualified([read(p) for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json')]),key=lambda r:r['id'])
    baseline={cid:read(strong/'memories'/winner/f'{cid}.json') for cid in sorted(cids)}
    previous_selected={cid:read(parent/'memories/t_source_selected'/f'{cid}.json') for cid in sorted(cids)}
    options={}
    checks={}
    for cid in sorted(cids):
        data=read(strong/'options'/f'{cid}.json')
        options[cid]=[[data['mapped'][o['id']] for o in group if o['id'] in data['mapped']] for group in data['groups']]
        checks[cid]=read(strong/'verification'/f'{cid}.json')
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('run_crossview_refinement.py','crossview_probes.py','retrieval_contract.py',
                         'budgeted_evidence.py','evidence_fidelity.py','evidence_utility.py',
                         'run_evidence_utility.py','evaluate_indexed.py','refine.py')}
    recipe={'name':'v_crossview_selected','family':'disjoint_query_view_source_feedback',
            'global_extra_budget':4000,'max_source_rounds':3,'source_step_budgets':[1000,2000],
            'source_families':['single','joint','joint_verified']}
    protocol={'args':vars(args),'source_sha256':source,'environment':read('environment.json'),
              'parent_protocol_sha256':core.digest(parent_protocol),'parent_memory_sha256':core.digest(baseline),
              'original_source_probe_digest':core.digest(original),'records':records,
              'excluded_holdout_conversations':manifest['holdout_convs'],'reader':core.READER,'recipes':[recipe],
              'reference':{'name':winner,'expected_f1':expected},'view_schema':SCHEMA,
              'view_writer':WRITER,'view_question_check':QUESTION_CHECK,
              'selection':'Same source-only no-observed-regression recursion as prior run, but fit questions use independent wording absent from every stored question cue. Audit query views evaluated only after source selection lock.',
              'limits':'Paraphrase audit tests other wording of the same source facts, judged by the same model; not unseen conversations. New benchmark question audit100 remains unopened. Full memory retained, +4000 token cap per conversation.'}
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
    state('source_query_view_generation_and_calibration')
    views=prepare(rt,sessions,original,out/'query_views')
    fit,audit=views['fit'],views['audit']
    require_fit(fit)
    ids={r['id'] for r in fit}
    assert ids=={r['id'] for r in audit}
    options={cid:[[o for o in group if o['unit']['source_probe_id'] in ids] for group in groups] for cid,groups in options.items()}
    forbidden={fingerprint(row['question']) for row in fit+audit}
    def no_query_copy(memory):
        assert not any(fingerprint(u['index_text']) in forbidden for units in memory.values() for u in units if 'index_text' in u)
    no_query_copy(baseline)
    for groups in options.values():
        assert not any(fingerprint(o['unit']['index_text']) in forbidden for group in groups for o in group)
    core.save(out/'query_view_lock.json',{'locked_at':time.time(),'fit_digest':core.digest(fit),
               'audit_digest':core.digest(audit),'paired_probes':len(fit),'no_query_view_copied_into_indexes':True})
    state('fit_view_baseline_assessment')
    current=baseline
    assessment=assess_view(rt,current,sessions,fit,'fit_baseline',out)
    core.save(out/'source_baseline_contract.json',assessment)
    state('fit_view_previous_method_diagnostic')
    previous_fit=assess_view(rt,previous_selected,sessions,fit,'fit_previous_method',out)
    core.save(out/'previous_method_fit_contract.json',previous_fit)
    caps={cid:sum(storage_cost(u,rt.ntok) for u in baseline[cid])+4000 for cid in cids}
    source_history=[]
    for iteration in range(3):
        candidates=[]
        for family in ('single','joint','joint_verified'):
            for budget in (1000,2000):
                name=f'iter{iteration}_{family}_{budget}'
                state('fit_view_counterfactual_evaluation',iteration=iteration,candidate=name)
                memory={}
                for cid in sorted(cids):
                    memory[cid],_=propose(current[cid],options[cid],checks[cid],assessment,rt.ntok,family,budget,caps[cid])
                no_query_copy(memory)
                if core.digest(memory)==core.digest(current):
                    continue
                trial=assess_view(rt,memory,sessions,fit,name,out)
                comparison=compare_contract(assessment,trial)
                cost=sum(sum(storage_cost(u,rt.ntok) for u in units) for units in memory.values())
                result={'name':name,'family':family,'step_budget':budget,'total_storage':cost,**comparison}
                core.save(out/'source_trials'/f'{name}.json',{'result':result,'contract':trial})
                print('CROSSVIEW_COUNTERFACTUAL '+str({'name':name,'repairs':len(comparison['repairs']),
                       'regressions':len(comparison['regressions']),'feasible':comparison['feasible']}),flush=True)
                if comparison['feasible']:
                    candidates.append((result,memory,trial))
        if not candidates:
            source_history.append({'iteration':iteration,'accepted':False,'reason':'No fit-view improvement without observed regression'})
            break
        chosen,memory,trial=max(candidates,key=lambda x:(len(x[0]['repairs']),-x[0]['total_storage'],x[0]['name']))
        source_history.append({'iteration':iteration,'accepted':True,**chosen})
        current,assessment=memory,trial
        core.save(out/'source_history.json',source_history)
    core.save(out/'source_history.json',source_history)
    lock=out/'source_selection_locked.json'
    core.save(lock,{'locked_at':time.time(),'source_history':source_history,'memory_sha256':core.digest(current),
                   'source_preserved':sum(v['preserved'] for v in assessment.values()),'source_probes':len(fit),
                   'selection_used_audit_views':False,'selection_used_benchmark_answers':False})
    name=recipe['name']
    core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':source,'committed_at':time.time()})
    for cid in sorted(cids):
        core.save(out/'memories'/name/f'{cid}.json',current[cid])
        core.save(out/'construction'/name/f'{cid}.json',{'storage_tokens':sum(storage_cost(u,rt.ntok) for u in current[cid]),'storage_cap':caps[cid]})
    state('dev_evaluation',round=name)
    evaluated=evaluate(rt,current,records,byid,args.budget,name,out)
    summary=core.summarize(evaluated)
    accepted=summary['official_f1']>expected+0.001
    history={'rounds':[{'name':name,'recipe':recipe,'dev':summary,'accepted':accepted}],
             'best_f1':summary['official_f1'] if accepted else expected,'winner':name if accepted else winner}
    core.save(out/'history.json',history)
    audit_summary={}
    for label,memory in [('baseline',baseline),('previous_method',previous_selected),('selected',current)]:
        state('locked_query_view_audit',memory=label)
        result=assess_view(rt,memory,sessions,audit,'audit_'+label,out,selection_lock=lock)
        core.save(out/'view_audit'/f'{label}.json',result)
        audit_summary[label]={'n':len(result),'preserved':sum(v['preserved'] for v in result.values())}
    core.save(out/'view_audit_summary.json',audit_summary)
    state('controlled_batch_complete',winner=history['winner'],best_f1=history['best_f1'],
          next='Continue methodological refinement; query-view audit is evaluation only; benchmark audit100 remains unopened.')


if __name__=='__main__':
    main()
