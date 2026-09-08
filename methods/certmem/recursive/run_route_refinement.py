"""Query-free factual routes preserving the strongest original readable memory."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import refine as core
from evaluate_indexed import evaluate
from coverage_routes import harvest,geometry,construct
from retrieval_contract import assess,qualified,compare_contract
from crossview_probes import assess_view,require_fit
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/coverage_routes_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B');ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4);ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048);args=ap.parse_args()
    out=Path(args.out);previous=Path('runs/provenance_payload_v2')
    assert read(previous/'status.json')['phase']=='controlled_batch_complete'
    previous_protocol=read(previous/'protocol.json')
    for name,value in previous_protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    manifest=read('runs/pilot100_v3/manifest.json');records=[r for r in manifest['records'] if r['split']=='dev']
    cids=sorted({r['conv_id'] for r in records});samples=read('data/locomo10.json')
    assert core.digest(samples)==manifest['dataset_sha256']
    byid={str(s['sample_id']):s for s in samples if str(s['sample_id']) in cids};del samples
    sessions={cid:core.session_data(s) for cid,s in byid.items()}
    assert sessions==read('research_data/source_sessions.json')['sessions_by_id']
    assert not set(cids).intersection(manifest['holdout_convs'])
    baseline={cid:read(Path('runs/parent_evidence_v1/memories/s_parent_single_2000')/f'{cid}.json') for cid in cids}
    fit=read('runs/crossview_refinement_v1/query_views/fit_views.json');require_fit(fit);ids={r['id'] for r in fit}
    q0=sorted([r for r in qualified([read(p) for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json')]) if r['id'] in ids],key=lambda r:r['id'])
    assert {r['id'] for r in q0}==ids
    source_names=('run_route_refinement.py','coverage_routes.py','retrieval_contract.py','crossview_probes.py',
                  'budgeted_evidence.py','evidence_fidelity.py','evidence_utility.py','run_evidence_utility.py','evaluate_indexed.py','refine.py')
    hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_names}
    recipes=[{'name':f'ab_{pool}_{budget}','family':'source_factual_routes','pool':pool,'extra_budget':budget} for budget in (2000,4000) for pool in ('supported','dependent')]
    expected=.5927437370002835
    protocol={'args':vars(args),'source_sha256':hashes,'environment':read('environment.json'),
              'previous_protocol_sha256':core.digest(previous_protocol),'baseline_memory_sha256':core.digest(baseline),
              'records':records,'excluded_holdout_conversations':manifest['holdout_convs'],'reader':core.READER,
              'reference':{'name':'s_parent_single_2000','expected_f1':expected},'recipes':recipes,
              'source_q0_digest':core.digest(q0),'source_fit_view_digest':core.digest(fit),
              'construction':'Use supported statements from the completed source-only compiler; require every cited source ID to belong to a whole baseline unit. New route search key is the statement; readable payload is that baseline unit unchanged. Keep the entire original memory as an identical prefix.',
              'objective':'Query-free facility-location coverage of all eligible supported statement embeddings. Initial coverage is maximum cosine to existing memory search keys. Greedily choose positive marginal mean coverage per actual key+payload token, excluding repeated sources across added routes.',
              'controls':'Supported versus deletion-sensitive candidate pools, each at 2000/4000 added tokens per conversation. Both use the same all-supported coverage targets. No benchmark question or source probe question enters route construction.',
              'invariants':'Original memory texts, keys, unit order and reader are preserved. New factual routes are appended, so retrieval ranks and packing can change. Each added payload is an exact copy of an original unit; no new readable statement or date rule.',
              'budget':'Actual new key and copied readable payload both count. Storage cap equals baseline plus declared added tokens. Dense index array bytes are recorded separately.',
              'selection':'After every route memory is locked, assess matched 75 facts in Q0/A. Require positive net preserved answers in both; maximize worst gain, sum, smaller total storage, name; otherwise baseline. Lock before new dev evaluation.',
              'limits':'Facility location and factual query expansion are established methods; no novelty claim. Semantic coverage is a proxy, not answer utility. Source support and deletion are same-model judgments. No audit100/B outcomes used.'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    for path in (previous/'cache').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.npy'):
            dest=out/'cache'/path.relative_to(previous/'cache');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():os.link(path,dest)
    status={'pid':os.getpid(),'started_at':time.time()}
    def state(phase,**fields):core.save(out/'status.json',{**status,'phase':phase,'heartbeat_at':time.time(),**fields})
    state('model_load');rt=core.Runtime(args,protocol['environment'])
    replay=evaluate(rt,baseline,records,byid,2048,'reference_replay',out)
    assert abs(core.summarize(replay)['official_f1']-expected)<1e-12
    old=read(previous/'reference_replay_dev_items.json');keys=('id','question','context','prediction','source_ids','read_tokens','memory_tokens','official_f1')
    assert [{k:r[k] for k in keys} for r in replay]==[{k:r[k] for k in keys} for r in old]
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':expected,'all_70_contexts_and_predictions_identical':True})
    source_payloads={cid:read(previous/'source_payloads'/f'{cid}.json') for cid in cids}
    source_lock=read(previous/'source_payloads_locked.json')
    assert core.digest(source_payloads)==source_lock['compiled_sha256']
    core.save(out/'source_payloads_locked.json',source_lock)
    candidates_by_id={};geometry_by_id={}
    for cid in cids:
        state('source_route_construction',conversation=cid)
        core.save(out/'base_memories'/f'{cid}.json',baseline[cid])
        core.save(out/'source_payloads'/f'{cid}.json',source_payloads[cid])
        base_vectors=rt.encode([u.get('index_text',u['text']) for u in baseline[cid]])
        path=out/'indexes/baseline'/f'{cid}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,base_vectors)
        candidates=harvest(baseline[cid],source_payloads[cid],rt.ntok)
        fact_vectors=rt.encode([c['unit']['index_text'] for c in candidates]) if candidates else np.zeros((0,base_vectors.shape[1]),dtype=np.float32)
        geom=geometry(fact_vectors,base_vectors)
        candidates_by_id[cid]=candidates;geometry_by_id[cid]=geom
        core.save(out/'route_candidates'/f'{cid}.json',candidates)
        path=out/'route_geometry'/f'{cid}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**geom)
        print('ROUTE_CANDIDATES '+str({'conv_id':cid,'supported':len(candidates),'dependent':sum(c['deletion_sensitive'] for c in candidates)}),flush=True)
    core.save(out/'routes_locked.json',{'locked_at':time.time(),'candidates_sha256':core.digest(candidates_by_id),'used_probe_questions':False,'used_benchmark_questions':False})
    memories={}
    for recipe in recipes:
        name=recipe['name'];memory={};state('construction',round=name)
        core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':hashes,'committed_at':time.time()})
        for cid in cids:
            memory[cid],details=construct(baseline[cid],candidates_by_id[cid],geometry_by_id[cid],rt.ntok,recipe['extra_budget'],recipe['pool'])
            vectors=rt.encode([u.get('index_text',u['text']) for u in memory[cid]])
            path=out/'indexes'/name/f'{cid}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,vectors)
            details['dense_index_bytes']=vectors.nbytes
            core.save(out/'memories'/name/f'{cid}.json',memory[cid]);core.save(out/'construction'/name/f'{cid}.json',details)
        memories[name]=memory
    core.save(out/'route_memories_locked.json',{'locked_at':time.time(),'memory_sha256':{n:core.digest(m) for n,m in memories.items()},'baseline_prefix_identical':True})
    state('source_baseline_assessment');baseq=assess(rt,baseline,sessions,q0,'source_baseline',out);basea=assess_view(rt,baseline,sessions,fit,'fit_baseline',out)
    core.save(out/'source_baseline_contract.json',{'q0':baseq,'fit_a':basea});feasible=[]
    for recipe in recipes:
        name=recipe['name'];state('source_route_assessment',round=name)
        cq=assess(rt,memories[name],sessions,q0,name,out);ca=assess_view(rt,memories[name],sessions,fit,name,out)
        dq,da=compare_contract(baseq,cq),compare_contract(basea,ca);nq,na=[len(d['repairs'])-len(d['regressions']) for d in (dq,da)]
        storage=sum(read(out/'construction'/name/f'{cid}.json')['storage_tokens'] for cid in cids)
        result={'name':name,'q0':dq,'fit_a':da,'net_q0':nq,'net_a':na,'worst_view_gain':min(nq,na),'total_storage':storage,'feasible':nq>0 and na>0}
        core.save(out/'source_trials'/f'{name}.json',{'result':result,'q0_contract':cq,'fit_a_contract':ca})
        if result['feasible']:feasible.append(result)
        print('ROUTE_ASSESS '+str(result),flush=True)
    chosen=max(feasible,key=lambda r:(r['worst_view_gain'],r['net_q0']+r['net_a'],-r['total_storage'],r['name'])) if feasible else None
    selected=chosen['name'] if chosen else 's_parent_single_2000'
    core.save(out/'source_selection_locked.json',{'locked_at':time.time(),'selected':selected,'result':chosen,
              'memory_sha256':core.digest(memories[selected] if chosen else baseline),'selection_used_new_dev_outcomes':False,'selection_used_audit_outcomes':False})
    history={'rounds':[],'best_f1':expected,'winner':'s_parent_single_2000'}
    for recipe in recipes:
        name=recipe['name'];state('dev_evaluation',round=name)
        values=evaluate(rt,memories[name],records,byid,2048,name,out);summary=core.summarize(values)
        accepted=summary['official_f1']>history['best_f1']+.001
        history['rounds'].append({'name':name,'recipe':recipe,'dev':summary,'accepted':accepted})
        if accepted:history['winner'],history['best_f1']=name,summary['official_f1']
        core.save(out/'history.json',history);print('ROUTE_DEV '+str({'name':name,'f1':summary['official_f1'],'accepted':accepted}),flush=True)
    state('controlled_batch_complete',winner=history['winner'],best_f1=history['best_f1'],next='Continue methodological source/dev refinement; audit100 remains a consumed fixed checkpoint.')


if __name__=='__main__':main()
