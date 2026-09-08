"""Source-only payload construction with fixed retrieval keys and deletion controls."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import refine as core
from evaluate_indexed import evaluate
from provenance_payload_v2 import compile_source,construct,WRITER,CHECKER,SCHEMA
from retrieval_contract import assess,qualified,compare_contract
from crossview_probes import assess_view,require_fit
from run_evidence_utility import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/provenance_payload_v2')
    ap.add_argument('--model',default='Qwen/Qwen3-8B');ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4);ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048);args=ap.parse_args()
    out=Path(args.out);previous=Path('runs/transfer_index_v1')
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
    source_names=('run_payload_refinement_v2.py','provenance_payload_v2.py','provenance_payload.py','contrastive_events.py','retrieval_contract.py','crossview_probes.py',
                  'budgeted_evidence.py','evidence_fidelity.py','evidence_utility.py','run_evidence_utility.py','evaluate_indexed.py','refine.py')
    hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in source_names}
    recipes=[{'name':'aa_'+mode,'family':'source_grounded_payload','mode':mode} for mode in ('raw','normalized','supported','dependent')]
    expected=.5927437370002835
    protocol={'args':vars(args),'source_sha256':hashes,'environment':read('environment.json'),
              'previous_protocol_sha256':core.digest(previous_protocol),'baseline_memory_sha256':core.digest(baseline),
              'records':records,'excluded_holdout_conversations':manifest['holdout_convs'],'reader':core.READER,
              'reference':{'name':'s_parent_single_2000','expected_f1':expected},'recipes':recipes,
              'source_q0_digest':core.digest(q0),'source_fit_view_digest':core.digest(fit),'writer':WRITER,'checker':CHECKER,'schema':SCHEMA,
              'construction':'Generate at most four statements from each unique source-ID focus plus up to two neighboring turns on either side under a 2048-source-token context cap. Every statement has exact literal source quotes and at least one focus citation. Support judge sees cited original turns only; deletion judge sees window minus those cited turns.',
              'generation_retry':'Reuse complete 384-token initial output; retry only incomplete JSON with unchanged source/prompt/schema at 768 then 1536 tokens. Preserve every attempt and reject any remaining incomplete output.',
              'controls':'raw original focus; all literal-grounded statements; cited-source-supported statements; supported statements whose evidence-deleted context is INSUFFICIENT. All preserve original focus dialogue, unless it exceeds old unit token count, when all controls retain the original payload and report fallback.',
              'invariants':'Search index strings and unit order/count identical to strongest baseline, so BM25 and dense ranking unchanged. Reader and packing algorithm fixed; packed unit membership can change with payload length. New statement source IDs may add neighboring evidence provenance.',
              'budget':'Per-unit readable payload no longer than old payload; both fixed search key and changed payload counted. Global storage at most twice old storage, not storage-matched to original baseline. Exact dialogue is not truncated.',
              'selection':'After source-only statements and all four memories are locked, assess matched 75 facts in Q0/A. Require positive net preserved answers in both; maximize worst gain, sum, smaller total storage, name; otherwise baseline. Lock before new dev evaluation.',
              'limits':'Literal quotes do not prove entailment. Same-model support/deletion judgments are not truth or causality certification; repeated true evidence may fail deletion sensitivity. No novelty claim for contextualization, citations or counterfactual deletion. No audit100/B outcomes used.'}
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    superseded=Path('runs/provenance_payload_v1')
    assert read(superseded/'protocol_superseded.json')['reason']=='incomplete_structured_output_length'
    for cache_parent in (previous,superseded):
        for path in (cache_parent/'cache').rglob('*'):
            if path.is_file() and path.suffix in ('.json','.npy'):
                dest=out/'cache'/path.relative_to(cache_parent/'cache');dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():os.link(path,dest)
    status={'pid':os.getpid(),'started_at':time.time()}
    def state(phase,**fields):core.save(out/'status.json',{**status,'phase':phase,'heartbeat_at':time.time(),**fields})
    state('model_load');rt=core.Runtime(args,protocol['environment'])
    replay=evaluate(rt,baseline,records,byid,2048,'reference_replay',out)
    assert abs(core.summarize(replay)['official_f1']-expected)<1e-12
    old=read(previous/'reference_replay_dev_items.json');keys=('id','question','context','prediction','source_ids','read_tokens','memory_tokens','official_f1')
    assert [{k:r[k] for k in keys} for r in replay]==[{k:r[k] for k in keys} for r in old]
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':expected,'all_70_contexts_and_predictions_identical':True})
    compiled={}
    for cid in cids:
        core.save(out/'base_memories'/f'{cid}.json',baseline[cid])
        path=out/'indexes/baseline'/f'{cid}.npy';path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,rt.encode([u.get('index_text',u['text']) for u in baseline[cid]]))
        state('source_payload_compilation',conversation=cid)
        path=out/'source_payloads'/f'{cid}.json'
        if path.exists():compiled[cid]=read(path)
        else:
            compiled[cid]=compile_source(rt,sessions[cid],baseline[cid]);core.save(path,compiled[cid])
        facts=[f for item in compiled[cid].values() for f in item['facts']]
        print('PAYLOAD_SOURCE '+str({'conv_id':cid,'groups':len(compiled[cid]),'literal_statements':len(facts),
              'supported':sum(f['supported'] for f in facts),'deletion_sensitive':sum(f['deletion_sensitive'] for f in facts)}),flush=True)
    core.save(out/'source_payloads_locked.json',{'locked_at':time.time(),'compiled_sha256':core.digest(compiled),
              'used_probe_questions':False,'used_benchmark_questions':False})
    memories={}
    for recipe in recipes:
        name=recipe['name'];memory={};state('construction',round=name)
        core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':hashes,'committed_at':time.time()})
        for cid in cids:
            memory[cid],details=construct(baseline[cid],sessions[cid],compiled[cid],rt.ntok,recipe['mode'])
            core.save(out/'memories'/name/f'{cid}.json',memory[cid]);core.save(out/'construction'/name/f'{cid}.json',details)
        memories[name]=memory
    core.save(out/'payload_memories_locked.json',{'locked_at':time.time(),'memory_sha256':{n:core.digest(m) for n,m in memories.items()},
              'all_search_keys_fixed':True})
    state('source_baseline_assessment');baseq=assess(rt,baseline,sessions,q0,'source_baseline',out);basea=assess_view(rt,baseline,sessions,fit,'fit_baseline',out)
    core.save(out/'source_baseline_contract.json',{'q0':baseq,'fit_a':basea});feasible=[]
    for recipe in recipes:
        name=recipe['name'];state('source_payload_assessment',round=name)
        cq=assess(rt,memories[name],sessions,q0,name,out);ca=assess_view(rt,memories[name],sessions,fit,name,out)
        dq,da=compare_contract(baseq,cq),compare_contract(basea,ca);nq,na=[len(d['repairs'])-len(d['regressions']) for d in (dq,da)]
        storage=sum(read(out/'construction'/name/f'{cid}.json')['storage_tokens'] for cid in cids)
        result={'name':name,'q0':dq,'fit_a':da,'net_q0':nq,'net_a':na,'worst_view_gain':min(nq,na),'total_storage':storage,'feasible':nq>0 and na>0}
        core.save(out/'source_trials'/f'{name}.json',{'result':result,'q0_contract':cq,'fit_a_contract':ca})
        if result['feasible']:feasible.append(result)
        print('PAYLOAD_ASSESS '+str(result),flush=True)
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
        core.save(out/'history.json',history);print('PAYLOAD_DEV '+str({'name':name,'f1':summary['official_f1'],'accepted':accepted}),flush=True)
    state('controlled_batch_complete',winner=history['winner'],best_f1=history['best_f1'],next='Continue methodological source/dev refinement; audit100 remains a consumed fixed checkpoint.')


if __name__=='__main__':main()
