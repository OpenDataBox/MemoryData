"""Source-only rigid memory alignment with cross-conversation transfer controls."""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import time
import numpy as np

import refine as core
from evaluate_indexed import evaluate as old_evaluate
from evaluate_vector_memory import evaluate
from source_index_learning import support_mask
from transfer_index_alignment import support_centroids,fit_rotation,apply_rotation,training_conversations
from assess_vector_memory import assess
from budgeted_evidence import storage_cost
from retrieval_contract import qualified,compare_contract
from crossview_probes import require_fit
from run_evidence_utility import read


def save_vectors(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        np.save(stream,values)
    tmp.replace(path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='runs/transfer_index_v1')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    out=Path(args.out)
    previous=Path('runs/source_index_v1')
    assert read(previous/'status.json')['phase']=='controlled_batch_complete'
    previous_protocol=read(previous/'protocol.json')
    for name,value in previous_protocol['source_sha256'].items():
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
    training=sorted(qualified([read(p) for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json')]),key=lambda r:r['id'])
    fit=read('runs/crossview_refinement_v1/query_views/fit_views.json')
    require_fit(fit)
    ids={r['id'] for r in fit}
    q0=[r for r in training if r['id'] in ids]
    assert {r['id'] for r in q0}==ids
    byconv=defaultdict(list)
    for row in training:
        byconv[row['conv_id']].append(row)
    sources=('run_transfer_refinement.py','transfer_index_alignment.py','source_index_learning.py','evaluate_vector_memory.py','assess_vector_memory.py',
             'refine.py','evaluate_indexed.py','budgeted_evidence.py','retrieval_contract.py','crossview_probes.py',
             'evidence_utility.py','evidence_fidelity.py','run_evidence_utility.py')
    hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
    recipes=[{'name':f'z_{scope}_{label}','family':'source_supervised_memory_index','method':'rigid_cayley_alignment','radius':radius,'scope':scope}
             for label,radius in [('010',.1),('030',.3)] for scope in ('within','transfer')]
    optimizer={'method':'Cayley rotation from the skew first-order support-to-query alignment direction',
               'supervision':'Each source query pairs with the unit-normalized centroid of source-ID-positive memory vectors.',
               'constraint':'One orthogonal map per target index; all document-document inner products preserved. Spectral movement bounded by radius. Identity on orthogonal complement of training span.',
               'scopes':'within uses target Q0; transfer excludes all target-conversation Q0 and uses only six other development conversations.',
               'parameters':'stored document vectors only; query/document encoders and reader weights are frozen; no iterative gradient training'}
    expected=.5927437370002835
    protocol={'args':vars(args),'source_sha256':hashes,'environment':read('environment.json'),
              'previous_protocol_sha256':core.digest(previous_protocol),'baseline_memory_sha256':core.digest(baseline),
              'records':records,'excluded_holdout_conversations':manifest['holdout_convs'],'reader':core.READER,
              'reference':{'name':'s_parent_single_2000','expected_f1':expected},'recipes':recipes,'optimizer':optimizer,
              'source_training_digest':core.digest(training),'source_q0_digest':core.digest(q0),'fit_view_digest':core.digest(fit),
              'selection':'After every index is aligned using original Q0 only under its declared within/transfer scope, assess matched 75 facts on Q0/A. Require positive net preserved answers in both views; maximize minimum gain, total gain, lower radius, name. Lock before new dev results; otherwise retain baseline.',
              'invariants':'Memory texts, sparse keys, sources, query encoder, rank-fusion formula, packing and reader unchanged; document-document geometry preserved. Stored dense vectors are the constructed state. Full original-context and prediction replay gate at the unmodified index.',
              'storage':'Text tokens held identical; report float32 dense-index array bytes separately, same shape and dtype for all candidates.',
              'limits':'Positive source overlap is weak supervision. Orthogonal alignment and Cayley maps are established methods, no novelty claim. Transfer excludes every target-conversation training query, but development conversations are not unseen globally. No audit100 or B outcomes enter fitting/selection.'}
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    for p in (previous/'cache').rglob('*'):
        if p.is_file() and p.suffix in ('.json','.npy'):
            dest=out/'cache'/p.relative_to(previous/'cache')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                os.link(p,dest)
    status={'pid':os.getpid(),'started_at':time.time()}
    def state(phase,**fields):
        core.save(out/'status.json',{**status,'phase':phase,'heartbeat_at':time.time(),**fields})
    state('model_load')
    rt=core.Runtime(args,protocol['environment'])
    original_vectors={}
    for cid in sorted(cids):
        original_vectors[cid]=rt.encode([u.get('index_text',u['text']) for u in baseline[cid]]).astype(np.float32)
        save_vectors(out/'indexes/baseline'/f'{cid}.npy',original_vectors[cid])
        core.save(out/'base_memories'/f'{cid}.json',baseline[cid])
    old=old_evaluate(rt,baseline,records,byid,2048,'original_reference',out)
    replay=evaluate(rt,baseline,original_vectors,records,byid,2048,'reference_replay',out)
    keys=('id','question','context','prediction','source_ids','read_tokens','memory_tokens','official_f1')
    assert [{k:r[k] for k in keys} for r in old]==[{k:r[k] for k in keys} for r in replay]
    assert abs(core.summarize(replay)['official_f1']-expected)<1e-12
    core.save(out/'reference_gate.json',{'passed':True,'expected_f1':expected,'all_70_contexts_and_predictions_identical':True})
    training_data={}
    for cid in sorted(cids):
        probes=byconv[cid]
        query_vectors=rt.encode([r['question'] for r in probes],query=True)
        positive=support_mask(baseline[cid],probes)
        training_data[cid]=(query_vectors,positive)
        p=out/'training'/f'{cid}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez(p,queries=query_vectors,positive=positive)
        core.save(out/'training'/f'{cid}.json',{'source_probe_ids':[r['id'] for r in probes],
                  'query_texts_sha256':core.digest([r['question'] for r in probes]),'uses_fit_view_a':False})
    vectors_by_name={}
    for recipe in recipes:
        name=recipe['name'];vectors={}
        core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':hashes,'committed_at':time.time()})
        for cid in sorted(cids):
            state('source_index_optimization',round=name,conversation=cid)
            train_cids=training_conversations(cid,sorted(cids),recipe['scope'])
            support=np.concatenate([support_centroids(original_vectors[k],training_data[k][1]) for k in train_cids])
            query=np.concatenate([training_data[k][0] for k in train_cids])
            transform,fit_details=fit_rotation(support,query,recipe['radius'])
            vectors[cid],apply_details=apply_rotation(original_vectors[cid],transform,recipe['radius'])
            details={**fit_details,**apply_details,'scope':recipe['scope'],'target_conversation':cid,
                     'training_conversations':train_cids,'training_probe_ids':[r['id'] for k in train_cids for r in byconv[k]],
                     'uses_target_questions':cid in train_cids}
            path=out/'transforms'/name/f'{cid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,**transform)
            assert vectors[cid].shape==original_vectors[cid].shape and vectors[cid].nbytes==original_vectors[cid].nbytes
            save_vectors(out/'indexes'/name/f'{cid}.npy',vectors[cid])
            core.save(out/'optimizer'/name/f'{cid}.json',details)
            core.save(out/'memories'/name/f'{cid}.json',baseline[cid])
            cost=sum(storage_cost(u,rt.ntok) for u in baseline[cid])
            core.save(out/'construction'/name/f'{cid}.json',{'storage_tokens':cost,'storage_cap':cost,
                       'dense_index_bytes':vectors[cid].nbytes,'text_memory_unchanged':True})
        vectors_by_name[name]=vectors
    core.save(out/'training_locked.json',{'locked_at':time.time(),'all_indexes_sha256':{
        name:{cid:hashlib.sha256((out/'indexes'/name/f'{cid}.npy').read_bytes()).hexdigest() for cid in sorted(cids)} for name in vectors_by_name},
        'gradient_training_used_only_q0':True,'fit_uses_only_q0':True,'no_iterative_gradient_training':True})
    state('source_baseline_assessment')
    baseq=assess(rt,baseline,original_vectors,sessions,q0,'source_baseline',out,'q0')
    basea=assess(rt,baseline,original_vectors,sessions,fit,'fit_baseline',out,'fit_a')
    core.save(out/'source_baseline_contract.json',{'q0':baseq,'fit_a':basea})
    feasible=[]
    for recipe in recipes:
        name=recipe['name'];state('source_index_assessment',round=name)
        cq=assess(rt,baseline,vectors_by_name[name],sessions,q0,name,out,'q0')
        ca=assess(rt,baseline,vectors_by_name[name],sessions,fit,name,out,'fit_a')
        dq,da=compare_contract(baseq,cq),compare_contract(basea,ca)
        nq,na=[len(d['repairs'])-len(d['regressions']) for d in (dq,da)]
        result={'name':name,'radius':recipe['radius'],'q0':dq,'fit_a':da,'net_q0':nq,'net_a':na,
                'worst_view_gain':min(nq,na),'feasible':nq>0 and na>0}
        core.save(out/'source_trials'/f'{name}.json',{'result':result,'q0_contract':cq,'fit_a_contract':ca})
        if result['feasible']:
            feasible.append(result)
        print('TRANSFER_SOURCE '+str(result),flush=True)
    chosen=max(feasible,key=lambda r:(r['worst_view_gain'],r['net_q0']+r['net_a'],-r['radius'],r['name'])) if feasible else None
    selected=chosen['name'] if chosen else 'baseline'
    core.save(out/'source_selection_locked.json',{'locked_at':time.time(),'selected':selected,'result':chosen,
        'selected_index_sha256':{cid:hashlib.sha256((out/'indexes'/selected/f'{cid}.npy').read_bytes()).hexdigest() for cid in sorted(cids)},
        'memory_texts_sha256':core.digest(baseline),'selection_used_new_dev_outcomes':False,'selection_used_audit_outcomes':False})
    history={'rounds':[],'best_f1':expected,'winner':'s_parent_single_2000'}
    for recipe in recipes:
        name=recipe['name'];state('dev_evaluation',round=name)
        values=evaluate(rt,baseline,vectors_by_name[name],records,byid,2048,name,out)
        summary=core.summarize(values)
        accepted=summary['official_f1']>history['best_f1']+.001
        history['rounds'].append({'name':name,'recipe':recipe,'dev':summary,'accepted':accepted})
        if accepted:
            history['winner'],history['best_f1']=name,summary['official_f1']
        core.save(out/'history.json',history)
        print('TRANSFER_DEV '+str({'name':name,'f1':summary['official_f1'],'accepted':accepted}),flush=True)
    state('controlled_batch_complete',winner=history['winner'],best_f1=history['best_f1'],
          next='Continue source/dev refinement; preserve the independent audit100 as a completed fixed checkpoint.')


if __name__=='__main__':
    main()
