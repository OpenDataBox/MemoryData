"""Reconstruct factual routes, coverage optimization and actual retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi
import refine as core
from coverage_routes import harvest,geometry,construct
from contrastive_events import original_context
from budgeted_evidence import storage_cost
from tokenizer_validation import load_counter
from verify_identity import changes


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def valid_json(value):return isinstance(value['object'],dict) and set(value['object'])=={'facts'} and isinstance(value['object']['facts'],list) and len(value['object']['facts'])<=4


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--run',default='coverage_routes_v1');args=ap.parse_args()
    root=args.root;run=root/args.run/'run';protocol=read(run/'protocol.json')
    for name,digest in protocol['source_sha256'].items():assert sha(run.parent/'source'/name)==digest
    ntok,metadata=load_counter(root/'tokenizer_reference/tokenizer_reference')
    model=protocol['environment']['models'][protocol['args']['model']];assert model==metadata['model_meta']
    cids=sorted({r['conv_id'] for r in protocol['records']})
    baseline={cid:read(run/'base_memories'/f'{cid}.json') for cid in cids}
    assert core.digest(baseline)==protocol['baseline_memory_sha256']
    assert baseline=={cid:read(root/'parent_evidence_v1/run/memories/s_parent_single_2000'/f'{cid}.json') for cid in cids}
    sessions=read(root/'source_sessions.json')['sessions_by_id']
    compiled={cid:read(run/'source_payloads'/f'{cid}.json') for cid in cids}
    previous=root/'provenance_payload_v2/run'
    previous_protocol=read(previous/'protocol.json')
    assert core.digest(previous_protocol)==protocol['previous_protocol_sha256']
    previous_validation=read(root/'PAYLOAD_SOURCE.validation.json')
    assert previous_validation['source_groups_reconstructed']==1862 and previous_validation['checker_prompt_cache_pairs_verified']==9486
    for name,digest in previous_protocol['source_sha256'].items():assert sha(previous.parent/'source'/name)==digest
    assert compiled=={cid:read(previous/'source_payloads'/f'{cid}.json') for cid in cids}
    compiler_lock=read(run/'source_payloads_locked.json')
    assert compiler_lock==read(previous/'source_payloads_locked.json')
    assert compiler_lock['compiled_sha256']==core.digest(compiled)
    candidates={};geometries={};geometry_error=0.
    for cid in cids:
        candidates[cid]=harvest(baseline[cid],compiled[cid],ntok)
        assert candidates[cid]==read(run/'route_candidates'/f'{cid}.json')
        geom=dict(np.load(run/'route_geometry'/f'{cid}.npz'))
        assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in geom.values())
        assert np.array_equal(geom['baseline_vectors'],np.load(run/'indexes/baseline'/f'{cid}.npy'))
        for key in ('fact_vectors','baseline_vectors'):assert np.max(np.abs(np.linalg.norm(geom[key],axis=1)-1))<2e-5
        replay=geometry(geom['fact_vectors'],geom['baseline_vectors'])
        for key in ('similarities','initial_coverage'):
            error=float(np.max(np.abs(replay[key]-geom[key])))
            assert error<1e-6;geometry_error=max(geometry_error,error)
        geometries[cid]=geom
    routes_lock=read(run/'routes_locked.json')
    assert routes_lock['candidates_sha256']==core.digest(candidates)
    assert not routes_lock['used_probe_questions'] and not routes_lock['used_benchmark_questions']
    memories={};costs={};construction={};index_vectors={'baseline':{}}
    for cid in cids:index_vectors['baseline'][cid]=np.load(run/'indexes/baseline'/f'{cid}.npy')
    for recipe in protocol['recipes']:
        name=recipe['name'];memories[name]={};costs[name]={};construction[name]={};index_vectors[name]={}
        for cid in cids:
            expected,details=construct(baseline[cid],candidates[cid],geometries[cid],ntok,recipe['extra_budget'],recipe['pool'])
            actual=read(run/'memories'/name/f'{cid}.json');assert actual==expected
            vectors=np.load(run/'indexes'/name/f'{cid}.npy');index_vectors[name][cid]=vectors
            assert vectors.dtype==np.float32 and vectors.shape==(len(actual),1024) and np.isfinite(vectors).all()
            assert np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))<2e-5
            assert np.max(np.abs(vectors[:len(baseline[cid])]-index_vectors['baseline'][cid]))<2e-5
            lookup={c['id']:i for i,c in enumerate(candidates[cid])}
            selected=[lookup[i] for i in details['selected_ids']]
            if selected:assert np.max(np.abs(vectors[len(baseline[cid]):]-geometries[cid]['fact_vectors'][selected]))<2e-5
            details['dense_index_bytes']=vectors.nbytes
            assert details==read(run/'construction'/name/f'{cid}.json')
            assert actual[:len(baseline[cid])]==baseline[cid]
            assert sum(storage_cost(u,ntok) for u in actual)==details['storage_tokens']<=details['storage_cap']
            for unit in actual[len(baseline[cid]):]:
                parent=baseline[cid][unit['parent_index']]
                assert all(unit[k]==parent[k] for k in ('text','sources','session'))
            memories[name][cid]=actual;costs[name][cid]=details['storage_tokens'];construction[name][cid]=details
        print(json.dumps({'verification':'coverage_greedy_and_memory_reconstructed','recipe':name}),flush=True)
    memory_lock=read(run/'route_memories_locked.json')
    assert memory_lock['memory_sha256']=={name:core.digest(m) for name,m in memories.items()} and memory_lock['baseline_prefix_identical']
    assert compiler_lock['locked_at']<=routes_lock['locked_at']<=memory_lock['locked_at']
    fit=read(root/'crossview_refinement_v1/run/query_views/fit_views.json');fit_ids={r['id'] for r in fit}
    q0=sorted([read(p) for p in (root/'evidence_utility_fit_v1/runs/evidence_utility_fit_v1/items').glob('*.json') if read(p)['id'] in fit_ids],key=lambda r:r['id'])
    assert core.digest(q0)==protocol['source_q0_digest'] and core.digest(fit)==protocol['source_fit_view_digest']
    assert len(q0)==len(fit)==75 and all(r['split']=='probe_fit' for r in q0+fit)
    query_maps={}
    for split,filename in [('dev','reference_replay_dev_items.json'),('q0','source_baseline_source_fit_items.json'),('fit_a','fit_baseline_source_view_fit_items.json')]:
        rows=read(run/filename);query_maps[split]={}
        for cid in cids:
            selected=[r for r in rows if r['conv_id']==cid];record=read(run/'evaluation_queries'/split/f'{cid}.json')
            assert record['ids']==[r['id'] for r in selected]
            assert record['cache_key']==core.digest([protocol['environment']['models'][protocol['args']['embed_model']],[r['question'] for r in selected],True,protocol['args']['embed_batch_size'],'oom_backoff_v1'])
            values=np.load(run/'evaluation_queries'/split/f'{cid}.npy');assert values.shape==(len(selected),1024)
            query_maps[split].update({r['id']:v for r,v in zip(selected,values)})
    bm25={name:{cid:BM25Okapi([core.lexical(u.get('index_text',u['text'])) or ['_empty'] for u in memory[cid]]) for cid in cids} for name,memory in {'baseline':baseline,**memories}.items()}
    def ranks(values):
        order=np.argsort(-values,kind='stable');r=np.empty(len(order),dtype=np.int64);r[order]=np.arange(len(order));return r
    checked=0;orders={}
    def check_rows(filename,memory,split,contract=None,index_name="baseline"):
        nonlocal checked
        rows=read(run/filename);cost={cid:sum(storage_cost(u,ntok) for u in memory[cid]) for cid in cids}
        assert len(rows)==(70 if split=='dev' else 75) and len({r['id'] for r in rows})==len(rows)
        if contract is not None:assert {r['id'] for r in rows}==contract.keys()==fit_ids
        for row in rows:
            cid=row['conv_id'];units=memory[cid];q=query_maps[split][row['id']]
            key=(index_name,split,row['id'])
            if key not in orders:
                dense=index_vectors[index_name][cid]@q;sparse=np.asarray(bm25[index_name][cid].get_scores(core.lexical(row['question'])))
                fused=1/(60+ranks(dense))+1/(60+ranks(sparse));orders[key]=np.argsort(-fused,kind='stable')[:min(len(units),120)]
            context,hits,tokens=core.pack(units,orders[key],ntok,2048)
            assert row['context']==context and row['read_tokens']==tokens==ntok(context)<=2048
            assert row['source_ids']==sorted({sid for i in hits for sid in units[i]['sources']})
            assert row['memory_tokens']==cost[cid] and row['memory_units']==len(units)
            assert abs(core.f1(row['prediction'],row['gold'],row['category'])-row['official_f1'])<1e-12
            if contract is not None:
                source={r['id']:r for r in (q0 if split=='q0' else fit)}[row['id']]
                assert row['question']==source['question'] and row['gold']==source['generations']['full']['text']
                assert row['category']==4 and row['split']==('source_fit' if split=='q0' else 'source_view_fit')
                c=contract[row['id']];assert c['prediction']==row['prediction'] and c['context_tokens']==tokens
                assert c['preserved']==bool(re.fullmatch(r'EQUIVALENT[.!]?',c['verdict'].strip().upper()))
            else:assert row['conv_id'] not in protocol['excluded_holdout_conversations'] and row['split']=='dev'
            checked+=1
        print(json.dumps({'verification':'candidate_ranking_and_packing_replayed','file':filename,'checked':checked}),flush=True)
        return rows
    reference=check_rows('reference_replay_dev_items.json',baseline,'dev')
    assert abs(core.summarize(reference)['official_f1']-protocol['reference']['expected_f1'])<1e-12
    before=read(root/'provenance_payload_v2/run/reference_replay_dev_items.json');keys=('id','question','context','prediction','source_ids','read_tokens','memory_tokens','official_f1')
    assert [{k:r[k] for k in keys} for r in reference]==[{k:r[k] for k in keys} for r in before]
    base=read(run/'source_baseline_contract.json');trials={};feasible=[]
    check_rows('source_baseline_source_fit_items.json',baseline,'q0',base['q0'])
    check_rows('fit_baseline_source_view_fit_items.json',baseline,'fit_a',base['fit_a'])
    assert memory_lock['locked_at']<(run/'source_baseline_source_fit_items.json').stat().st_mtime+1
    for recipe in protocol['recipes']:
        name=recipe['name'];t=read(run/'source_trials'/f'{name}.json');trials[name]=t
        check_rows(f'{name}_source_fit_items.json',memories[name],'q0',t['q0_contract'],name)
        check_rows(f'{name}_source_view_fit_items.json',memories[name],'fit_a',t['fit_a_contract'],name)
        dq,da=changes(base['q0'],t['q0_contract']),changes(base['fit_a'],t['fit_a_contract'])
        nq,na=[len(d['repairs'])-len(d['regressions']) for d in (dq,da)]
        result={'name':name,'q0':dq,'fit_a':da,'net_q0':nq,'net_a':na,'worst_view_gain':min(nq,na),'total_storage':sum(costs[name].values()),'feasible':nq>0 and na>0}
        assert t['result']==result
        if result['feasible']:feasible.append(result)
        check_rows(f'{name}_dev_items.json',memories[name],'dev',index_name=name)
    choice=max(feasible,key=lambda r:(r['worst_view_gain'],r['net_q0']+r['net_a'],-r['total_storage'],r['name'])) if feasible else None
    lock=read(run/'source_selection_locked.json');chosen=choice['name'] if choice else 's_parent_single_2000'
    assert lock['result']==choice and lock['selected']==chosen and lock['memory_sha256']==core.digest(memories[chosen] if choice else baseline)
    assert not lock['selection_used_new_dev_outcomes'] and not lock['selection_used_audit_outcomes']
    for name in memories:assert lock['locked_at']<(run/f'{name}_dev_items.json').stat().st_mtime+1
    result={'source_hashes_verified':True,'previous_compiler_exact_digest_link_verified':True,
            'previous_payload_validation_sha256':sha(root/'PAYLOAD_SOURCE.validation.json'),
            'supported_route_candidates':sum(map(len,candidates.values())),
            'dependent_route_candidates':sum(c['deletion_sensitive'] for values in candidates.values() for c in values),
            'geometry_max_absolute_recompute_error':geometry_error,'greedy_coverage_replayed':True,
            'memory_collections_reconstructed':len(memories),'baseline_prefix_and_added_original_payloads_verified':True,
            'source_and_dev_retrieval_predictions_verified':checked,'all_selection_locks_verified':True,'source_selected':chosen}
    core.save(root/'ROUTES_SOURCE.validation.json',result)
    history=read(run/'history.json');scores={r['name']:r['dev']['official_f1'] for r in history['rounds']};n=lambda c:sum(r['preserved'] for r in c.values())
    lines=['# 원문 보존 검색 경로 구성 검증','',f"Source 선택은 **{chosen}**이며 개발 최고 **{history['best_f1']*100:.2f}**를 유지했다.",'',
           '| 구성 | Q0 보존/75 | A 보존/75 | 추가 경로 | 평균 커버리지 증가 | 개발 F1 |','|---|---:|---:|---:|---:|---:|',
           f"| 기준 | {n(base['q0'])} | {n(base['fit_a'])} | 0 | 0 | {100*protocol['reference']['expected_f1']:.2f} |"]
    for recipe in protocol['recipes']:
        name=recipe['name'];t=trials[name];d=list(construction[name].values())
        gain=sum(x['final_mean_coverage']-x['initial_mean_coverage'] for x in d)/len(d)
        lines.append(f"| {name} | {n(t['q0_contract'])} | {n(t['fit_a_contract'])} | {sum(x['added_routes'] for x in d)} | {gain:.6f} | {100*scores[name]:.2f} |")
    lines+=['',f"기존 검증된 source compiler 결과와 파일 내용·프로토콜·해시 연결을 확인한 뒤 {result['supported_route_candidates']}개 후보를 재구성했다. 저장된 임베딩으로 cosine 커버리지를 재계산하고, 기록된 float32 행렬로 탐욕 선택·예산·출처 중복 금지와 네 메모리를 재현했다. 새 인코딩과 원래 인코딩은 2e-5 허용오차로 비교했다.",
            f"실제 후보별 dense/BM25 순위, 2048-token packing, 문맥·출처·토큰·F1을 {checked}건 재현했다. 기존 본문·키·순서를 prefix로 그대로 두고, 추가 경로의 본문은 기존 unit의 정확한 사본임을 확인했다.",
            '기존 compiler의 원문·인용·checker 검사는 PAYLOAD_SOURCE 검증을 재사용하며 여기서 독립적으로 다시 실행한 것으로 세지 않는다. 새 경로는 검색 순위와 packing을 바꾸고 같은 본문을 중복 노출할 수 있다.',
            '커버리지는 source 문장 임베딩 간 유사도 대리목표다. 의미적 사실성, 답변 유용성, 일반화나 방법의 독창성을 보장하지 않는다. 반복 사용한 dev70을 독립 테스트로 부르지 않으며 audit100/B는 튜닝하지 않았다.','']
    (root/'ROUTES_SOURCE_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result))



if __name__=='__main__':main()
