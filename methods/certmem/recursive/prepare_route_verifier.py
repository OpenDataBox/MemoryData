"""Produce a separate route verifier from the completed payload replay harness."""
from pathlib import Path


def main():
    here=Path(__file__).parent
    source=(here/'verify_payload.py').read_text(encoding='utf-8')
    def replace(old,new):
        nonlocal source
        assert old in source,old
        source=source.replace(old,new)
    replace('Independently reconstruct source payloads, provenance and actual retrieval.','Reconstruct factual routes, coverage optimization and actual retrieval.')
    replace('from provenance_payload_v2 import window_for,literal_facts,construct,WRITER,CHECKER,SCHEMA','from coverage_routes import harvest,geometry,construct')
    replace("default='provenance_payload_v2'","default='coverage_routes_v1'")
    start=source.index('    generated=');end=source.index("    fit=read(root/'crossview",start)
    source=source[:start]+'''    previous=root/'provenance_payload_v2/run'
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
''' +source[end:]
    replace("    vectors={cid:np.load(run/'indexes/baseline'/f'{cid}.npy') for cid in cids}\n",'')
    replace("    bm25={cid:BM25Okapi([core.lexical(u.get('index_text',u['text'])) or ['_empty'] for u in baseline[cid]]) for cid in cids}",
            "    bm25={name:{cid:BM25Okapi([core.lexical(u.get('index_text',u['text'])) or ['_empty'] for u in memory[cid]]) for cid in cids} for name,memory in {'baseline':baseline,**memories}.items()}")
    replace('def check_rows(filename,memory,split,contract=None):','def check_rows(filename,memory,split,contract=None,index_name="baseline"):')
    replace("key=(split,row['id'])","key=(index_name,split,row['id'])")
    replace("dense=vectors[cid]@q;sparse=np.asarray(bm25[cid].get_scores(core.lexical(row['question'])))","dense=index_vectors[index_name][cid]@q;sparse=np.asarray(bm25[index_name][cid].get_scores(core.lexical(row['question'])))")
    replace('fixed_ranking_and_packing_replayed','candidate_ranking_and_packing_replayed')
    replace("root/'transfer_index_v1/run/reference_replay_dev_items.json'","root/'provenance_payload_v2/run/reference_replay_dev_items.json'")
    replace("memories[name],'q0',t['q0_contract'])","memories[name],'q0',t['q0_contract'],name)")
    replace("memories[name],'fit_a',t['fit_a_contract'])","memories[name],'fit_a',t['fit_a_contract'],name)")
    replace("memories[name],'dev')","memories[name],'dev',index_name=name)")
    start=source.index("    result={'source_hashes_verified':True");end=source.index("\n\n\nif __name__",start)
    source=source[:start]+'''    result={'source_hashes_verified':True,'previous_compiler_exact_digest_link_verified':True,
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
    (root/'ROUTES_SOURCE_KO.md').write_text('\\n'.join(lines),encoding='utf-8')
    print(json.dumps(result))
''' +source[end:]
    (here/'verify_routes.py').write_text(source,encoding='utf-8',newline='\n')


if __name__=='__main__':main()
