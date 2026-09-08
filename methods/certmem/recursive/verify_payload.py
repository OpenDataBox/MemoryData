"""Independently reconstruct source payloads, provenance and actual retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi
import refine as core
from provenance_payload_v2 import window_for,literal_facts,construct,WRITER,CHECKER,SCHEMA
from contrastive_events import original_context
from budgeted_evidence import storage_cost
from tokenizer_validation import load_counter
from verify_identity import changes


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def valid_json(value):return isinstance(value['object'],dict) and set(value['object'])=={'facts'} and isinstance(value['object']['facts'],list) and len(value['object']['facts'])<=4


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--run',default='provenance_payload_v2');args=ap.parse_args()
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
    generated=retry_groups=literal_count=supported_count=dependent_count=checker_count=0
    for cid in cids:
        expected_windows={core.digest(sorted(set(u['sources']))):window_for(sessions[cid],u['sources'],ntok) for u in baseline[cid]}
        assert compiled[cid].keys()==expected_windows.keys()
        turns={t['id']:t for s in sessions[cid] for t in s['turns']}
        for key,item in compiled[cid].items():
            window=item['window'];assert window==expected_windows[key]
            value=item['generation'];attempts=value['attempts'];generated+=1
            if window['oversized']:
                assert not attempts and value=={'object':None,'raw':'','finish_reason':'oversized_source','attempts':[],'complete_json':False}
            else:
                caps=[a['max_tokens'] for a in attempts];assert caps in ([384],[384,768],[384,768,1536])
                previous_complete=False;last=None
                for attempt in attempts:
                    assert not previous_complete
                    digest=core.digest(['provenance_payload_v1',model,protocol['args']['seed'],WRITER,window['user'],SCHEMA,attempt['max_tokens']])
                    assert digest==attempt['cache_key'];last=read(run/'payload_generation_cache'/f'{digest}.json')
                    try:parsed=json.loads(last['raw'])
                    except json.JSONDecodeError:parsed=None
                    assert parsed==last['object']
                    previous_complete=valid_json(last);assert attempt['complete_json']==previous_complete
                assert value=={**last,'attempts':attempts,'complete_json':previous_complete}
                retry_groups+=len(attempts)>1
            facts,rejected=literal_facts(window,value,turns)
            assert rejected==item['rejected']
            assert facts==[{k:f[k] for k in ('id','text','evidence')} for f in item['facts']]
            for fact in item['facts']:
                literal_count+=1;cited={e['source_id'] for e in fact['evidence']};remaining=set(window['window_ids'])-cited
                contexts={'support':original_context(sessions[cid],cited),
                          'deleted':original_context(sessions[cid],remaining) if remaining else '(no conversation evidence)'}
                for mode,context in contexts.items():
                    user=f'Original conversation:\n{context}\n\nStatement: {fact["text"]}\n\nVerdict:'
                    digest=core.digest([model,protocol['args']['seed'],CHECKER,user,16,False])
                    cached=read(run/'payload_checker_cache'/f'{digest}.json')
                    assert cached['text']==fact[mode+'_verdict'];checker_count+=1
                supported=bool(re.fullmatch(r'SUPPORTED[.!]?',fact['support_verdict'].strip().upper()))
                dependent=supported and bool(re.fullmatch(r'INSUFFICIENT[.!]?',fact['deleted_verdict'].strip().upper()))
                assert fact['supported']==supported and fact['deletion_sensitive']==dependent
                supported_count+=supported;dependent_count+=dependent
        print(json.dumps({'verification':'source_compiler_reconstructed','conv_id':cid,'groups':len(compiled[cid])}),flush=True)
    compiler_lock=read(run/'source_payloads_locked.json')
    assert compiler_lock['compiled_sha256']==core.digest(compiled) and not compiler_lock['used_probe_questions'] and not compiler_lock['used_benchmark_questions']
    memories={};costs={};construction={}
    for recipe in protocol['recipes']:
        name=recipe['name'];memories[name]={};costs[name]={};construction[name]={}
        for cid in cids:
            expected,details=construct(baseline[cid],sessions[cid],compiled[cid],ntok,recipe['mode'])
            actual=read(run/'memories'/name/f'{cid}.json');assert actual==expected
            assert details==read(run/'construction'/name/f'{cid}.json')
            assert len(actual)==len(baseline[cid])
            assert [u['index_text'] for u in actual]==[u.get('index_text',u['text']) for u in baseline[cid]]
            assert sum(storage_cost(u,ntok) for u in actual)==details['storage_tokens']<=details['storage_cap']
            memories[name][cid]=actual;costs[name][cid]=details['storage_tokens'];construction[name][cid]=details
    memory_lock=read(run/'payload_memories_locked.json')
    assert memory_lock['memory_sha256']=={name:core.digest(m) for name,m in memories.items()} and memory_lock['all_search_keys_fixed']
    assert compiler_lock['locked_at']<=memory_lock['locked_at']
    fit=read(root/'crossview_refinement_v1/run/query_views/fit_views.json');fit_ids={r['id'] for r in fit}
    q0=sorted([read(p) for p in (root/'evidence_utility_fit_v1/runs/evidence_utility_fit_v1/items').glob('*.json') if read(p)['id'] in fit_ids],key=lambda r:r['id'])
    assert core.digest(q0)==protocol['source_q0_digest'] and core.digest(fit)==protocol['source_fit_view_digest']
    assert len(q0)==len(fit)==75 and all(r['split']=='probe_fit' for r in q0+fit)
    vectors={cid:np.load(run/'indexes/baseline'/f'{cid}.npy') for cid in cids}
    query_maps={}
    for split,filename in [('dev','reference_replay_dev_items.json'),('q0','source_baseline_source_fit_items.json'),('fit_a','fit_baseline_source_view_fit_items.json')]:
        rows=read(run/filename);query_maps[split]={}
        for cid in cids:
            selected=[r for r in rows if r['conv_id']==cid];record=read(run/'evaluation_queries'/split/f'{cid}.json')
            assert record['ids']==[r['id'] for r in selected]
            assert record['cache_key']==core.digest([protocol['environment']['models'][protocol['args']['embed_model']],[r['question'] for r in selected],True,protocol['args']['embed_batch_size'],'oom_backoff_v1'])
            values=np.load(run/'evaluation_queries'/split/f'{cid}.npy');assert values.shape==(len(selected),1024)
            query_maps[split].update({r['id']:v for r,v in zip(selected,values)})
    bm25={cid:BM25Okapi([core.lexical(u.get('index_text',u['text'])) or ['_empty'] for u in baseline[cid]]) for cid in cids}
    def ranks(values):
        order=np.argsort(-values,kind='stable');r=np.empty(len(order),dtype=np.int64);r[order]=np.arange(len(order));return r
    checked=0;orders={}
    def check_rows(filename,memory,split,contract=None):
        nonlocal checked
        rows=read(run/filename);cost={cid:sum(storage_cost(u,ntok) for u in memory[cid]) for cid in cids}
        assert len(rows)==(70 if split=='dev' else 75) and len({r['id'] for r in rows})==len(rows)
        if contract is not None:assert {r['id'] for r in rows}==contract.keys()==fit_ids
        for row in rows:
            cid=row['conv_id'];units=memory[cid];q=query_maps[split][row['id']]
            key=(split,row['id'])
            if key not in orders:
                dense=vectors[cid]@q;sparse=np.asarray(bm25[cid].get_scores(core.lexical(row['question'])))
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
        print(json.dumps({'verification':'fixed_ranking_and_packing_replayed','file':filename,'checked':checked}),flush=True)
        return rows
    reference=check_rows('reference_replay_dev_items.json',baseline,'dev')
    assert abs(core.summarize(reference)['official_f1']-protocol['reference']['expected_f1'])<1e-12
    before=read(root/'transfer_index_v1/run/reference_replay_dev_items.json');keys=('id','question','context','prediction','source_ids','read_tokens','memory_tokens','official_f1')
    assert [{k:r[k] for k in keys} for r in reference]==[{k:r[k] for k in keys} for r in before]
    base=read(run/'source_baseline_contract.json');trials={};feasible=[]
    check_rows('source_baseline_source_fit_items.json',baseline,'q0',base['q0'])
    check_rows('fit_baseline_source_view_fit_items.json',baseline,'fit_a',base['fit_a'])
    assert memory_lock['locked_at']<(run/'source_baseline_source_fit_items.json').stat().st_mtime+1
    for recipe in protocol['recipes']:
        name=recipe['name'];t=read(run/'source_trials'/f'{name}.json');trials[name]=t
        check_rows(f'{name}_source_fit_items.json',memories[name],'q0',t['q0_contract'])
        check_rows(f'{name}_source_view_fit_items.json',memories[name],'fit_a',t['fit_a_contract'])
        dq,da=changes(base['q0'],t['q0_contract']),changes(base['fit_a'],t['fit_a_contract'])
        nq,na=[len(d['repairs'])-len(d['regressions']) for d in (dq,da)]
        result={'name':name,'q0':dq,'fit_a':da,'net_q0':nq,'net_a':na,'worst_view_gain':min(nq,na),'total_storage':sum(costs[name].values()),'feasible':nq>0 and na>0}
        assert t['result']==result
        if result['feasible']:feasible.append(result)
        check_rows(f'{name}_dev_items.json',memories[name],'dev')
    choice=max(feasible,key=lambda r:(r['worst_view_gain'],r['net_q0']+r['net_a'],-r['total_storage'],r['name'])) if feasible else None
    lock=read(run/'source_selection_locked.json');chosen=choice['name'] if choice else 's_parent_single_2000'
    assert lock['result']==choice and lock['selected']==chosen and lock['memory_sha256']==core.digest(memories[chosen] if choice else baseline)
    assert not lock['selection_used_new_dev_outcomes'] and not lock['selection_used_audit_outcomes']
    for name in memories:assert lock['locked_at']<(run/f'{name}_dev_items.json').stat().st_mtime+1
    result={'source_hashes_verified':True,'source_groups_reconstructed':generated,'retried_source_groups':retry_groups,'literal_statements_verified':literal_count,
            'supported_statements':supported_count,'deletion_sensitive_statements':dependent_count,'checker_prompt_cache_pairs_verified':checker_count,
            'memory_collections_reconstructed':len(memories),'fixed_search_keys_and_rankings_verified':True,'all_payload_caps_and_fallbacks_verified':True,
            'source_and_dev_retrieval_predictions_verified':checked,'all_selection_locks_verified':True,'source_selected':chosen}
    core.save(root/'PAYLOAD_SOURCE.validation.json',result)
    history=read(run/'history.json');scores={r['name']:r['dev']['official_f1'] for r in history['rounds']};n=lambda c:sum(r['preserved'] for r in c.values())
    lines=['# 인용 기반 payload 구성 검증','',f"개발 평가 전에 source 기준으로 고른 후보는 **{chosen}**다. 개발 점수 최고는 별도로 **{history['winner']} / {100*history['best_f1']:.2f}**다.",'',
           '| 구성 | Q0 보존/75 | A 보존/75 | 추가 문장 수 | 원래 본문 fallback | 개발 F1 |','|---|---:|---:|---:|---:|---:|',
           f"| 기준 | {n(base['q0'])} | {n(base['fit_a'])} | — | — | {100*protocol['reference']['expected_f1']:.2f} |"]
    for recipe in protocol['recipes']:
        name=recipe['name'];t=trials[name];d=list(construction[name].values())
        lines.append(f"| {name} | {n(t['q0_contract'])} | {n(t['fit_a_contract'])} | {sum(x['selected_statements'] for x in d)} | {sum(x['fallback_original_payloads'] for x in d)} | {100*scores[name]:.2f} |")
    lines+=['',f"원문 묶음 {generated}개와 재시도 {retry_groups}건, 정확한 인용을 가진 문장 {literal_count}개, support/deletion 검사 {checker_count}건을 원문·생성 캐시로 재구성했다. 메모리 4개를 재구성하고 실제 검색 문맥·출처·읽기 및 저장 토큰·F1 {checked}건을 검증했다.",
            '검색 키·순서·모델은 고정되어 검색 순위가 같고, 본문 길이에 따라 읽기에 포함되는 unit은 달라질 수 있다. 원문 focus가 기존 본문 상한보다 길면 원래 payload를 그대로 보존하며 fallback에 포함했다.',
            '저장 비용에는 고정 검색 키와 새 본문이 모두 포함된다. 기존 최고 대비 상한은 2배이며 저장량이 같다고 주장하지 않는다. 추가 문장 수는 메모리 사본을 포함하므로 독립 사실 수와 다르다.',
            '문자열 인용과 같은 모델의 지지·삭제 판정은 사실성이나 인과성 인증이 아니다. 반복된 참 사실도 삭제 조건에 실패할 수 있다. source 선택은 같은 사실의 두 표현에서 순증가를 요구하며 관측된 개별 퇴행을 허용한다.',
            '모든 구성은 원문만으로 완료된 뒤 Q0/A를 사용해 선택했다. 이미 소비한 audit100과 노출된 B는 구성·선택 입력이 아니다. 개발 점수는 반복 사용한 70문항 탐색 결과다.','']
    (root/'PAYLOAD_SOURCE_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
