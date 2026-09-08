"""Reconstruct paired event controls and independently recount their budgets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import refine as core
from contrastive_events import original_context,construct,WRITER,SCHEMA
from budgeted_evidence import storage_cost
from tokenizer_validation import load_counter
from verify_identity import changes


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_graph(units,embedding,edges):
    """Allow cross-BLAS cosine rounding, but verify every greedy eligible choice."""
    assert len(edges)==64
    matrix=embedding@embedding.T
    candidates=[]
    sources=[set(u['sources']) for u in units]
    for i in range(len(units)):
        for j in range(i+1,len(units)):
            if units[i]['session']!=units[j]['session'] and not sources[i]&sources[j] and matrix[i,j]>0:
                candidates.append((float(matrix[i,j]),i,j))
    degree=[0]*len(units)
    used=set()
    for edge in edges:
        i,j=edge['parent_indices']
        eligible=[e for e in candidates if degree[e[1]]<2 and degree[e[2]]<2 and (e[1],e[2]) not in used]
        assert (i,j) not in used and degree[i]<2 and degree[j]<2
        assert any(a==i and b==j for _,a,b in eligible)
        assert abs(float(matrix[i,j])-edge['similarity'])<1e-5
        assert edge['similarity']>=max(e[0] for e in eligible)-1e-5
        degree[i]+=1;degree[j]+=1;used.add((i,j))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    root=args.root;run=root/'contrastive_events_v1_results/run'
    protocol=read(run/'protocol.json');cids=sorted({r['conv_id'] for r in protocol['records']})
    ntok,meta=load_counter(root/'tokenizer_reference/tokenizer_reference')
    model=protocol['environment']['models'][protocol['args']['model']]
    assert meta['model_meta']==model
    for name,value in protocol['source_sha256'].items():
        assert hashlib.sha256((run.parent/'source'/name).read_bytes()).hexdigest()==value
    assert protocol['precommit']['source_sha256']==protocol['source_sha256']
    separation=read(root/'CROSS_EXPERIMENT_SEPARATION.json')
    assert protocol['precommit']['committed_at']==separation['precommit_at']<separation['independent_audit_memory_lock_at']
    parents={cid:read(run/'graph_parents'/f'{cid}.json') for cid in cids}
    baseline={cid:read(root/'parent_evidence_v1/run/memories/s_parent_single_2000'/f'{cid}.json') for cid in cids}
    sessions=read(root/'source_sessions.json')['sessions_by_id']
    assert core.digest(parents)==protocol['parent_memory_sha256']
    assert core.digest(baseline)==protocol['baseline_memory_sha256']
    graphs={cid:read(run/'event_graph'/f'{cid}.json') for cid in cids}
    graph_lock=read(run/'event_graph_locked.json')
    assert core.digest(graphs)==graph_lock['graphs_digest']
    generated_count=supported_count=0
    for cid in cids:
        units=parents[cid];graph=graphs[cid]
        verify_graph(units,np.load(run/'graph_parent_embeddings'/f'{cid}.npy'),graph['proposal_edges'])
        legal=[e for e in graph['proposal_edges'] if ntok(units[e['parent_indices'][0]]['text']+'\n\n'+units[e['parent_indices'][1]]['text'])<=2048]
        assert legal==graph['legal_edges']
        accepted_by_edge={tuple(c['edge']['parent_indices']):c for c in graph['candidates']}
        rejected_by_edge={tuple(c['edge']['parent_indices']):c for c in graph['rejected_generations']}
        assert not accepted_by_edge.keys()&rejected_by_edge.keys()
        assert accepted_by_edge.keys()|rejected_by_edge.keys()=={tuple(e['parent_indices']) for e in legal}
        for edge in legal:
            i,j=edge['parent_indices']
            left=original_context(sessions[cid],units[i]['sources']);right=original_context(sessions[cid],units[j]['sources'])
            user=f'EXCERPT A:\n{left}\n\nEXCERPT B:\n{right}\n\nJSON:'
            key=core.digest(['contrastive_event_v1',model,protocol['args']['seed'],WRITER,user,SCHEMA,320])
            value=read(run/'event_generation_cache'/f'{key}.json')
            generated_count+=1;obj=value['object']
            if (i,j) in rejected_by_edge:
                assert value==rejected_by_edge[(i,j)]['generation']
                assert not (isinstance(obj,dict) and set(obj)=={'relation','key'} and obj['relation']=='distinct_events' and isinstance(obj['key'],str) and obj['key'].strip())
                continue
            c=accepted_by_edge[(i,j)]
            assert c['edge']==edge and c['id']==f'pair_{i}_{j}' and c['key_generation']==value
            assert json.loads(value['raw'])==obj and obj['relation']=='distinct_events' and obj['key'].strip()
            expected={'text':units[i]['text']+'\n\n'+units[j]['text'],'index_text':obj['key'].strip(),
                      'sources':sorted(set(units[i]['sources'])|set(units[j]['sources'])),
                      'session':min(units[i]['session'],units[j]['session']),
                      'sessions':sorted({units[i]['session'],units[j]['session']}),
                      'parent_indices':[i,j],'kind':'contrastive_event_pair'}
            assert c['unit']==expected and c['common_cost']==storage_cost(expected,ntok)
            assert c['accepted']==bool(re.fullmatch(r'SUPPORTED[.!]?',c['verdict'].strip().upper()))
            supported_count+=c['accepted']
    a=read(root/'crossview_refinement_v1/run/query_views/fit_views.json');ids={r['id'] for r in a}
    q0=sorted([read(p) for p in (root/'evidence_utility_fit_v1/runs/evidence_utility_fit_v1/items').glob('*.json')
               if read(p)['id'] in ids],key=lambda r:r['id'])
    assert core.digest(q0)==protocol['source_q0_digest'] and core.digest(a)==protocol['source_fit_view_digest']
    assert all(r['split']=='probe_fit' for r in q0+a)
    base=read(run/'source_baseline_contract.json');source_checked=dev_checked=0
    def check(name,contract,view,memory):
        nonlocal source_checked
        rows=q0 if view=='q0' else a;byid={r['id']:r for r in rows}
        split='source_fit' if view=='q0' else 'source_view_fit'
        values=read(run/f'{name}_{split}_items.json')
        assert len(values)==75 and {r['id'] for r in values}==byid.keys()==contract.keys()
        costs={cid:sum(storage_cost(u,ntok) for u in memory[cid]) for cid in cids}
        for row in values:
            original=byid[row['id']];got=contract[row['id']]
            assert row['question']==original['question'] and row['gold']==original['generations']['full']['text']
            assert row['split']==split and row['category']==4
            assert row['read_tokens']==ntok(row['context'])<=2048 and row['memory_tokens']==costs[row['conv_id']]
            assert abs(core.f1(row['prediction'],row['gold'],4)-row['official_f1'])<1e-12
            assert got['prediction']==row['prediction'] and got['context_tokens']==row['read_tokens']
            assert got['preserved']==bool(re.fullmatch(r'EQUIVALENT[.!]?',got['verdict'].strip().upper()))
            source_checked+=1
        return costs
    check('source_baseline',base['q0'],'q0',baseline);check('fit_baseline',base['fit_a'],'fit_a',baseline)
    memories={};trials={};choices=[];strict=[];paired={}
    for recipe in protocol['recipes']:
        name=recipe['name'];memory={}
        for cid in cids:
            expected,details=construct(baseline[cid],graphs[cid]['candidates'],ntok,recipe['mode'],recipe['extra_budget'])
            memory[cid]=read(run/'memories'/name/f'{cid}.json')
            assert memory[cid]==expected and read(run/'construction'/name/f'{cid}.json')==details
            key=(cid,recipe['extra_budget'])
            assert key not in paired or paired[key]==details['selected_ids']
            paired[key]=details['selected_ids']
        memories[name]=memory;trial=read(run/'source_trials'/f'{name}.json');trials[name]=trial
        costs=check(name,trial['q0_contract'],'q0',memory)
        assert check(name,trial['fit_a_contract'],'fit_a',memory)==costs
        dq,da=changes(base['q0'],trial['q0_contract']),changes(base['fit_a'],trial['fit_a_contract'])
        nq,na=[len(d['repairs'])-len(d['regressions']) for d in (dq,da)]
        expected={'name':name,'q0':dq,'fit_a':da,'net_q0':nq,'net_a':na,'worst_view_gain':min(nq,na),
                  'total_storage':sum(costs.values()),'worst_view_feasible':nq>0 and na>0,
                  'strict_feasible':not dq['regressions'] and not da['regressions'] and nq+na>0}
        assert trial['result']==expected
        if expected['worst_view_feasible']:choices.append(expected)
        if expected['strict_feasible']:strict.append(expected)
    rank=lambda r:(r['worst_view_gain'],r['net_q0']+r['net_a'],-r['total_storage'],r['name'])
    chosen=max(choices,key=rank) if choices else None;strict_choice=max(strict,key=rank) if strict else None
    lock=read(run/'source_selection_locked.json')
    assert lock['result']==chosen and lock['strict_result']==strict_choice
    assert lock['selected']==(chosen['name'] if chosen else 's_parent_single_2000')
    assert lock['strict_selected']==(strict_choice['name'] if strict_choice else 's_parent_single_2000')
    assert lock['memory_sha256']==core.digest(memories[chosen['name']] if chosen else baseline)
    assert not lock['selection_used_audit_views'] and not lock['selection_used_new_dev_outcomes']
    history=read(run/'history.json');scores={r['name']:r['dev']['official_f1'] for r in history['rounds']}
    for name,memory in [('reference_replay',baseline),*memories.items()]:
        path=run/f'{name}_dev_items.json'
        if name!='reference_replay':assert lock['locked_at']<path.stat().st_mtime+1
        costs={cid:sum(storage_cost(u,ntok) for u in memory[cid]) for cid in cids}
        for row in read(path):
            assert ntok(row['context'])==row['read_tokens']<=2048 and row['memory_tokens']==costs[row['conv_id']]
            dev_checked+=1
    result={'source_predictions_verified':source_checked,'dev_contexts_and_storage_recounted':dev_checked,
            'event_generations_source_provenance_verified':generated_count,'supported_by_self_judge':supported_count,
            'memory_collections_reconstructed':len(memories),'pair_identity_and_common_cost_verified':True,
            'source_selection_and_timing_verified':True,'precommit_before_independent_audit_verified':True,
            'tokenizer_revision':model['revision'],'selected':lock['selected'],'strict_selected':lock['strict_selected']}
    core.save(root/'CONTRASTIVE_SOURCE.validation.json',result)
    n=lambda x:sum(v['preserved'] for v in x.values())
    lines=['# 사건 구분 검색 키의 대조 결과','',
           '같은 원문 쌍에 사건 구분 키를 붙인 방식은 literal 대조군보다 높았지만 기존 최고 F1 59.27을 넘지 못했다. 새 최고로 채택하지 않았다.','',
           '| 구성 | Q0 보존/75 | A 보존/75 | Q0 퇴행 | A 퇴행 | 개발 F1 |','|---|---:|---:|---:|---:|---:|',
           f"| 기준 | {n(base['q0'])} | {n(base['fit_a'])} | — | — | {100*protocol['reference']['expected_f1']:.2f} |"]
    for recipe in protocol['recipes']:
        name=recipe['name'];t=trials[name];r=t['result']
        lines.append(f"| {name} | {n(t['q0_contract'])} | {n(t['fit_a_contract'])} | {len(r['q0']['regressions'])} | {len(r['fit_a']['regressions'])} | {100*scores[name]:.2f} |")
    lines+=['',f"원문 생성 {generated_count}건, source 예측 {source_checked}건, 개발 문맥 {dev_checked}건을 검증했다. 토크나이저 자산의 hash를 확인하고 실제 문맥 및 저장 토큰을 다시 계산했다. 4개 메모리를 부모 원문에서 재구성했고, 같은 예산의 두 방식이 같은 원문 쌍을 사용했음을 확인했다.",
            '양쪽 질문 표현의 순증가를 요구하는 주 선택과 개별 퇴행을 금지하는 보조 선택 모두 기존 메모리를 유지했다. 선택은 새 개발 점수 전에 고정되었다.',
            '생성기와 검증기는 원문 대화 및 세션 날짜를 받았다. 437건의 SUPPORTED는 동일 모델의 판정으로, 사실성을 외부에서 증명한 수치가 아니다. 질문 표현 둘은 같은 사실을 공유하므로 독립 150표본으로 해석하지 않는다.',
            '구조 설정 및 코드가 별도 audit100의 메모리 고정 이전에 확정되었음을 확인했다. 그 audit100은 이미 소비한 검증 집합이며 이후 튜닝에 재사용하지 않는다. 사건 그래프나 대조 키 자체의 신규성을 주장하지 않는다.','']
    (root/'CONTRASTIVE_SOURCE_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
