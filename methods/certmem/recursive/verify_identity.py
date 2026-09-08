"""Reconstruct all evidence identities and independently verify two-view choices."""
import argparse
import json
from pathlib import Path
import re

import refine as core
from identity_consolidation import consolidate,identity


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def changes(a,b):
    assert a.keys()==b.keys()
    lost=sorted(k for k in a if a[k]['preserved'] and not b[k]['preserved'])
    fixed=sorted(k for k in a if not a[k]['preserved'] and b[k]['preserved'])
    return {'regressions':lost,'repairs':fixed,'feasible':not lost and bool(fixed)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    args=ap.parse_args()
    root=args.root
    run=root/'identity_refinement_v1/run'
    protocol=read(run/'protocol.json')
    source=[read(p) for p in (root/'evidence_utility_fit_v1/runs/evidence_utility_fit_v1/items').glob('*.json')]
    q0=sorted([r for r in source if 'analysis' in r and r['analysis']['source_dependent']
               and r['generations']['full']['source_answer_f1']>=.8],key=lambda r:r['id'])
    a=read(root/'crossview_refinement_v1/run/query_views/fit_views.json')
    assert core.digest(q0)==protocol['source_probe_digest'] and core.digest(a)==protocol['fit_view_digest']
    assert all(r['split']=='probe_fit' for r in q0+a)
    cids={r['conv_id'] for r in q0}
    parents={label:{cid:read(root/f'{folder}/run/memories'/name/f'{cid}.json') for cid in cids}
             for label,(folder,name) in protocol['parent_specs'].items()}
    assert {k:core.digest(v) for k,v in parents.items()}==protocol['parent_memory_sha256']
    base=read(run/'source_baseline_contract.json')
    checked=0
    def check(name,contract,view):
        nonlocal checked
        rows=q0 if view=='q0' else a
        byid={r['id']:r for r in rows}
        split='source_fit' if view=='q0' else 'source_view_fit'
        values=read(run/f'{name}_{split}_items.json')
        assert len(values)==len(rows) and {r['id'] for r in values}==byid.keys()==contract.keys()
        cost={}
        for r in values:
            original=byid[r['id']]
            assert r['question']==original['question'] and r['gold']==original['generations']['full']['text']
            assert r['split']==split and r['category']==4 and r['read_tokens']<=2048
            got=contract[r['id']]
            assert got['prediction']==r['prediction'] and got['context_tokens']==r['read_tokens']
            assert got['preserved']==bool(re.fullmatch(r'EQUIVALENT[.!]?',got['verdict'].strip().upper()))
            assert r['conv_id'] not in cost or cost[r['conv_id']]==r['memory_tokens']
            cost[r['conv_id']]=r['memory_tokens']
            checked+=1
        return cost
    base_cost=check('source_baseline',base['q0'],'q0')
    assert check('fit_baseline',base['fit_a'],'fit_a')==base_cost
    details_by_name={}
    trials={}
    feasible=[]
    memories_by_name={}
    for recipe in protocol['recipes']:
        name=recipe['name']
        details_by_name[name]={}
        memory={}
        for cid in cids:
            memory[cid]=read(run/'memories'/name/f'{cid}.json')
            expected,stats=consolidate(parents[recipe['parent']][cid],len,recipe['mode'])
            assert memory[cid]==expected
            assert len({identity(u) for u in memory[cid]})==len(memory[cid])
            assert {identity(u) for u in parents[recipe['parent']][cid]}=={identity(u) for u in memory[cid]}
            details=read(run/'construction'/name/f'{cid}.json')
            for key in ('input_units','output_units','removed_duplicate_payloads','merged_groups','all_evidence_identities_preserved'):
                assert stats[key]==details[key]
            assert details['storage_tokens']<=details['storage_cap']==base_cost[cid]+4000
            details_by_name[name][cid]=details
        memories_by_name[name]=memory
        trial=read(run/'source_trials'/f'{name}.json')
        trials[name]=trial
        cq=check(name,trial['q0_contract'],'q0')
        ca=check(name,trial['fit_a_contract'],'fit_a')
        assert cq==ca=={cid:d['storage_tokens'] for cid,d in details_by_name[name].items()}
        dq,da=changes(base['q0'],trial['q0_contract']),changes(base['fit_a'],trial['fit_a_contract'])
        result=trial['result']
        assert result['q0']==dq and result['fit_a']==da
        assert result['total_repairs']==len(dq['repairs'])+len(da['repairs'])
        assert result['total_storage']==sum(cq.values())
        assert result['feasible']==(not dq['regressions'] and not da['regressions'] and result['total_repairs']>0)
        if result['feasible']:
            feasible.append(result)
    choice=max(feasible,key=lambda r:(r['total_repairs'],-r['total_storage'],r['name'])) if feasible else None
    lock=read(run/'source_selection_locked.json')
    assert lock['result']==choice
    chosen=choice['name'] if choice else 's_parent_single_2000'
    assert lock['selected']==chosen
    assert lock['memory_sha256']==core.digest(memories_by_name[chosen] if choice else parents['best'])
    assert not lock['selection_used_audit_views'] and not lock['selection_used_new_dev_outcomes']
    history=read(run/'history.json')
    for step in history['rounds']:
        assert lock['locked_at']<(run/f"{step['name']}_dev_items.json").stat().st_mtime+1
    score_by_name={r['name']:r['dev']['official_f1'] for r in history['rounds']}
    source_score=score_by_name[chosen] if choice else protocol['reference']['expected_f1']
    n=lambda x:sum(v['preserved'] for v in x.values())
    result={'source_predictions_verified':checked,'memory_collections_reconstructed':len(memories_by_name),
            'all_distinct_evidence_identities_preserved':True,'all_source_choices_and_caps_verified':True,
            'source_choice_before_new_dev_verified':True,'source_selected':chosen,'source_selected_dev_f1':source_score,
            'dev_selected_best':history['winner'],'dev_best_f1':history['best_f1'],
            'distinct_memory_collections':len({core.digest(m) for m in memories_by_name.values()})}
    core.save(root/'IDENTITY_SOURCE.validation.json',result)
    lines=['# 증거 동일성 통합과 두 질문 표현의 보존','',
           f"개발 평가 전 source 조건으로 고른 결과는 **{chosen} / F1 {100*source_score:.2f}**다. 개발 점수 기준 최고는 별도로 **{history['winner']} / {100*history['best_f1']:.2f}**다.",'',
           '| 구성 | Q0 보존/97 | A 보존/75 | Q0 퇴행 | A 퇴행 | 중복 사본 제거 | 개발 F1 |',
           '|---|---:|---:|---:|---:|---:|---:|',
           f"| 기준 | {n(base['q0'])} | {n(base['fit_a'])} | — | — | — | {100*protocol['reference']['expected_f1']:.2f} |"]
    for recipe in protocol['recipes']:
        name=recipe['name']
        t=trials[name]
        d=t['result']
        removed=sum(v['removed_duplicate_payloads'] for v in details_by_name[name].values())
        lines.append(f"| {name} | {n(t['q0_contract'])} | {n(t['fit_a_contract'])} | {len(d['q0']['regressions'])} | {len(d['fit_a']['regressions'])} | {removed} | {100*score_by_name[name]:.2f} |")
    lines+=['',f"{checked:,}개 source 예측과 9개 메모리 모음을 검증했다. 고정 부모로부터 구성을 재현하고, 본문·출처 ID·세션이 다른 증거가 삭제되거나 합쳐지지 않았음을 확인했다.",
            'Q0와 A는 일부 동일 사실의 두 표현이다. 합계는 독립 표본 수가 아니며, 두 집합 중 어느 쪽에서도 관측 퇴행이 없어야 source 선택을 통과한다. 이 조건은 새 질문의 성능 보장이 아니다.',
            '모든 대조군은 고정 부모 3개로부터 만든 뒤, 새 개발 점수를 보기 전에 source 선택을 고정했다. 개발 결과를 모두 보고 최선을 보관하는 탐색 결과는 source 선택 성능과 구분한다.',
            'B 표현은 이전 실험에서 결과가 노출되었고 이 선택에는 사용하지 않았다. 새 benchmark audit100과 source session audit86은 여전히 사용하지 않았다.','']
    (root/'IDENTITY_SOURCE_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
