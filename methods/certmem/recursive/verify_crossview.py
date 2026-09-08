"""Independent checks of generated view provenance, decisions and audit timing."""
import argparse
import copy
import json
from pathlib import Path
import re

import refine as core
from evidence_utility import context_for
from crossview_probes import assign_views,fingerprint,WRITER,SCHEMA


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def delta(before,after):
    assert before.keys()==after.keys()
    return {'regressions':sorted(k for k in before if before[k]['preserved'] and not after[k]['preserved']),
            'repairs':sorted(k for k in before if not before[k]['preserved'] and after[k]['preserved'])}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    args=ap.parse_args()
    root=args.root
    run=root/'crossview_refinement_v1/run'
    protocol=read(run/'protocol.json')
    fit_results=[read(p) for p in (root/'evidence_utility_fit_v1/runs/evidence_utility_fit_v1/items').glob('*.json')]
    original=sorted([r for r in fit_results if 'analysis' in r and r['analysis']['source_dependent']
                     and r['generations']['full']['source_answer_f1']>=.8],key=lambda r:r['id'])
    assert all(r['split']=='probe_fit' for r in original)
    assert core.digest(original)==protocol['original_source_probe_digest']
    orig={r['id']:r for r in original}
    sessions=read(root/'source_sessions.json')['sessions_by_id']
    fit,audit=[read(run/'query_views'/f'{v}_views.json') for v in ('fit','audit')]
    view_lock=read(run/'query_view_lock.json')
    assert core.digest(fit)==view_lock['fit_digest'] and core.digest(audit)==view_lock['audit_digest']
    assert {r['id'] for r in fit}=={r['id'] for r in audit}
    quality=read(run/'query_views/quality.json')
    assert not quality['rejected_structure']
    forbidden={fingerprint(r['question']) for r in original}
    paired=[]
    model=protocol['environment']['models'][protocol['args']['model']]
    for row in original:
        turns={t['id']:t for s in sessions[row['conv_id']] for t in s['turns']}
        context=context_for(row,turns,row['candidate_context_ids'])
        user=f"Original dialogue:\n{context}\n\nOriginal question: {row['question']}"+f"\nReference answer (do not leak into the questions): {row['generations']['full']['text']}\n\nJSON:"
        key=core.digest(['crossview_json_v1',model,protocol['args']['seed'],WRITER,user,SCHEMA,384])
        generated=read(run/'query_view_generation_cache'/f'{key}.json')
        views=assign_views(row['id'],generated['object'],forbidden)
        cal=quality['calibration'][row['id']]
        for view in ('fit','audit'):
            assert cal[view]['question']==views[view]
            accepted=all(bool(re.fullmatch(r'EQUIVALENT[.!]?',cal[view][k].strip().upper()))
                         for k in ('question_verdict','answer_verdict'))
            assert cal[view]['accepted']==accepted
        if cal['fit']['accepted'] and cal['audit']['accepted']:
            paired.append(row['id'])
    assert set(paired)=={r['id'] for r in fit}
    for view,rows in [('fit',fit),('audit',audit)]:
        for row in rows:
            expected=copy.deepcopy(orig[row['id']])
            cal=quality['calibration'][row['id']][view]
            expected['original_question']=expected['question']
            expected['question']=cal['question']
            expected['view']=view
            if view=='audit':
                expected['split']='probe_view_audit'
            expected['generations']['full']['text']=cal['reference_answer']
            expected['generations']['full']['source_answer_f1']=core.generic_f1(cal['reference_answer'],expected['answer'])
            expected['crossview_quality']=cal
            assert row==expected
    # Global intersection is checked too, beyond the pairwise generator rule.
    a_keys={fingerprint(r['question']) for r in fit}
    b_keys={fingerprint(r['question']) for r in audit}
    cross_overlap=a_keys & b_keys
    baseline=read(run/'source_baseline_contract.json')
    previous=read(run/'previous_method_fit_contract.json')
    checked=0
    def check(name,contract,view):
        nonlocal checked
        byid={r['id']:r for r in (fit if view=='fit' else audit)}
        values=read(run/f'{name}_source_view_{view}_items.json')
        assert len(values)==len(byid) and {r['id'] for r in values}==contract.keys()==byid.keys()
        costs={}
        for r in values:
            source=byid[r['id']]
            assert r['question']==source['question'] and r['gold']==source['generations']['full']['text']
            assert r['category']==4 and r['split']=='source_view_'+view
            assert 0<=r['read_tokens']<=2048
            assert contract[r['id']]['context_tokens']==r['read_tokens']
            assert contract[r['id']]['prediction']==r['prediction']
            verdict=contract[r['id']]['verdict']
            assert contract[r['id']]['preserved']==bool(re.fullmatch(r'EQUIVALENT[.!]?',verdict.strip().upper()))
            assert r['conv_id'] not in costs or costs[r['conv_id']]==r['memory_tokens']
            costs[r['conv_id']]=r['memory_tokens']
            checked+=1
        return costs
    base_cost=check('fit_baseline',baseline,'fit')
    check('fit_previous_method',previous,'fit')
    trials={p.stem:read(p) for p in (run/'source_trials').glob('*.json')}
    for name,trial in trials.items():
        costs=check(name,trial['contract'],'fit')
        assert sum(costs.values())==trial['result']['total_storage']
        assert all(costs[cid]<=base_cost[cid]+4000 for cid in costs)
    history=read(run/'source_history.json')
    current=baseline
    for step in history:
        candidates=[]
        for name,trial in trials.items():
            if not name.startswith(f"iter{step['iteration']}_"):
                continue
            d=delta(current,trial['contract'])
            result=trial['result']
            assert result['regressions']==d['regressions'] and result['repairs']==d['repairs']
            assert result['feasible']==(not d['regressions'] and bool(d['repairs']))
            if result['feasible']:
                candidates.append(result)
        if step['accepted']:
            best=max(candidates,key=lambda r:(len(r['repairs']),-r['total_storage'],r['name']))
            assert step['name']==best['name']
            current=trials[best['name']]['contract']
        else:
            assert not candidates
    lock=read(run/'source_selection_locked.json')
    assert lock['source_history']==history and lock['source_preserved']==sum(v['preserved'] for v in current.values())
    assert view_lock['locked_at']<lock['locked_at']
    assert not lock['selection_used_audit_views'] and not lock['selection_used_benchmark_answers']
    memory={p.stem:read(p) for p in (run/'memories/v_crossview_selected').glob('*.json')}
    assert core.digest(memory)==lock['memory_sha256']
    assert not any(fingerprint(u['index_text']) in (a_keys|b_keys) for units in memory.values() for u in units if 'index_text' in u)
    assert lock['locked_at']<=read(run/'plans/v_crossview_selected.json')['committed_at']
    assert lock['locked_at']<(run/'v_crossview_selected_dev_items.json').stat().st_mtime+1
    b={}
    for label in ('baseline','previous_method','selected'):
        b[label]=read(run/'view_audit'/f'{label}.json')
        check('audit_'+label,b[label],'audit')
        assert lock['locked_at']<(run/'view_audit'/f'{label}.json').stat().st_mtime+1
    old=root/'retrieval_contract_v1/run'
    q0base=read(old/'source_baseline_contract.json')
    q0steps=read(old/'source_history.json')
    chosen=[s for s in q0steps if s['accepted']][-1]['name']
    q0final=read(old/'source_trials'/f'{chosen}.json')['contract']
    n=lambda c:sum(c[k]['preserved'] for k in paired)
    dev=read(run/'history.json')['rounds'][0]['dev']['official_f1']
    result={'original_sources_verified':len(original),'paired_views_verified':len(paired),
            'generated_objects_and_hash_assignment_verified':True,'source_predictions_verified':checked,
            'counterfactual_trials_verified':len(trials),'global_fit_audit_text_overlap':len(cross_overlap),
            'fit_a':{'baseline':n(baseline),'previous_method':n(previous),'selected':n(current)},
            'audit_b':{k:n(v) for k,v in b.items()},
            'matched_q0':{'baseline':n(q0base),'previous_method':n(q0final)},
            'audit_b_changes':{k:delta(b['baseline'],v) for k,v in b.items() if k!='baseline'},
            'source_choice_and_storage_caps_verified':True,'final_memory_hash_verified':True,
            'lock_precedes_dev_and_audit_artifacts':True,'dev_f1':dev,
            'reference_dev_f1':protocol['reference']['expected_f1']}
    core.save(root/'CROSSVIEW_SOURCE.validation.json',result)
    lines=['# 다른 질문 표현으로 검증한 source-only refinement','',
           f"Fit A의 보존 답은 **{n(baseline)} → {n(current)} / {len(paired)}**로 늘었지만, 선택 고정 후 audit B는 **{n(b['baseline'])} → {n(b['selected'])}**로 줄었다. 개발 F1도 **{100*protocol['reference']['expected_f1']:.2f} → {100*dev:.2f}**로 하락했다.",'',
           '| 동일 75개 원문 사실 | 기준 메모리 | 이전 Q0 선택 메모리 | A 선택 메모리 |',
           '|---|---:|---:|---:|',
           f"| Q0 (이전 선택에 사용) | {n(q0base)} | {n(q0final)} | 미평가 |",
           f"| Fit A (이번 선택에 사용) | {n(baseline)} | {n(previous)} | {n(current)} |",
           f"| Audit B (선택 후 메모리 평가) | {n(b['baseline'])} | {n(b['previous_method'])} | {n(b['selected'])} |",'',
           f"원문 97개로 생성한 JSON과 질문 배정 hash, 두 표현의 의미 판정·답변에서 동일하게 통과한 75개 선정, {checked:,}개 검색 후 예측, {len(trials)}개 후보의 선택·저장 상한, 최종 메모리 hash를 검증했다. A/B 사이 전체 질문 문구 중복은 {len(cross_overlap)}개다.",
           'B의 질문 및 전체 원문 답변은 후보 메모리 평가 전에 데이터 품질 교정에 사용했다. B의 메모리 성능은 구성 고정 후에만 평가했다. 따라서 B 문구를 전혀 보지 않았다는 의미는 아니다.',
           '질문 표현의 개선이 다른 표현으로 유지되지 않았다. 이 결과만으로 원인이 문구 과적합인지, 불충분한 사실 범위인지, 검색 경쟁인지 확정하지 않는다. 동일 모델의 생성·판정이며, 같은 사실의 다른 문구를 검사한 결과다.',
           '이제 B 결과가 노출되었으므로 후속 방법의 새 비공개 audit로 재사용하지 않는다. 별도 새 벤치마크 100문항과 source session audit 86개는 사용하지 않았다.','']
    (root/'CROSSVIEW_SOURCE_KO.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
