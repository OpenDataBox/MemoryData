"""Audit source-only selection decisions independently from benchmark F1."""
import argparse
import json
from pathlib import Path
import re

import refine as core


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def differences(before,after):
    assert before.keys()==after.keys()
    lost=sorted(k for k in before if before[k]['preserved'] and not after[k]['preserved'])
    repaired=sorted(k for k in before if not before[k]['preserved'] and after[k]['preserved'])
    return lost,repaired


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('snapshot',type=Path)
    ap.add_argument('--fit-run',type=Path,required=True)
    ap.add_argument('--parent-snapshot',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    run=args.snapshot/'run'
    protocol=read(run/'protocol.json')
    fit=[read(p) for p in (args.fit_run/'items').glob('*.json')]
    probes={r['id']:r for r in fit if 'analysis' in r and r['analysis']['source_dependent']
            and r['generations']['full']['source_answer_f1']>=0.8}
    assert core.digest(sorted(probes.values(),key=lambda r:r['id']))==protocol['source_probe_digest']
    baseline=read(run/'source_baseline_contract.json')
    assert baseline.keys()==probes.keys()
    parent_name=protocol['reference']['name']
    cids={r['conv_id'] for r in probes.values()}
    parent={cid:read(args.parent_snapshot/'run/memories'/parent_name/f'{cid}.json') for cid in cids}
    assert core.digest(parent)==protocol['parent_memory_sha256']
    source_rows_checked=0
    def validate_assessment(name,contract):
        nonlocal source_rows_checked
        values=read(run/f'{name}_source_fit_items.json')
        assert len(values)==len(probes) and {v['id'] for v in values}==probes.keys()==contract.keys()
        costs={}
        for value in values:
            row=probes[value['id']]
            assert value['question']==row['question']
            assert value['gold']==row['generations']['full']['text']
            assert value['split']=='source_fit' and value['category']==4
            assert 0<=value['read_tokens']<=2048
            assert contract[value['id']]['prediction']==value['prediction']
            assert contract[value['id']]['context_tokens']==value['read_tokens']
            verdict=contract[value['id']]['verdict']
            assert contract[value['id']]['preserved']==bool(re.fullmatch(r'EQUIVALENT[.!]?',verdict.strip().upper()))
            cid=value['conv_id']
            assert cid not in costs or costs[cid]==value['memory_tokens']
            costs[cid]=value['memory_tokens']
            source_rows_checked+=1
        return costs
    base_cost=validate_assessment('source_baseline',baseline)
    trials={p.stem:read(p) for p in (run/'source_trials').glob('*.json')}
    all_costs={}
    for name,trial in trials.items():
        costs=validate_assessment(name,trial['contract'])
        assert sum(costs.values())==trial['result']['total_storage']
        assert all(costs[cid]<=base_cost[cid]+4000 for cid in costs)
        all_costs[name]=costs
    history=read(run/'source_history.json')
    assert len(history)<=3
    before=baseline
    decisions=[]
    for step in history:
        iteration=step['iteration']
        candidates=[]
        for name,trial in trials.items():
            if not name.startswith(f'iter{iteration}_'):
                continue
            lost,repaired=differences(before,trial['contract'])
            result=trial['result']
            assert result['regressions']==lost and result['repairs']==repaired
            assert result['feasible']==(not lost and bool(repaired))
            if result['feasible']:
                candidates.append(result)
        if step['accepted']:
            chosen=max(candidates,key=lambda r:(len(r['repairs']),-r['total_storage'],r['name']))
            assert chosen['name']==step['name']
            assert step['repairs']==chosen['repairs'] and not step['regressions']
            before=trials[step['name']]['contract']
            decisions.append({'iteration':iteration,'name':step['name'],'repairs':len(step['repairs']),
                              'regressions':0,'preserved':sum(v['preserved'] for v in before.values())})
        else:
            assert not candidates
    lock=read(run/'source_selection_locked.json')
    assert lock['source_history']==history
    assert lock['source_preserved']==sum(v['preserved'] for v in before.values())
    memory={cid:read(run/'memories/t_source_selected'/f'{cid}.json') for cid in cids}
    assert core.digest(memory)==lock['memory_sha256']
    plan=read(run/'plans/t_source_selected.json')
    assert lock['locked_at']<=plan['committed_at']
    assert lock['locked_at']<= (run/'t_source_selected_dev_items.json').stat().st_mtime+1
    dev=read(run/'history.json')['rounds'][0]['dev']['official_f1']
    result={'source_queries_verified':len(probes),'source_predictions_verified':source_rows_checked,
            'counterfactual_trials_verified':len(trials),'baseline_preserved':sum(v['preserved'] for v in baseline.values()),
            'selected_preserved':lock['source_preserved'],'decisions':decisions,
            'choice_maximization_verified':True,'observed_nonregression_verified':True,
            'all_trial_storage_caps_verified':True,'final_memory_digest_verified':True,
            'source_lock_precedes_new_dev_output':True,'selected_dev_f1':dev,
            'reference_dev_f1':protocol['reference']['expected_f1']}
    core.save(args.out.with_suffix('.validation.json'),result)
    lines=['# 실제 검색 경로에서의 source-only 반복 선택','',
           f"원문 기반 {len(probes)}개 진단에서 보존한 답은 **{result['baseline_preserved']} → {lock['source_preserved']}**로 늘었다. 개발 F1은 **{100*protocol['reference']['expected_f1']:.2f} → {100*dev:.2f}**여서 기존 최고를 갱신하지 못했다.",'',
           '| 반복 | source 기준 선택 | 새로 회복 | 관측 퇴행 | 보존 답 수 |','|---|---|---:|---:|---:|']
    for item in decisions:
        lines.append(f"| {item['iteration']} | {item['name']} | {item['repairs']} | {item['regressions']} | {item['preserved']} |")
    lines+=['',f"{len(trials)}개 counterfactual 후보의 {source_rows_checked:,}개 source 예측, 모든 저장 상한, 선택 우선순위, 최종 메모리 hash와 dev 평가 전 선택 고정을 검증했다.",
            '모든 관측 후보의 source 퇴행은 0이었다. 따라서 이 실행만으로 퇴행 거절 조건이 선택을 개선했다고 주장할 수 없다.',
            '원문 기반 질문이 저장한 검색 질문 문구와 일치하는 경우가 있다. 진단 개선이 다른 표현이나 다른 사실의 질문으로 전이하는지는 별도 검증해야 한다. 이번 결과만으로 문구 과적합을 확정할 수도 없다.',
            'source 보존은 같은 모델의 의미 판정이며 사람 검증이나 새 질문에서의 보장과 다르다. 벤치마크 새 100문항은 이 구성·선택에 사용하지 않았다.','']
    args.out.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
