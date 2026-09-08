"""Verify a completed frozen checkpoint without printing individual audit QA."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import refine as core
from compile_research_data import cards_from_generations,split_probe_pool
from report_continuous import assert_summary_equal
from evidence_utility import subset_analysis


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('snapshot',type=Path)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--old-manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    run=args.snapshot/'run'
    protocol=read(run/'protocol.json')
    for name,value in protocol['source_sha256'].items():
        assert hashlib.sha256((args.snapshot/'source'/name).read_bytes()).hexdigest()==value
    assert read(run/'status.json')['phase']=='audit_checkpoint_complete'
    audit=protocol['audit_manifest']
    old=read(args.old_manifest)
    samples=read(args.data)
    assert core.digest(samples)==audit['dataset_sha256']==old['dataset_sha256']
    ids={r['id'] for r in audit['records']}
    assert len(ids)==len(audit['records'])==100
    assert not ids.intersection(r['id'] for r in old['records'])
    cids=set(audit['conversations'])
    assert cids==set(old['holdout_convs'])
    byid={str(s['sample_id']):s for s in samples if str(s['sample_id']) in cids}
    corpus=read(run/'source_sessions.json')
    assert set(corpus)=={'dataset_sha256','sessions_by_id'}
    assert corpus['sessions_by_id']=={cid:core.session_data(s) for cid,s in byid.items()}
    sessions=corpus['sessions_by_id']
    pool=read(run/'source_probes.json')
    dump=read(run/'source_generations.json')
    assert split_probe_pool(sessions,cards_from_generations(dump),seed=20260908,per_session=2)==pool
    # Recover every writer input window independently from the frozen recipe.
    op=protocol['plans']['r24_source_qa_cards']['recipe']['operations'][0]
    expected_windows=[]
    for cid,ss in sorted(sessions.items()):
        for session,turns in core.windows(ss,size=op['window'],overlap=op['overlap']):
            expected_windows.append((cid,session['num'],[t['id'] for t in turns]))
    assert expected_windows==[(r['conv_id'],r['session'],r['turn_ids']) for r in dump['records']]
    selection=read(run/'utility_selection_locked.json')
    assert selection['source_corpus_sha256']==core.digest(corpus)
    assert selection['probe_pool_sha256']==core.digest(pool)
    assert selection['policy']==protocol['utility_policy']
    selected=sorted([r for r in pool['records'] if r['split']=='probe_fit'],key=lambda r:r['id'])
    assert selection['selected_ids']==[r['id'] for r in selected]
    selected_byid={r['id']:r for r in selected}
    rows=defaultdict(list)
    traces=0
    utility_rows=[read(p) for p in sorted((run/'utility/items').glob('*.json'))]
    assert len(utility_rows)==len(selected) and {r['id'] for r in utility_rows}==selected_byid.keys()
    for row in utility_rows:
        assert all(row[k]==v for k,v in selected_byid[row['id']].items())
        assert row['split']=='probe_fit'
        rows[row['conv_id']].append(row)
        turns={t['id']:t for s in sessions[row['conv_id']] for t in s['turns']}
        for span in row['answer_spans']:
            assert turns[span['source_id']]['body'][span['start']:span['end']]==span['verbatim']
            assert span['verbatim'].lower()==row['answer'].lower()
        if 'skip' in row:
            continue
        n=len(set(row['candidate_context_ids']))
        assert len(row['subsets'])==3+n+n*(n-1)//2
        lengths=set()
        for subset in row['subsets']:
            assert set(subset['source_ids'])<=set(row['candidate_context_ids'])
            assert 0<=subset['context_tokens']<=protocol['utility_policy']['read_budget']
            score=subset['score']
            probs=score['token_logprobs']
            assert probs and all(math.isfinite(p) and p<=1e-5 for p in probs)
            assert len(probs)==score['answer_tokens']
            assert abs(sum(probs)-score['sum_logprob'])<1e-10
            assert abs(sum(probs)/len(probs)-score['mean_logprob'])<1e-12
            lengths.add(len(probs))
            traces+=1
        assert len(lengths)==1
        assert subset_analysis(row['subsets'])==row['analysis']
        for g in row['generations'].values():
            assert abs(core.generic_f1(g['text'],row['answer'])-g['source_answer_f1'])<1e-12
            assert g['context_tokens']<=2048
    memories={name:{cid:read(run/'memories'/name/f'{cid}.json') for cid in cids} for name in protocol['methods']}
    lock=read(run/'memory_selection_locked.json')
    assert not lock['construction_used_benchmark_questions_or_answers']
    assert lock['protocol_sha256']==core.digest(protocol)
    assert lock['memories_sha256']==core.digest(memories)
    assert lock['source_corpus_sha256']==core.digest(corpus)
    assert lock['source_fit_results_sha256']==core.digest(dict(rows))
    gate=read(run/'reference_gate.json')
    assert gate['passed'] and gate['expected_f1']==protocol['frozen_method_dev_f1']
    reference=read(run/'reference_replay_dev_items.json')
    assert len(reference)==70 and {r['id'] for r in reference}=={r['id'] for r in old['records'] if r['split']=='dev'}
    assert abs(core.summarize(reference)['official_f1']-gate['expected_f1'])<1e-12
    for row in reference:
        assert abs(core.f1(row['prediction'],row['gold'],row['category'])-row['official_f1'])<1e-12
    results=read(run/'audit_results.json')
    assert results['no_audit_based_selection'] and results['no_further_tuning_on_these_questions']
    assert results['frozen_method']==protocol['frozen_method']=='s_parent_single_2000'
    all_rows={}
    cost={}
    for name in protocol['methods']:
        path=run/f'{name}_question_audit_items.json'
        assert lock['locked_at']<path.stat().st_mtime+1
        values=read(path)
        assert len(values)==100 and {r['id'] for r in values}==ids
        manifest_byid={r['id']:r for r in audit['records']}
        cost[name]={}
        for r in values:
            record=manifest_byid[r['id']]
            assert all(r[k]==v for k,v in record.items())
            qa=byid[r['conv_id']]['qa'][r['qa_index']]
            assert r['question']==qa['question'] and r['gold']==str(qa['answer']) and r['category']==int(qa['category'])
            assert abs(core.f1(r['prediction'],r['gold'],r['category'])-r['official_f1'])<1e-12
            assert 0<=r['read_tokens']<=2048
            assert r['memory_units']==len(memories[name][r['conv_id']])
            assert r['conv_id'] not in cost[name] or cost[name][r['conv_id']]==r['memory_tokens']
            cost[name][r['conv_id']]=r['memory_tokens']
        assert_summary_equal(core.summarize(values),results['summaries'][name])
        all_rows[name]=values
    for cid in cids:
        detail=read(run/'construction/s_parent_single_2000'/f'{cid}.json')
        assert detail['storage_tokens']==cost['s_parent_single_2000'][cid]
        assert detail['parent_tokens']==cost['r40_fused_four_turn'][cid]
        assert detail['storage_tokens']<=detail['storage_cap']==detail['parent_tokens']+2000
        parent=memories['r40_fused_four_turn'][cid]
        assert memories['s_parent_single_2000'][cid][:len(parent)]==parent
        valid_sources={t['id'] for s in sessions[cid] for t in s['turns']}
        assert all(set(u['sources'])<=valid_sources for name in memories for u in memories[name][cid])
    paired=core.paired_interval(all_rows['r40_fused_four_turn'],all_rows['s_parent_single_2000'],20260908)
    assert paired==results['paired_delta_parent_vs_r40']
    per_conversation={cid:{name:core.summarize([r for r in values if r['conv_id']==cid])['official_f1']
                          for name,values in all_rows.items()} for cid in sorted(cids)}
    validation={'source_hashes_verified':True,'source_corpus_and_literal_probe_recompilation_verified':True,
                'source_generation_windows_verified':len(dump['records']),'source_fit_probes_verified':len(utility_rows),
                'source_probability_traces_verified':traces,'source_session_audit_unused':sum(r['split']=='probe_audit' for r in pool['records']),
                'new_question_ids_disjoint_from_original_100':True,'reference_predictions_verified':70,
                'new_question_predictions_verified':300,'read_and_parent_storage_caps_verified':True,
                'all_memory_locks_precede_audit_outputs':True,'locked_method':'s_parent_single_2000',
                'audit_summaries':results['summaries'],'per_conversation_f1':per_conversation,'paired_comparison':paired,
                'interpretation':'Frozen policy checkpoint on 100 new questions in three previously exposed conversations; never tune from these results.'}
    core.save(args.out.with_suffix('.validation.json'),validation)
    lines=['# 고정 정책의 새 100문항 검증','',
           '최초 100문항과 겹치지 않는 새 질문에서, 정책 및 세 방법의 메모리를 모두 고정한 뒤 평가했다. 대상 대화 3개는 최초 pilot에 등장했던 대화이므로 미사용 대화의 일반화 점수는 아니다. 아래 결과로 방법을 선택하거나 다음 refinement를 조정하지 않는다.','',
           '| 방법 | 새 100문항 F1 | 저장 토큰/질문 평균 | 읽기 토큰 |','|---|---:|---:|---:|']
    for name in protocol['methods']:
        s=results['summaries'][name]
        lines.append(f"| {name} | {100*s['official_f1']:.2f} | {s['memory_tokens']:,.0f} | {s['read_tokens']:,.0f} |")
    lines+=['',f"고정 parent 정책과 r40의 차이는 **{100*paired['delta']:+.2f}점**이다. 대화별 bootstrap 95% 구간은 [{100*paired['ci95_cluster_bootstrap'][0]:+.2f}, {100*paired['ci95_cluster_bootstrap'][1]:+.2f}]점이며, 대화가 3개뿐이므로 추론의 한계가 크다.",'',
            '| 대화 | seed F1 | r40 F1 | 고정 parent F1 |','|---|---:|---:|---:|']
    for cid,s in per_conversation.items():
        lines.append(f"| {cid} | {100*s['seed']:.2f} | {100*s['r40_fused_four_turn']:.2f} | {100*s['s_parent_single_2000']:.2f} |")
    lines+=['',f"300개 새 질문 예측과 70개 기준 재현 예측을 재채점했다. 공개 원문에서의 source corpus 및 probe 재컴파일, {len(utility_rows)}개 source-fit probe와 {traces:,}개 확률 추적, source session audit 미사용, 문항 분리, 메모리 hash, 저장·읽기 상한 및 평가 전 고정을 검증했다.",
            '저장 토큰 표는 평가 질문 빈도로 가중한 평균이다. parent의 추가 저장 상한은 r40보다 대화당 2000토큰이며 실제 사용량은 다를 수 있다. 서로 다른 저장 예산을 같은 비용이라고 해석하지 않는다.',
            '질문 단서는 해당 대화 원문에서 생성했으며 벤치마크 질문·정답을 구성 함수에 넘기지 않았다. Source 생성·검증은 같은 Qwen 계열 모델에 의존한다. 이 checkpoint 이후에도 개발 7개 대화의 연구는 계속하되 이 100문항의 개별 실패에 맞춰 조정하지 않는다.','']
    lines+=['데이터 접근 범위: 준비 단계는 원본 JSON 전체를 파싱한 뒤 대상 대화의 source 세션만 추출했다. 따라서 파일 바이트나 QA 문자열이 프로세스 메모리에 전혀 로드되지 않았다는 뜻은 아니다. 고정 protocol의 QA 미입력 문구는 해당 벤치마크 QA 필드를 준비·구성·선택 입력으로 사용하지 않았다는 데이터 흐름상의 범위로 해석한다. 개별 새 질문과 정답을 평가 입력으로 추출한 시점은 메모리 고정 이후다.','']
    args.out.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(validation))


if __name__=='__main__':
    main()
