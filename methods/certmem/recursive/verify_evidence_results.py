"""Recheck archived probability traces and source spans without opening audit outcomes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import refine as core


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive-root',type=Path,required=True)
    ap.add_argument('--sources',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--run-name',default='evidence_utility_v1')
    args=ap.parse_args()
    run=args.archive_root/'runs'/args.run_name
    expanded=args.run_name=='evidence_utility_fit_v1'
    protocol=read(run/'protocol.json')
    for name,expected in protocol['source_sha256'].items():
        assert hashlib.sha256((args.archive_root/name).read_bytes()).hexdigest()==expected
    corpus=read(args.sources)
    assert core.digest(corpus)==protocol['source_corpus_sha256']
    rows=[read(p) for p in (run/'items').glob('*.json')]
    assert {r['id'] for r in rows}==set(protocol['selected_ids']) and len(rows)==len(protocol['selected_ids'])
    traces=0
    for row in rows:
        sessions=corpus['sessions_by_id'][row['conv_id']]
        turns={t['id']:t for s in sessions for t in s['turns']}
        for span in row['answer_spans']:
            assert turns[span['source_id']]['body'][span['start']:span['end']]==span['verbatim']
            assert span['verbatim'].lower()==row['answer'].lower()
        if 'skip' in row:
            continue
        n=len(set(row['candidate_context_ids']))
        assert len(row['subsets'])==3+n+n*(n-1)//2
        answer_lengths=set()
        for subset in row['subsets']:
            assert set(subset['source_ids'])<=set(row['candidate_context_ids'])
            assert 0<=subset['context_tokens']<=protocol['policy']['read_budget']
            score=subset['score']
            tokens=score['token_logprobs']
            assert tokens and all(math.isfinite(p) and p<=1e-5 for p in tokens)
            assert len(tokens)==score['answer_tokens']
            assert abs(sum(tokens)-score['sum_logprob'])<1e-10
            assert abs(sum(tokens)/len(tokens)-score['mean_logprob'])<1e-12
            answer_lengths.add(len(tokens))
            traces+=1
        assert len(answer_lengths)==1
        empty=next(s for s in row['subsets'] if s['kind']=='empty')['score']['mean_logprob']
        full=next(s for s in row['subsets'] if s['kind']=='full')['score']['mean_logprob']
        assert abs(full-empty-row['analysis']['full_advantage'])<1e-12
        for generation in row['generations'].values():
            assert abs(core.generic_f1(generation['text'],row['answer'])-generation['source_answer_f1'])<1e-12
            assert generation['context_tokens']<=protocol['policy']['read_budget']
    fits=[r for r in rows if r['split']=='probe_fit' and 'analysis' in r]
    valid=[r for r in fits if r['analysis']['source_dependent'] and
           r['generations']['full']['source_answer_f1']>=protocol['policy']['minimum_full_answer_f1']]
    avg=lambda values:sum(values)/len(values) if values else None
    summary={'scored':len(fits),'valid':len(valid),
             'mean_full_tokens':avg([r['analysis']['full_tokens'] for r in valid]),
             'mean_joint_tokens':avg([r['analysis']['minimum_tokens'] for r in valid]),
             'mean_single_tokens':avg([r['analysis']['single_policy_tokens'] for r in valid]),
             'joint_f1':avg([r['generations']['minimum_bundle']['source_answer_f1'] for r in valid]),
             'single_f1':avg([r['generations']['single_policy']['source_answer_f1'] for r in valid]),
             'full_f1':avg([r['generations']['full']['source_answer_f1'] for r in valid]),
             'pair_needed_for_nll_preservation':sum(r['analysis']['has_preserving_pair_without_preserving_single'] for r in valid)}
    verification={'source_hashes_verified':True,'source_span_offsets_verified':True,'probes_verified':len(rows),
                  'probability_traces_verified':traces,'read_cap_verified':True,'audit_outcomes_not_displayed':True,
                  'fit_summary':summary}
    core.save(args.out/('evidence_fit_validation.json' if expanded else 'evidence_validation.json'),verification)
    ratio=100*(1-summary['mean_joint_tokens']/summary['mean_single_tokens'])
    lines=['# 원문 결합 효용 진단 — '+('전체 source-fit 결과' if expanded else 'pilot 결과'),'',
           '벤치마크 F1 개선 결과가 아니라, 메모리 구성에 쓸 원문 증거의 결합 효과를 측정한 진단이다.',
           f"{len(rows)}개 probe와 {traces:,}개 확률 추적을 검증했다. fit {len(fits)}개 중 원문 답변 일관성 및 증거 의존 기준을 통과한 {len(valid)}개를 아래에 표시한다. 세션 audit 결과는 정책 조정에 쓰지 않도록 이 보고서에 펼치지 않는다.",'',
           '| 정책 | 남긴 증거 토큰/probe | 원문 답과 lexical F1 |',
           '|---|---:|---:|',
           f"| 전체 원문 | {summary['mean_full_tokens']:.2f} | {100*summary['full_f1']:.2f} |",
           f"| 단일 문장 또는 전체 | {summary['mean_single_tokens']:.2f} | {100*summary['single_f1']:.2f} |",
           f"| 단일·쌍·전체 공동 후보 | {summary['mean_joint_tokens']:.2f} | {100*summary['joint_f1']:.2f} |",'',
           f"공동 후보는 단일 정책 대비 증거 토큰이 {ratio:.2f}% 적었다. 단일 후보로는 NLL 보존 문턱을 통과하지 못하고 두 문장으로 통과한 probe는 {summary['pair_needed_for_nll_preservation']}개였다.",'',
           '이 수치는 전체 메모리 저장 비용이 아니다. 원문을 인용하는 답과 의미가 같은 바꿔쓰기 사이에도 lexical F1 차이가 난다. 따라서 낮은 F1을 모두 사실 손실로, 높은 NLL을 모두 의미 보존으로 해석할 수 없다. 전체 원문 대비 손실도 남아 있다. 표본이 작고 fit만의 탐색 결과이므로 기여나 일반화 성능이 입증된 것은 아니다.',
           '같은 정책·문턱·reader를 유지하여 source-fit 278개 전체로 확장했다. 그 결과로 저장 예산 내에서 단일/쌍/전체 표현을 고르는 구성을 비교한다. 벤치마크 질문과 정답은 이 구성 과정에 들어가지 않는다.','']
    if expanded:
        delta=100*(summary['joint_f1']-summary['single_f1'])
        lines+=['## 판단','',f'전체 fit에서는 공동 최소 증거의 lexical F1이 단일 정책 대비 {delta:+.2f}점이었다. 작은 pilot의 개선은 유지되지 않았다. 높은 목표 답변 확률과 실제 greedy 답변의 보존이 다르므로 NLL만을 보존 판정으로 삼을 수 없다. 실제 생성 답변과 후보 원문의 의미적 일치 여부를 검사하는 구성과, 검증 없는 대조군을 다음 실험에서 비교한다.','']
    (args.out/('EVIDENCE_FIT_KO.md' if expanded else 'EVIDENCE_PILOT_KO.md')).write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(verification))


if __name__=='__main__':
    main()
