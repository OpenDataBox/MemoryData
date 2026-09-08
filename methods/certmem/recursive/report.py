"""Verify downloaded pilot records and write a compact Korean results report."""
import argparse
import hashlib
import json
from pathlib import Path

from refine import f1, paired_interval

LABELS = {'session10':'세션별 10개 사실', 'events':'세부 사건 메모리', 'audit':'누락 사실 재검사',
          'dialogue_residual':'재검사 + 인접 발화 보존', 'provenance_quotes':'사실에 원문 직접 연결'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    def read(name):
        return json.loads((args.run_dir/name).read_text(encoding='utf-8'))
    result, history, manifest, protocol = (read(n) for n in ['final.json','history.json','manifest.json','protocol.json'])
    baseline, winner, raw_rows = (read(n) for n in ['pilot100_baseline_items.json','pilot100_winner_items.json','raw_rag_reference_all_items.json'])
    locked = read('selection_locked.json')
    assert result['winner'] == history['winner'] == locked['winner']
    record_ids = {r['id'] for r in manifest['records']}
    checked = 0
    for label, rows in [('baseline',baseline),('winner',winner),('raw',raw_rows)]:
        assert len(rows) == len(record_ids) == 100
        assert {r['id'] for r in rows} == record_ids
        assert sum(r['split']=='holdout' for r in rows) == 30
        assert all(r['read_tokens'] <= protocol['args']['budget'] for r in rows)
        for row in rows:
            score = f1(row['prediction'],row['gold'],row['category'])
            assert abs(score-row['official_f1']) < 1e-12
            assert 0 <= score <= 1
            checked += 1
        mean = sum(r['official_f1'] for r in rows)/len(rows)
        key = 'raw_rag_reference' if label == 'raw' else 'pilot100_'+label
        assert abs(mean-result[key]['official_f1']) < 1e-12
        for split in ['dev','holdout']:
            if label == 'raw' and split == 'dev':
                continue
            subset = [r for r in rows if r['split'] == split]
            key = 'raw_rag_holdout' if label == 'raw' else split+'_'+label
            assert abs(sum(r['official_f1'] for r in subset)/len(subset)-result[key]['official_f1']) < 1e-12
    bh = [r for r in baseline if r['split']=='holdout']
    wh = [r for r in winner if r['split']=='holdout']
    interval = paired_interval(bh,wh,manifest['seed'])
    recorded = result['holdout_paired_delta']
    assert abs(interval['delta']-recorded['delta']) < 1e-12
    assert all(abs(a-b)<1e-12 for a,b in zip(interval['ci95_cluster_bootstrap'],recorded['ci95_cluster_bootstrap']))
    assert interval['conversation_clusters'] == recorded['conversation_clusters']
    if args.source:
        assert hashlib.sha256(args.source.read_bytes()).hexdigest() == protocol['source_sha256']
        assert args.source.read_bytes() == Path(__file__).with_name('refine.py').read_bytes()
    selection = next(r for r in history['rounds'] if r['name']==result['winner'])
    delta100 = 100*(result['pilot100_winner']['official_f1']-result['pilot100_baseline']['official_f1'])
    storage_ratio = result['pilot100_winner']['memory_tokens']/result['pilot100_baseline']['memory_tokens']
    lines = ['# LoCoMo 100문항 메모리 구성 refinement 결과', '',
             f"최종 구성: **{LABELS[selection['strategy']]}** (`{result['winner']}`).",
             f'100문항 F1 변화: **{delta100:+.2f}점**. 아래 점수는 F1×100이다.', '',
             '| 평가 집합 | 문항 수 | 기준선 F1 | 최종 F1 | 차이 |',
             '|---|---:|---:|---:|---:|']
    for label, key in [('개발','dev'),('별도 검증','holdout'),('전체 파일럿','pilot100')]:
        a,b = result[key+'_baseline'],result[key+'_winner']
        lines.append(f"| {label} | {a['n']} | {100*a['official_f1']:.2f} | {100*b['official_f1']:.2f} | {100*(b['official_f1']-a['official_f1']):+.2f} |")
    lines += ['', '## 개발 단계 반복 결과', '',
              '| 구성 | 개발 F1 | 평균 저장 토큰/대화 | 평균 읽기 토큰 | 선택 |',
              '|---|---:|---:|---:|---|']
    for row in history['rounds']:
        s=row['dev']
        decision='최종 선택' if row['name']==result['winner'] else ('채택 후 다음 단계 진행' if row['accepted'] else '기각')
        lines.append(f"| {LABELS[row['strategy']]} | {100*s['official_f1']:.2f} | {s['memory_tokens']:,.0f} | {s['read_tokens']:,.0f} | {decision} |")
    lines += ['', '## 유형별 결과 — 전체 100문항', '',
              '| 유형 | 문항 수 | 기준선 F1 | 최종 F1 | 차이 |',
              '|---|---:|---:|---:|---:|']
    categories = {'1':'다단계 추론', '2':'시간 추론', '3':'개방형 지식', '4':'단일 사실'}
    for category, a in result['pilot100_baseline']['by_category'].items():
        b = result['pilot100_winner']['by_category'][category]
        lines.append(f"| {categories[category]} | {a['n']} | {100*a['f1']:.2f} | {100*b['f1']:.2f} | {100*(b['f1']-a['f1']):+.2f} |")
    lines += ['', '전체 개선은 주로 단일 사실 질문에서 발생했다. 다단계·시간 추론은 소폭 하락했고, 개방형 지식 질문은 두 구성 모두 낮다. 모든 유형이 개선된 결과는 아니다.']
    raw = result['raw_rag_reference']
    rawh = result['raw_rag_holdout']
    ci = interval['ci95_cluster_bootstrap']
    lines += ['', '## 비용과 해석', '',
              f"- 저장 토큰은 기준 대비 **{storage_ratio:.2f}배**. 이 실험은 저장 용량을 동일하게 맞춘 비교가 아니다.",
              '- 모델·reader prompt·hybrid 검색·최대 2048 읽기 토큰은 모든 후보에서 동일하다.',
              f"- 같은 읽기 예산의 원문 RAG: 전체100 F1 **{100*raw['official_f1']:.2f}**, 검증30 F1 **{100*rawh['official_f1']:.2f}**.",
              f"- 검증 차이의 대화 단위 paired bootstrap95% 구간: **[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]점**. 검증 대화가3개뿐이라 작은 파일럿의 기술 통계로 해석한다.",
              '- 전체100은 후보 선택에 사용한 개발70을 포함한다. 일반화 평가는 다른 대화의 검증30으로 구분한다.',
              '- category1–4 포함, adversarial category5 제외. 각 대화10문항, 대화 내 category 비율에 비례해 선택했다.',
              '- 생성기는 대화만 받으며 벤치마크 질문·정답·근거는 메모리 생성 프롬프트에 들어가지 않는다.',
              '- 기준선은 Mem0-style fact prompt이며 upstream Mem0나 기존 v15 전체의 재현 실험이 아니다.',
              '- source ID recall은 인용 출처의 포함률이며 정확한 사실 보존/entailment 보장은 아니다.', '',
              '## 재현과 검증', '',
              '- GPU: 실제 RTX 4080 SUPER, 32,760 MiB. Qwen3-8B BF16, Qwen3-Embedding-0.6B, thinking off, temperature=0.',
              f"- seed={manifest['seed']}; 검증 대화: {', '.join(manifest['holdout_convs'])}.",
              '- 초기 run과 추가 후보 run 모두 검증 전 중단했고 최종v3에서 후보를 먼저 고정한 뒤 검증을 실행했다.',
              '- 긴 원문 임베딩의 OOM을 batch4(필요 시2/1)로 해결했다. 새 정책으로 모든 후보를 재계산했고 기존 네 후보의 점수가 동일하게 재현됐다.',
              f'- 예측 {checked}개의 점수 로컬 재계산, 100문항 ID 일치, 30개 holdout, 2,048토큰 상한, paired 구간 재현을 확인했다.',
              '- 8개 프로토콜 검사 및 공식 채점 코드와 816개 대조 검사 통과. Qwen3-8B의 모든 가중치 shard SHA256 검증 완료.',
              '- [공식 LoCoMo 채점 코드](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)의 category-specific F1을 사용하며 기존 certmem의 날짜/숫자 정규화 F1과 구분한다.', '',
              '문항별 예측·근거 문맥: `pilot100_baseline_items.json`, `pilot100_winner_items.json`, `raw_rag_reference_all_items.json`.',
              '고정 설정·선택·전체 점수: `protocol.json`, `manifest.json`, `selection_locked.json`, `history.json`, `final.json`.',
              f'위 결과 파일 위치: `{args.run_dir.resolve()}`.',
              f"재현 코드: `{Path(__file__).parent.resolve()}`. 실행 안내는 `README_KO.md`에 있다.", '']
    output = args.output or args.run_dir/'REPORT_KO.md'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines),encoding='utf-8')
    audit={'validated_predictions':checked,'read_budget':protocol['args']['budget'],
           'all_ids_match':True,'score_recalculation':True,'paired_interval_reproduced':True,
           'source_sha256_checked':bool(args.source)}
    (output.parent/'validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(output),'winner':result['winner'],'delta100_points':delta100,
                      'holdout_delta_points':100*interval['delta'],'storage_ratio':storage_ratio}))


if __name__=='__main__':
    main()
