"""Verify and summarize a snapshot of ongoing development-only refinement."""
import argparse
import hashlib
import json
from pathlib import Path

from refine import f1, summarize


def assert_summary_equal(actual, expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:
            assert_summary_equal(actual[key],expected[key])
    elif isinstance(actual,float):
        # Python runtime versions can sum floats differently in the final bit.
        assert abs(actual-expected)<1e-12
    else:
        assert actual==expected


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run_dir',type=Path)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    def read(name):
        return json.loads((args.run_dir/name).read_text(encoding='utf-8'))
    history,protocol=read('history.json'),read('protocol.json')
    source_dir=args.run_dir.parent/'source'
    for name,expected in protocol['source_sha256'].items():
        assert hashlib.sha256((source_dir/name).read_bytes()).hexdigest()==expected
    ids={r['id'] for r in protocol['records']}
    excluded=set(protocol['excluded_holdout_conversations'])
    checked=0
    summaries={}
    versions=[('seed',history['seed'])]+[(r['name'],r['dev']) for r in history['rounds'] if 'dev' in r]
    for name,summary in versions:
        rows=read(f'{name}_dev_items.json')
        assert len(rows)==len(ids)==70
        assert {r['id'] for r in rows}==ids
        assert all(r['split']=='dev' and r['conv_id'] not in excluded for r in rows)
        assert all(0<=r['read_tokens']<=protocol['args']['budget'] for r in rows)
        for row in rows:
            assert abs(row['official_f1']-f1(row['prediction'],row['gold'],row['category']))<1e-12
            checked+=1
        computed=summarize(rows)
        assert_summary_equal(computed,summary)
        summaries[name]=computed
    winner=history['winner']
    assert abs(summaries[winner]['official_f1']-history['best_f1'])<1e-12
    last_score=history['seed']['official_f1']
    last_name='seed'
    for row in history['rounds']:
        if row['accepted']:
            assert row['dev']['official_f1']>last_score+protocol['args']['min_delta']
            last_score,last_name=row['dev']['official_f1'],row['name']
    assert last_name==winner
    seed,best=history['seed'],summaries[winner]
    delta=100*(best['official_f1']-seed['official_f1'])
    lines=['# 지속 refinement — 개발 점수 스냅샷','',
           f"현재 최선: **{winner}**. 기존 최선 개발 F1 **{100*seed['official_f1']:.2f} → {100*best['official_f1']:.2f} ({delta:+.2f}점)**.",'',
           '기존 파일럿의 개발 70문항만 사용했다. 이전 검증 30문항은 이 반복에서 제외했으며 새 일반화 성능을 주장하지 않는다.', '',
           '| 후보 | 출발 구성 | 개발 F1 | 저장 토큰/대화 | 읽기 토큰 | 채택 |',
           '|---|---|---:|---:|---:|---|',
           f"| seed | {history['seed_name']} | {100*seed['official_f1']:.2f} | {seed['memory_tokens']:,.0f} | {seed['read_tokens']:,.0f} | 기준 |"]
    for row in history['rounds']:
        if 'dev' not in row:
            lines.append(f"| {row['name']} | {row['parent']} | 실행 오류 | — | — | 기각 |")
            continue
        s=row['dev']
        accepted='현재 최선' if row['name']==winner else ('채택' if row['accepted'] else '기각')
        lines.append(f"| {row['name']} | {row['parent']} | {100*s['official_f1']:.2f} | {s['memory_tokens']:,.0f} | {s['read_tokens']:,.0f} | {accepted} |")
    lines+=['','| 질문 유형 | 문항 수 | 기존 최선 | 현재 최선 | 차이 |','|---|---:|---:|---:|---:|']
    names={'1':'다단계','2':'시간','3':'개방형 지식','4':'단일 사실'}
    for category,a in seed['by_category'].items():
        b=best['by_category'][category]
        lines.append(f"| {names[category]} | {a['n']} | {100*a['f1']:.2f} | {100*b['f1']:.2f} | {100*(b['f1']-a['f1']):+.2f} |")
    lines+=['',f"예측 {checked}개를 로컬에서 재채점했다. 문항 ID·개발 대화 분리·2,048토큰 상한·전체 요약 점수·최선 선택을 검증했다.",
            '메모리 생성기는 대화만 받는다. reader·채점·읽기 예산과 BM25+dense RRF 검색 알고리즘은 고정했다. v3에서는 구성기가 만든 검색용 표현과 읽기용 증거의 분리도 비교했으며, 서로 다르면 둘 다 저장 비용에 포함한다.',
            '저장 예산을 일치시킨 비교는 아니며 개발 70문항에 대한 반복 선택 편향이 있다. 최고 점수는 새 일반화 성능이 아니다.',
            'r24_source_qa_cards와 그 파생 후보는 생성 출력 형식 실패로 원문 fallback이 대부분이었다. 따라서 올바르게 구성한 Q&A 메모리 방식의 성능으로 해석하지 않는다.',
            '후보 개수를 완료 조건으로 삼지 않는다. 실행 중인 worker에 새 후보를 추가하고, 결과를 분석해 다음 구성을 정한다.',
            f"결과와 문항별 근거: `{args.run_dir.resolve()}`.",'']
    output=args.output or args.run_dir/'PROGRESS_KO.md'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines),encoding='utf-8')
    validation={'predictions_verified':checked,'versions_verified':len(versions),'winner':winner,
                'dev70_f1':best['official_f1'],'delta_from_seed_points':delta,
                'holdout_excluded':True,'source_hashes_verified':True,'status':'ongoing'}
    (output.parent/'continuous_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation))


if __name__=='__main__':
    main()
