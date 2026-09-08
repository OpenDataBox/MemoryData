"""Verify benchmark comparisons and declared per-conversation storage ceilings."""
import argparse
import hashlib
import json
from pathlib import Path
import refine as core
from report_continuous import assert_summary_equal


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('snapshot',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    run=args.snapshot/'run'
    history,protocol=read(run/'history.json'),read(run/'protocol.json')
    for name,value in protocol['source_sha256'].items():
        assert hashlib.sha256((args.snapshot/'source'/name).read_bytes()).hexdigest()==value
    ids={r['id'] for r in protocol['records']}
    excluded=set(protocol['excluded_holdout_conversations'])
    gate=read(run/'reference_gate.json')
    assert gate['passed']
    def check_rows(name):
        values=read(run/f'{name}_dev_items.json')
        assert len(values)==len(ids)==70 and {r['id'] for r in values}==ids
        for row in values:
            assert row['conv_id'] not in excluded and row['split']=='dev'
            assert 0<=row['read_tokens']<=protocol['args']['budget']
            assert abs(core.f1(row['prediction'],row['gold'],row['category'])-row['official_f1'])<1e-12
        return values
    reference=core.summarize(check_rows('reference_replay'))
    assert abs(reference['official_f1']-gate['expected_f1'])<1e-12
    count=70
    digests={}
    summaries={}
    recipe_by_name={r['name']:r for r in protocol['recipes']}
    vector_family=any(r.get('family')=='source_supervised_memory_index' for r in protocol['recipes'])
    payload_family=any(r.get('family')=='source_grounded_payload' for r in protocol['recipes'])
    route_family=any(r.get('family') in ('source_factual_routes','source_route_identity') for r in protocol['recipes'])
    index_digests={}
    best=gate['expected_f1']
    for result in history['rounds']:
        name=result['name']
        assert result['recipe']==recipe_by_name[name]
        values=check_rows(name)
        count+=len(values)
        assert_summary_equal(core.summarize(values),result['dev'])
        cids={r['conv_id'] for r in values}
        details={cid:read(run/'construction'/name/f'{cid}.json') for cid in cids}
        memories={cid:read(run/'memories'/name/f'{cid}.json') for cid in cids}
        for row in values:
            item=details[row['conv_id']]
            cap=item.get('storage_cap',item.get('budget'))
            assert cap is not None and row['memory_tokens']==item['storage_tokens']<=cap
        digests[name]=core.digest(memories)
        if vector_family or route_family:
            import numpy as np
            index_hashes={}
            for cid in cids:
                path=run/'indexes'/name/f'{cid}.npy'
                matrix=np.load(path)
                assert matrix.dtype==np.float32 and matrix.shape[0]==len(memories[cid])
                assert matrix.nbytes==details[cid]['dense_index_bytes']
                if vector_family:assert all(r['dense_index_bytes']==matrix.nbytes for r in values if r['conv_id']==cid)
                index_hashes[cid]=hashlib.sha256(path.read_bytes()).hexdigest()
            index_digests[name]=core.digest(index_hashes)
        summaries[name]=result['dev']
        accepted=result['dev']['official_f1']>best+0.001
        assert accepted==result['accepted']
        if accepted:
            best=result['dev']['official_f1']
    assert abs(best-history['best_f1'])<1e-12
    unique=len(set(digests.values()))
    validation={'source_hashes_verified':True,'reference_gate_verified':True,'predictions_verified':count,
                'storage_caps_verified':True,'read_caps_verified':True,'holdout_excluded':True,
                'completed_recipes':len(history['rounds']),'distinct_memory_collections':unique,
                'best_f1':history['best_f1'],'winner':history['winner']}
    if vector_family or route_family:
        validation['distinct_dense_indexes']=len(set(index_digests.values()))
        validation['dense_index_bytes_verified']=True
    core.save(args.out.with_suffix('.validation.json'),validation)
    lines=['# 방법론 비교 — 개발 70문항','',
           f"기존 최고 F1 **{100*gate['expected_f1']:.2f}**를 재현한 뒤 비교했다. 이 비교의 기준과 후보 중 최고는 **{history['winner']} / {100*history['best_f1']:.2f}**다.",
           '메모리는 원문으로 구성하고, 필요한 source 선택에는 원문 기반 probe를 사용했다. 아래 점수는 기존 개발 70문항 평가이며 새 일반화 성능이 아니다.','',
           '| 구성 | 개발 F1 | 저장 토큰/대화 | 읽기 토큰 |','|---|---:|---:|---:|',
           f"| 기존 최고 재현 | {100*reference['official_f1']:.2f} | {reference['memory_tokens']:,.0f} | {reference['read_tokens']:,.0f} |"]
    for result in history['rounds']:
        s=result['dev']
        lines.append(f"| {result['name']} | {100*s['official_f1']:.2f} | {s['memory_tokens']:,.0f} | {s['read_tokens']:,.0f} |")
    lines+=['',f"{count:,}개 예측을 재채점하고 소스 hash·문항 분리·읽기 상한·대화별 저장 상한을 검증했다. {len(history['rounds'])}개 recipe가 만든 서로 다른 텍스트 메모리 모음은 {unique}개다."+
            (f" 텍스트는 동일하지만 학습한 dense index는 {len(set(index_digests.values()))}개로 서로 다르다. 벡터 배열 바이트도 별도로 검증했다." if vector_family else ' 동일 메모리를 독립 반복 실험으로 세지 않는다.'),
            '동일 저장 상한을 적용했지만 실제 사용량은 다를 수 있다. 검색 키와 원문 payload를 모두 비용에 포함한다. 같은 모델의 의미 검증은 사실성 증명이 아니다.',
            ('검색 키·순서와 모델을 고정하고 읽기 payload만 구성했다. 원문 focus를 보존하며 기존 unit보다 길어지면 원래 payload로 되돌린다. 검색 순위는 같지만 payload 길이에 따라 읽기에 포함되는 unit은 달라질 수 있다. 변경된 본문과 고정 검색 키를 모두 저장하므로 기존 최고 대비 저장량이 늘 수 있고, 상한은 기존의 2배다.' if payload_family else
             '원문·검색 텍스트·출처·모델 가중치는 고정하고 저장된 문서 벡터만 학습했다. 텍스트와 벡터 배열의 저장량은 모든 대조군에서 같다. 학습 source 질문의 개선이 새 질문의 개선을 보장하지 않는다.' if vector_family else
             '동일 본문·출처 ID·세션을 가진 메모리 사본을 한 읽기 unit으로 통합한 대조 실험이다. 모든 서로 다른 증거 본문은 보존한다. 검색 키 통합으로 검색 순위도 달라질 수 있으므로 단순 저장 공간 절감과 성능 효과를 구분한다.'
             if any(r.get('family')=='exact_evidence_identity_consolidation' for r in protocol['recipes']) else
             '기존 전체 메모리를 residual로 남기는 증거 추가·검색 단서 구성 실험이다. 원문을 영구 삭제하는 압축의 우월성을 입증한 것으로 해석하지 않는다.'),'']
    if 'b00_raw' in summaries:
        local_best=max(history['rounds'],key=lambda r:r['dev']['official_f1'])
        delta=100*(local_best['dev']['official_f1']-summaries['b00_raw']['official_f1'])
        lines += [f"원문 기준 대비 이 묶음의 최선은 {local_best['name']}이며 F1 차이는 {delta:+.2f}점이다. 기존의 더 강한 메모리 대비 효과는 별도 실험에서 확인한다.",'']
    args.out.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(validation))


if __name__=='__main__':
    main()
