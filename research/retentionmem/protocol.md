# RetentionMem 실험 프로토콜 및 실행 감사

작성일: 2026-09-03. 결과를 채우기 전 고정할 연구 기준이다.

## 검증할 주장

1. 병합 전 성공하고 병합 후 실패하는 회귀를 일반적인 오답과 구분해 예측할 수 있다.
2. 작은 정책과 잔여 정보가 같은 질의 문맥·실현 압축률에서 규칙/검증 기준보다 유리하다.
3. 압축기 또는 답변 모델을 바꿔도 재학습 없이 효과가 남고, 구축 비용까지 회수할 수 있다.

세 조건이 실제 benchmark에서 확인되기 전 성능 개선·일반성·ICLR 채택 가능성을 확정하지 않는다. 학습용 검사 질문의 정확도는 benchmark 일반화의 증거를 대신하지 않는다.

## 데이터 및 누수 방지

| 용도 | 데이터 | 분할 단위 |
|---|---|---|
| 파일럿 | 합성 source 영역 500개, 기본 train400/validation100 | 원문 가족 |
| 본 학습 | train3000 + validation750 후보; 합성 및 ALFWorld train | 원문 가족/환경 |
| QA test | 공식 LoCoMo 10대화, 범주1–4 1540문항 | 대화 |
| QA test | LongMemEval cleaned S 500문항 | 독립 haystack; 중복 원문이 있으면 상위 그룹으로 재집계 |
| 행동 test | ALFWorld 공식 unseen 환경 | gamefile |

현재 합성기는 주소·배송·출장 중심 템플릿이다. 본 학습에서는 도메인 다양화와 ALFWorld 원문이 추가로 필요하다. 테스트 질문·정답·미래 관찰은 수집기에 전달하지 않는다. 파생 원문은 모두 같은 그룹에 넣고 split을 바꾸지 않는다. 임계값은 validation만 사용한다.

공식 LoCoMo 다운로드 확인: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. 전체 1,986문항 중 446개가 adversarial 범주5다. 원본 이미지가 아닌 제공된 caption을 텍스트에 포함한 조건이다. LongMemEval 500문항 실제 다운로드·전수 실행은 아직 수행하지 않았다.

## 대조할 선행연구

| 연구 | 연결할 설계 또는 가장 가까운 차이 |
|---|---|
| [SimpleMem](https://arxiv.org/abs/2601.02553) | 구조화 압축과 검색 계획. 우리 비교에서는 압축 제안기와 손실 제어기를 분리 |
| [LightMem](https://arxiv.org/abs/2510.18866) | 주제별 묶음·offline 유지보수. 병합 시 작은 정책 적용 |
| [AdaMEM](https://arxiv.org/abs/2606.05684) | 행동 단계에 따른 경험/전략 활용. 검색 흐름을 유지한 경험 표현 교체 실험 |
| [TrustMem](https://arxiv.org/abs/2606.25161) | 가장 가까운 학습 기반 병합 검증. 과제 결과 라벨과 작은 잔여 정보의 추가 효용을 직접 비교해야 함 |
| [TierMem](https://arxiv.org/abs/2602.17913) | 원문으로 복원하는 방식과 차이를 분리. 원문 보관 자체를 신규성으로 주장하지 않음 |
| [Useful Memories Become Faulty](https://arxiv.org/abs/2605.12978) | 반복 업데이트의 손실을 기여로 독점하지 않음. 손실 제어 방법의 효과를 보여야 함 |

제공 원고의 `5_finegrained.tex` M1/M2 및 `4_overall.tex` RQ5는 문제 설정의 출발점이다. 해당 수치를 새 시스템의 실험 결과로 옮기지 않는다. 원고의 전반적인 인과 해석도 재검증한다.

## 현재 clone / baseline 감사

2026-09-03 확인한 6개 clone은 모두 존재하며 각각의 worktree는 clean이었다.

| 시스템 | clone revision | 실제 baseline 실행 상태 |
|---|---|---|
| SimpleMem | `db80b6a7c591e0ea730a058e9f5fc4eb06572299` | 미실행; planning/reflection/synthesis 설정 명시 필요 |
| LightMem | `aa1c484cc6fd964c8ea1af897e36a0c3ba06d7db` | 미실행; root adapter의 direct ingestion은 정식 전체 경로와 구분 |
| AdaMEM | `4ea93e239f8dbec2fa6013a28bc8555419037e12` | 미실행; index 보존·value 교체 도구와 log 집계 구현 |
| Mem0 | `9a7924befd7026e41e445ba809370009e5e985a6` | 미실행; root vendored `main.py:227` fact2개 제한 확인 |
| A-MEM | `ceffb860f0712bbae97b184d440df62bc910ca8d` | 미실행; link/evolve 활성화 및 비용 기록 필요 |
| LangMem | `f8c7ebd6110c124a36995dab645a8cb0eb0b8210` | 미실행; 공식 API 기반 benchmark adapter 추가 필요 |

기존 root 수정 경로를 원래 시스템의 정식 재현이라고 표시하지 않는다. TrustMem/TierMem 원 모델 실행도 아직 없다. `criteria_verifier`는 공개된 coverage/preservation/faithfulness 개념을 구현한 제어군이며 원 논문의 모델 성능을 대신하지 않는다. TierMem의 원문 복원 비교는 현재 미구현이다.

## 매칭과 분석

writer/embedding/reader/검색 혼합 가중치/질의 토큰 예산을 동일하게 놓고 선택 메커니즘을 비교한다. 원문 RAG, 항상 병합, 유지, heuristic, criteria verifier, learned, learned+residual이 구현되어 있다. `raw`와 `keep`은 동일 동작이므로 한 개의 독립 기준선으로 취급한다.

1. 동일 질의 예산 2048/4096/8192. 입력 prompt overhead와 답변 생성 토큰은 별도 보고.
2. 동일 실현 압축률. 현재 4:1은 요약 목표일 뿐이다. paired region에 동일 `maintain(..., budget=...)`를 적용한 별도 측정이 필요.
3. 반복 pass 1/2/4/8과 실제 unit 계보 깊이 둘 다 보고. 처음 구축의 병합 횟수를 빼먹지 않음.
4. 정보 갱신: 이전 값/현재 값/별개 사건 질문을 구분하고 최신 source prefix로 기준 답을 갱신.
5. 이식성: structured→topical 및 Qwen3-8B→Qwen2.5-7B-Instruct. reader만 교체하는 경우 writer는 고정.
6. 학습 seed0/1/2, QA는 대화/haystack, ALFWorld는 gamefile 단위 bootstrap.
7. API usage, embedding, 정책 CPU 시간, raw/active/index 저장, 수집·학습 비용 포함. 가격 미설정 시 비용 절감 금액을 계산하지 않음.
8. 실제 답변 구절 recall과 source ID reference recall을 구분. 현재 코드의 후자는 상한 성격의 provenance metric이며 전자를 대신하지 않음.

ALFWorld 주 지표는 환경 성공률과 행동 수다. 선행 조건 위반률의 실제 환경 판정기는 추가 작업이며 현재 null이다. 제한된 질문/환경만 실행한 smoke test와 전체 공식 split을 구분한다.

## 완료 상태와 제출 일정

코어/API/adapter/수집·학습/QA 평가/AdaMEM 값 변환/테스트/초안은 구현했다. 실제 라벨 수집, 정책 학습, 공식 QA 결과, unseen 행동 결과, 정식 시스템 baseline, 같은 실현 압축률 분석과 원문 복원 비교는 수행 전이다. 준비된 코드를 실험 완료로 간주하지 않는다.

초록 2026-09-18, 본문 2026-09-25 AoE. [ICLR 2027 공식 CFP](https://iclr.cc/Conferences/2027/CallForPapers)

9/3–7 파일럿·선행연구·baseline 감사 → 9/8–12 라벨·정책·프로토콜 확정 → 9/13–18 QA/행동 검증·초록 → 9/19–22 반복 병합·비용·이식성 → 9/23–25 재현 확인·제출. 관측된 효과가 좁으면 논문 주장도 그 범위로 줄인다.
