# 지식·경험·작업 기억 논문 초안

제목: **Learning to Remember from Failure: Knowledge, Experience, and Working Memory**

사용자 요청에 맞춰 작성한 독립 영문 원고다. 기존 `brain_memory`와 `retentionmem` 원고는 수정하지 않았다. ICLR 2027 공식 style을 사용한다. 실제 제출 또는 심사 중이라고 표시하지 않는다.

## 중심 주장

장기 지식 bank는 출처가 있는 사실을 보존한다. 장기 경험 bank는 과거 실패에서 얻은 조건부 개선 절차를 보존한다. 현재 작업 기억은 확보한 근거, 부족한 정보, 이미 시도한 검색을 추적한다. 경험은 저장·검색·답변 방식을 바꿀 수 있지만, 정답의 사실적 근거로 사용하지 않는다.

실패 분석의 대상은 저장 누락, 검색 실패, 근거 사용 오류다. 한 구성요소씩 바꾸는 replay와 별도 validation을 통과한 절차만 경험 bank에 반영한다. source 기록은 지식 bank 내부에 있으며 네 번째 학습 메모리로 취급하지 않는다. 모델 가중치는 고정한다.

Recuris의 경험-작업 상태 결합, 실패 진단, 검증 후 수정과 겹치는 부분을 본문에 명시했다. 지식 bank의 encoding repair가 실제로 추가 가치를 만드는지는 실험할 가설이다. 메모리 수 자체를 신규성으로 주장하지 않는다.

## 실험 세 가지

| 실험 | 주요 내용 |
|---|---|
| 성능과 비용 | LoCoMo와 cleaned LongMemEval-S의 답변 품질, 근거 회수, latency, 구축·경험 학습·질의 비용 |
| 세 메모리의 기여 | 경험 제거, 작업 기억 제거, 한 저장소로 통합, 고정 규칙, encoding repair 제거 |
| 지속적 전이 | 경험 축적 곡선, 이전 능력 유지, LoCoMo ↔ LongMemEval 전이 |

R1/R2 또는 RQ1/RQ2 식으로 서술하지 않았다. 주요 독립변수만 본문에 남겼다. source 접근, retrieval/action budget, reader, embedding을 통제한다. planned primary stack은 Qwen3-8B + Qwen3-Embedding-4B이며 메모리 입력 예산은 4K다.

LoCoMo는 conversation 단위, LongMemEval은 공유 history/source 단위로 development/validation/test를 분리하는 5-fold cross-fitting을 제안한다. 이 점수는 공식 고정 bank 리더보드와 구분한다. 경험 generation이나 선택에 test 답변·근거·점수를 쓰지 않는다. cross-benchmark transfer에서는 target feedback을 사용하지 않는다.

## 현재 상태

초록부터 결론까지 작성된 **실험 전 단계의 방법 논문 초안**이다. 실제 구현·benchmark 실행·성능 수치는 포함하지 않는다. 수치 표나 우월성 주장은 실험 완료 후 채워야 한다. 현재 로컬의 구버전 Mem0 fact 제한 또는 LightMem 단순화 경로를 원 논문 재현 결과로 취급해서는 안 된다.

작성 완료와 실험 완료는 별개다. 원고가 제안한 효과, 통합 구현, 정확한 split manifest, 모델 snapshot 및 commit 고정은 후속 실험에서 확정해야 한다.

## 파일 및 빌드

- `main.tex`: 편집 가능한 본문
- `references.bib`: 확인한 1차 출처의 참고문헌
- `iclr2027_conference.sty`, `iclr2027_conference.bst`: 공식 배포 파일
- `source_manifest.json`: 참고문헌과 style 출처 기록
- `build.ps1`: 로컬 Tectonic 또는 표준 pdfLaTeX/BibTeX 빌드

현재 workspace에서는 `./build.ps1`로 PDF를 다시 만들 수 있다. Overleaf에는 배포한 source ZIP을 업로드하고 `main.tex`를 main document로 지정하면 된다.

ICLR style 출처: https://iclr.cc/Conferences/2027/AuthorGuidelines

PDF는 `D:/MemoryData/output/pdf/three_memory_iclr.pdf`에 배포한다.
