# CertMem abstract / introduction 작성 근거

작성일: 2026-09-08. 요청된 다섯 논문의 아래 판본에서 본문과 부록을 읽었다. CertMem은 이 작업 폴더의 연구 가칭으로 사용했다. 새 실험은 실행하지 않았으며, 원고는 제안 단계다.

## 읽은 판본과 정확한 해석

| 논문 | 확인한 출판 상태·원문 | 핵심 근거 | 원고 작성에 반영한 구분 |
|---|---|---|---|
| RMM | [ACL 2025 Main, 24쪽](https://aclanthology.org/2025.acl-long.413/) | §4가 메모리 항목을 `(topic summary, raw dialogue)`로 정의한다. §5는 topic별 추출·병합, §6은 citation을 이용한 reranker 학습이다. Appendix D.2는 summary와 original turns를 함께 제공하며 citation은 original turns를 기준으로 평가한다. | 요약을 검색 키로 쓰고 선택된 원문을 읽는 방식은 신규성이 아니다. RMM의 학습 대상은 주로 retrieval ranking이다. |
| HiGMem | [ACL 2026 Findings, 10쪽](https://aclanthology.org/2026.findings-acl.1690/) | §3.1의 Turn은 원문+metadata, Event는 short summary+fact sheet이고 명시적 Turn index를 가진다. §3.3에서 직접 Turn 검색과 Event 기반 예측을 합친 뒤 LLM filtering을 한다. | 단순히 summary만 보고 한 번 Turn을 고르는 구조로 축약하지 않았다. Table 3과 Appendix B.3의 비용 절감은 비싼 답변 모델과 저렴한 중간 모델을 조합한 배치에서의 비용 추정이다. 최종 답변 토큰 절감과 전체 모델 토큰 절감은 다르다. |
| TierMem | [방법 원문: arXiv:2602.17913v1, 22쪽](https://arxiv.org/abs/2602.17913v1); [COLM 2026 accepted list](https://colm.cc/Conferences/2026/AcceptedPapers) | §2의 summary index는 immutable raw page로 연결된다. sufficiency router, linked-page 우선 확장, bounded additional retrieval, verified write-back을 포함한다. §3에는 paired S/R 결과 기반 supervision 및 cost-aware router training이 이미 있다. | 조건부 raw 접근, 비용을 고려한 router, 회수한 근거의 재저장만으로는 차별화되지 않는다. COLM 채택명은 **TierMem: Balancing Compressed Memory and Raw Evidence for Long-Horizon Agent Memory**이다. 직접 읽은 공개 방법 판본의 제목은 **From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents**이므로 BibTeX는 그 판본을 인용했다. COLM 최종 원고를 읽었다고 표시하지 않았다. |
| H-MEM | [EACL 2026 Main, 10쪽](https://aclanthology.org/2026.eacl-long.15/) | §3.1–3.2의 Domain → Category → Memory Trace → Episode, semantic vector와 self/child position indices, top-down routing을 확인했다. Episode에는 textual contextual memory와 profile도 포함된다. | 하위 기억으로 내려가는 위치 인덱스와 원문 provenance는 구별했다. 이 논문을 TierMem처럼 불변 원문 로그를 byte 단위로 보관하는 시스템이라고 단정하지 않았다. 주요 효율 분석은 검색 연산량·검색 시간이며 전체 LLM 토큰 비용과 동일하지 않다. |
| SEEM | [ACL 2026 Main, 17쪽](https://aclanthology.org/2026.acl-long.277/) | §3.2의 event frame과 fusion, §3.3의 relational quadruples, §3.4의 RPE 및 raw+event+fact context synthesis를 확인했다. | Eq. (6)은 관련 provenance의 합집합으로 설명하지만, §4.4의 실제 설정에서는 selective RPE로 최종 evidence set을 초기 검색 크기의 최대 두 배로 제한한다. 따라서 '항상 무제한으로 모든 event 원문을 읽는다'고 비판하지 않았다. extraction/fusion 비용과 오류 전파는 저자들이 Limitations에서 명시한다. |

로컬 PDF 경로·쪽수·SHA256은 `D:/MemoryData/tmp/pdfs/memory_reading_20260907/manifest.json`에 있다. 이번 작업에서 논문 내용을 바꾸거나 재작성하지 않았다.

## 채택한 논문 방향

가제: **CertMem: Evidence Residuals for Token-Efficient Agent Memory**.

핵심 질문은 **“어떤 누락을 지금 짧은 원문 구절로 보완하고, 어떤 누락은 질문이 들어올 때 추가로 읽을 것인가?”**이다. 요약의 출처만 보관하는 것으로 끝내지 않고, 제한된 검사에서 어떤 정보가 남았고 어떤 정보가 빠졌는지를 저장 단계와 읽기 단계에 공통으로 사용한다.

1. **Write:** 길이가 다른 summary 후보를 만든다. source-grounded proposition과 일부 조건·시간·참조 관계를 검사한다. 누락 때문에 생길 답변 손실을 줄이는 데 유용한 짧은 원문 구절을 residual로 남긴다.
2. **Read:** 요약·residual·남은 검사 항목을 이용해 추가로 읽을 source span을 선택한다. 일부 span을 읽었다고 source 전체가 복구된 것으로 처리하지 않고 다시 평가한다. 요약의 연결 바깥에 있는 근거를 찾기 위한 bounded raw search도 포함한다.
3. **Objective:** answer 모델 입력 길이만 줄이는 대신, 압축·검사·보완·검색·router·답변 생성에 소비한 실제 토큰을 모두 비교한다. 입력과 출력, 모델별 토큰, 모델별 비용은 따로 보고한다. raw archive와 metadata를 포함한 지속 저장량도 별도로 센다.

이것은 **검증할 연구 가설**이다. 다섯 논문의 구조를 결합했다는 사실이나 새로운 명칭 자체로 신규성이 입증되지는 않는다. 특히 TierMem도 S/R paired outcomes와 cost-aware training을 사용하므로, 'task loss로 router를 학습한다'는 점을 독립적인 최초 기여로 내세우면 안 된다. 결정적인 차별화 실험은 같은 예산에서 **누락 진단을 두 단계에 공유하는 residual 선택**이 단순 관련도·fact coverage·query-time routing보다 나은지를 보여주는 것이다.

추가로 [TRUSTMEM](https://arxiv.org/abs/2606.25161)의 memory transition 검증과 [DeMem](https://arxiv.org/abs/2605.10870)의 decision-loss 관점을 확인해 도입부에서 해당 선행을 인정했다. 이 두 논문은 요청된 다섯 편처럼 전 본문을 정독한 목록에는 포함하지 않는다.

## 기존 CertMem과 제안 사이의 경계

기존 `methods/certmem/scripts/run_system_v15.py`의 adaptive compression, residual sentences, provenance, selective expansion을 설계 출발점으로 삼았다. `research/certmem/iclr_differentiation_20260907_ko.md`와 `research/retentionmem/draft_ko.md`의 기존 논의도 반영했다.

현재 v15의 residual 선택은 주로 누락 fact 수와 token 수를 이용하며, read-time의 relevance-weighted fact-loss는 정답 오류 확률이 아니다. 이 원고에서 제안하는 downstream loss 학습, bundle 단위 선택, 원문 확장 후 재평가, 독립 validation은 구현·실험으로 완성할 부분이다. 원고는 이 기능들이 이미 검증되었다고 주장하지 않는다.

**CertMem이라는 가칭은 유지했지만, 개별 답변을 인증한다는 주장은 넣지 않았다.** audit은 검사한 항목에 대한 기록이다. 통계적 risk certificate를 추가하려면 독립 표본 단위, 정책 선택 편향, 검증기 오차, 분포 가정을 명시한 별도 방법·증명이 필요하다. 목표식의 `R_+ <= epsilon` 역시 현재 달성한 보장이 아니다.

## 실험 후 보완할 부분

- 같은 reader·dataset split·budget 아래 직접 재실행한 성능–전체 토큰 곡선. 서로 다른 원 논문의 표 숫자를 단순 비교하지 않는다.
- summary only, raw retrieval, TierMem 및 나머지 요청 baseline; residual 없음, random/relevance/fact-coverage residual, audit sharing 없음 등의 대조군.
- false stop, raw expansion 비율, answer correctness, 추가 span당 실제 손실 회복. retrieval recall은 정답률과 분리한다.
- write 비용을 포함해 질의 횟수별 이득이 생기는 구간. 반복 질문은 고정-memory 평가와 online write-back 평가를 구분하고 메모리 증가량도 센다.
- benchmark test 질문·정답은 write-time 검사 생성, residual 선택, threshold tuning에 쓰지 않는다. 의존된 질문들을 모두 독립 표본으로 간주하지 않는다.

실험 결과가 들어오기 전에는 abstract에 “outperforms”, “achieves state-of-the-art”, “reduces tokens by X%”와 같은 성과 문장을 추가하지 않는다.

## 파일

- `main.tex`: 요청한 영문 abstract와 introduction, references만 포함한 원고.
- `draft_en.md`: 같은 본문의 읽기·복사용 Markdown.
- `references.bib`: 공식 ACL Anthology 및 arXiv에서 확인한 서지정보.
- `iclr2027_conference.sty`, `iclr2027_conference.bst`: 공식 ICLR 2027 배포본.
- `source_manifest.json`: 이번에 받은 인용 정보와 공식 style 출처 기록.
- `main.pdf`: 컴파일된 작업 초안. 성능 평가 전이며 제출·채택된 논문이라는 의미가 아니다.
