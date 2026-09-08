# Reading and novelty notes

확인일: 2026-09-07. 요청한 다섯 편은 공개 원문의 본문·실험·한계·부록을 읽었다. 핵심 구조 그림도 확인했다. TierMem은 아래 명시한 공개 arXiv 판본을 읽었으며, COLM 최종 원고와 동일하다고 가정하지 않았다. 추가 논문들은 필요한 방법 절을 중심으로 확인했다.

본문은 독립적인 예시와 논리로 작성했다. 아래 메커니즘 요약은 기존 연구에 귀속된다. 제출본에는 일반 선행연구 주장에 맞는 인용을 연결하고 Related Work를 추가해야 한다. 본 노트는 신규성의 완전한 증명이나 모든 논문을 포괄한 systematic review가 아니다.

## 요청한 다섯 논문

| 논문·확인 판본 | 읽은 내용 중 이 제안과 관계있는 점 | DEB가 기존 기여로 인정해야 할 부분 |
|---|---|---|
| [RMM, ACL 2025](https://aclanthology.org/2025.acl-long.413/) · 24쪽 | topic summary와 대응 원문을 저장하며 add/merge를 수행한다. 생성기의 citation attribution으로 작은 reranker를 온라인 학습한다. | 원문 연결, 기억 갱신, 피드백 기반 검색 학습은 이미 있다. |
| [HiGMem, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1690/) · 10쪽 | event summary·fact sheet·turn 연결을 갱신한다. event 경유 탐색과 직접 turn 검색을 병행한다. | “모든 방법이 요약을 통과해야만 원문을 검색한다”는 비판은 부정확하다. event affiliation과 raw fallback 자체를 새 기여로 삼지 않는다. |
| [TierMem 공개 원고](https://arxiv.org/abs/2602.17913) · 22쪽 | 요약 sufficiency router, 원문 page pointer, 제한된 전역 검색, 선택적 근거 확인이 있다. 주 benchmark에서는 write-back을 끈다. | 원문을 필요할 때만 읽는 학습 router는 이미 있다. DEB는 여러 event assignment를 지속시키고 그 사이의 답 차이를 활용하는지를 검증해야 한다. |
| [H-MEM, EACL 2026](https://aclanthology.org/2026.eacl-long.15/) · 10쪽 | Domain→Category→Memory Trace→Episode와 positional index로 탐색한다. 외부 memory weight에 피드백을 반영한다. | 계층 탐색, index, confidence weight는 새 기여가 아니다. 모든 node weight를 구조적 대안의 확률로 해석해서도 안 된다. |
| [SEEM, ACL 2026](https://aclanthology.org/2026.acl-long.277/) · 17쪽 | event frame·관계 facts·원문 연결, 동일 사건 판정에 따른 fusion, reverse provenance expansion을 사용한다. | 가장 직접적인 event-binding 비교군이다. source pointer를 유지하는 것과 여러 배타적 사건 연결을 지속 상태로 유지하는 것을 구분해 비교한다. |

TierMem의 읽은 PDF 표제는 *From Lossy to Verified: A Provenance-Aware Tiered Memory for Agents*다. [COLM 2026 공식 채택 목록](https://colm.eventhosts.cc/Conferences/2026/AcceptedPapers)에는 *TierMem: Balancing Compressed Memory and Raw Evidence for Long-Horizon Agent Memory*로 올라 있다. 학회 판본의 방법이 공개 원고와 완전히 같다는 주장은 하지 않는다.

다섯 논문의 headline 점수는 backbone, 데이터 subset, adversarial 처리, 비용 계산이 서로 달라 그대로 비교하지 않았다. 특히 HiGMem은 직접 turn 검색을, TierMem은 bounded global search를 포함하므로 둘을 엄격한 summary-only gate라고 묘사하지 않았다. SEEM의 provenance expansion도 실험 구현의 passage budget 제한이 있다.

## 추가로 확인한 가까운 연구

**[T-Mem](https://arxiv.org/abs/2606.15405).** 공개 v1의 초록·방법·검색 절을 확인했다. memory에 미래 관련 상황을 표현하는 trigger를 쓰고 evidence와 검색 cue를 분리한다. 따라서 “원문에 도달하도록 예상 질문을 저장한다”를 새 아이디어로 선택하지 않았다.

**[Document Optimization for Black-Box Retrieval via Reinforcement Learning](https://arxiv.org/abs/2604.05087).** 방법 §3과 저자 설명을 확인했다. frozen retriever에서 문서를 다시 쓰고, 양성·음성 질의의 counterfactual ranking gain을 보상으로 사용한다. 단순한 “검색 성능과 다른 기억의 손해를 함께 측정해 key를 고친다” 역시 독립적인 새 기여로 주장하기 어렵다.

**[xMemory: Beyond RAG for Agent Memory](https://arxiv.org/abs/2602.02007).** 방법 §3 및 부록의 uncertainty expansion 정의를 확인했다. 구성 요소를 묶은 구조를 split/merge로 수정하고, reader predictive entropy 감소를 기준으로 원문을 확장한다. “기억 구조를 나중에 바꿀 수 있다”거나 “불확실하면 더 읽는다”는 주장만으로는 차이가 없다. DEB의 비교는 persistent competing assignments와 질문별 구조적 답 차이에 집중해야 한다.

**[Event and Entity Coreference using Trees to Encode Uncertainty in Joint Decisions](https://aclanthology.org/2021.crac-1.11/).** 초록과 §3을 확인했다. event/entity cluster tree로 여러 flat clustering의 불확실성을 표현하고 상호 의존성을 이용한다. uncertain event grouping은 2021년 이전에도 연구된 문제다. DEB의 후보 기여는 이를 발명했다는 것이 아니라, 여러 interaction에 걸쳐 보존할 불확실성과 해소할 근거를 downstream question utility와 비용으로 선택하는 memory policy다. 최종 하나로 확정하는 방법뿐 아니라 uncertain representation을 유지하도록 확장한 강한 대조군이 필요하다.

## 원고 인용 연결 지도

Introduction의 event organization·source preservation 문단에는 RMM, HiGMem, H-MEM, SEEM 중 실제 주장에 해당하는 논문을 연결한다. revisability·selective evidence acquisition 문단에는 xMemory와 TierMem을 연결한다. event coreference·uncertain clustering의 기존 기반을 언급한 문단에는 Yadav et al. (2021)을 연결한다.

Abstract에는 기존 논문 이름이나 표현을 넣지 않았다. Introduction 역시 사용자가 요청한 문제·방법 중심 초안으로 작성했으므로 in-text citation은 분리해 두었다. 이는 학술적 선행성을 주장하는 근거가 아니다. 범위가 정해진 제출본에서는 공통 기법과 비교 주장에 위 인용을 복원한다.

## 읽기 기록

원문별 URL, 페이지 수, SHA-256은 작업 폴더의 manifest.json과 extra_manifest.json에 기록했다. 추가 연구 전체를 완독했다고 주장하지 않는다. 본문 다섯 편을 읽은 것과 신규성 후보를 점검한 추가 방법 절 확인을 구분했다.
