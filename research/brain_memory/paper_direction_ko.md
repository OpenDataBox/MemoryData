# Brain-Inspired Multi-Timescale Memory for Language Agents

`framework.png`와 `framework_locomo_example.png`를 출발점으로 새로 작성한 방법론 논문이다. 작업명은 **MTM (Multi-Timescale Memory)**이다. 기존 Related Work의 38개 인용을 유지하면서 시스템 비교 연구의 포지셔닝을 새 방법론에 맞게 바꿨다.

## 논문을 관통하는 주장

**경험을 지식으로 바꾸고 답변에 활용할 때 만들어진 근거 연결을, 나중에 잘못된 지식을 수정하는 경로로도 사용한다.**

빠른 episodic encoding, 질의 시 evidence binding, 느린 semantic/procedural consolidation을 분리한다. 어떤 경험이 어떤 지식과 절차를 뒷받침했고 어느 답변에서 사용됐는지 연결한다. 새로운 관찰이 기존 지식을 반박하면 영향받는 항목을 즉시 stale로 표시하고, 배경 replay에서 해당 항목을 검증·갱신한다. 이 구체적 결합을 **evidence-bound reconsolidation**으로 정의했다.

## 기존 접근의 문제와 선행연구를 인정할 부분

### 관련 정보를 찾는 것과 필요한 관계를 복원하는 것은 다르다

책 이름과 사람 이름을 찾아도 누가 누구에게 추천했고, 나중에 누가 그 책을 읽었는지는 별개다. 세션별 요약은 앞선 추천과 나중의 “그 책” 사이 연결을 약화시킬 수 있다. 필요한 단위는 단어 집합이 아니라 근거가 붙은 관계다.

다만 [HippoRAG](https://arxiv.org/abs/2405.14831)와 [HippoRAG 2](https://arxiv.org/abs/2502.14802)가 이미 연관 검색을 다룬다. “기존 방법은 모두 독립 문서 검색만 한다”는 식으로 쓰지 않았다. 우리 질문은 검색 시 만든 근거 연결을 후속 수정까지 일관되게 활용할 수 있는지다.

### 사실을 수정해도 그 사실에서 파생된 기억은 낡은 상태로 남을 수 있다

현재 주소를 수정해도 예전 주소를 전제로 만든 요약이나 계획은 남을 수 있다. 반대로 최신 값으로 전부 덮어쓰면 과거 시점의 질문에 답할 수 없다. 현재 유효성, 역사적 범위, 파생 항목의 버전을 함께 관리해야 한다.

[Zep](https://arxiv.org/abs/2501.13956)의 시간 정보, [A-MEM](https://arxiv.org/abs/2502.12110)의 memory evolution, [Hindsight](https://arxiv.org/html/2512.12818v1)의 원 사실 변경에 따른 observation 재생성은 이미 존재한다. 새 주장은 이를 없던 기능처럼 소개하는 것이 아니라, **episode → semantic claim → procedure condition/use trace**에 걸친 dependency와 **즉시 invalidation / 느린 repair**의 결합이다. 이 결합의 추가 가치는 아직 실험으로 검증해야 한다.

### 답변 생성과 성공 피드백을 혼동하면 잘못된 기억을 강화한다

자연스러운 답변이나 같은 모델의 동의를 성공 신호로 쓰면 자기 확인 순환이 생긴다. 새로운 관찰, 명시적 사용자 정정, 실제 도구 결과처럼 피드백 출처를 구분해야 한다. 평가 정답은 시스템의 피드백이 아니다.

[Reflexion](https://arxiv.org/abs/2303.11366)과 [MemRL](https://arxiv.org/abs/2601.03192)이 이미 피드백을 활용한다. 우리 초점은 피드백이 어떤 근거·버전에 연결되고, 어디까지 수정해야 하는지 명시하는 데 있다.

## 그림을 구체적인 방법으로 바꾼 부분

- Sensory buffer / attention: 입력을 보존하고 검색·replay 우선순위를 정한다. 중요도가 낮다는 이유로 원문을 즉시 삭제하지 않는다.
- Working memory: 현재 답변에 실제 노출되는 제한된 문맥이다.
- Episodic memory: 화자·시점·원문 구간을 가진 사건 기록이다.
- Semantic memory: 근거, 유효 시간 범위, 버전을 가진 주장이다.
- Procedural memory: 적용 조건과 확인 가능한 동작을 가진 검색·행동 절차다. 정답 문자열 캐시가 아니다.
- Index/router + fusion: 필요한 기억 유형을 선택하고 관계·시간·출처를 확인한 evidence bundle을 문맥 예산 안에 넣는다.
- Replay + reconsolidation: 실제로 영향받는 항목을 재검토한다. 정정된 사실을 사용했다는 이유만으로 모든 절차를 폐기하지 않고, 그 사실에 의존하는 조건과 검증 기록을 구분한다.

이들은 **새 프레임워크의 구체적 설계**이지, 이미 전체 구현과 성능 검증이 끝났다는 설명이 아니다. 사람 뇌와의 관계도 기능적 영감으로 제한한다. multimodal cognition과 일반적인 tool-use 성능은 현재 QA 설계로 주장하지 않는다.

## LoCoMo 예시에서 확인한 것

`conv-26`의 2023년 7월 12일 `D7:11`에는 Caroline의 Becoming Nicole 추천이 있고, 10월 13일 `D17:10`에는 Melanie가 이전에 추천받은 책을 읽고 있다는 발언이 있다. 해당 질문의 정답과 두 evidence ID가 실제 데이터와 일치함을 확인했다. [공식 평가 코드](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py) 기준 category **1은 multi-hop**이다.

두 PNG는 원본을 변경하지 않고 **본문에 배치했다. Figure 1은 방법론 도입부, Figure 2는 LoCoMo 예시 설명 안에 있다.** 부록에는 그림을 중복 수록하지 않는다.

그림의 **LoCoMo Gold Check → Outcome Feedback → Reconsolidation**은 그대로 실험에 구현하면 정답 누수 위험이 있다. 캡션과 본문에 gold check가 evaluator 전용임을 명시했다. 원래 대화를 시간순으로 읽는 과정에서는 새로운 관찰에 따라 기억을 갱신할 수 있다. 이후 각 평가 질문은 같은 history-only snapshot의 독립 복사본에서 답하고, 질문·답변으로 바뀐 상태는 다음 질문에 전달하지 않는다.

“resolve relation + coreference”라는 고정 프롬프트만으로 절차를 학습했다고 할 수 없다. 경험에서 유도한 절차를 같은 길이의 수작업 절차와 비교해야 한다. 이 예시는 설계 설명이며 실제 시스템 성공 로그가 아니다.

## 두 벤치마크로 끝내는 핵심 실험

- **LoCoMo:** 10개 대화, 공식 category 1–4의 1,540문항. 공식 category별 F1을 적용하고 문항 평균을 보고한다. Category 5는 평가 범위에서 제외한다고 명시한다.
- **LongMemEval:** cleaned S의 500문항. 공식 question-type 및 abstention rubric으로 정확도를 평가한다. M, oracle, V2 결과와 섞지 않는다.
- **공통 조건:** reader·embedding·memory-writing model을 지원 가능한 범위에서 통일하고, 최종 reader에 들어가는 기억은 4K tokens로 고정한다. 모델/데이터 버전과 judge 설정을 실행 전에 고정한다.

| 실험 | 비교 | 결과물 |
|---|---|---|
| 전체 성능과 비용 | raw hybrid RAG, LightMem, HippoRAG 2, Hindsight, MTM | 두 벤치마크 점수, online p50/p95 지연, 구축·replay·검증까지 포함한 토큰 비용 표 1개 |
| 핵심 모듈 기여 | 전체 모델에서 procedural reuse, conflict-aware fusion, evidence-bound reconsolidation을 각각 하나씩 제거 | 동일 예산 ablation 표 1개. 전체/제거 모델은 동일한 3개 seed 사용 |
| 어떤 질문에서 좋아지는가 | 같은 실행 결과의 LoCoMo multi-hop/temporal, LongMemEval multi-session/temporal/knowledge-update/abstention 분석 | 유형별 성능과 실제 reader에 노출된 근거 회수 분석 그림 1개 |

별도 합성 benchmark, 인위적인 correction 데이터, 추가 질문은 만들지 않는다. 세 번째 분석도 앞의 실행을 재사용한다. LongMemEval의 abstention은 별도 question type이 아니라 공식 question-ID의 `_abs` 표식으로 구분한다. 지식 갱신 점수가 좋아져도 임의의 dependency repair나 답변 이후 lifelong learning까지 검증됐다고 확대 해석하지 않는다.

Procedure를 학습했다고 주장하려면, benchmark와 독립된 development 대화에서 유도하고 사전에 동결한 절차를 같은 예산의 고정 seed 규칙과 추가 비교한다. 단순히 procedure를 없앤 모델보다 낫다는 결과만으로 학습 효과를 주장하지 않는다.

원문 화자·날짜·제공된 텍스트 caption은 유지하되, 정답·evidence ID·answer-session ID·`has_answer` 표식·사전 요약은 memory 구축 입력에서 제외한다. 근거 회수는 실제 reader에 보인 원문으로 평가한다. 일부 session만 전달했다면 answer-bearing turn이 포함됐는지 확인한다. Gold annotation은 evaluator만 사용한다. 실행 실패를 분모에서 빼지 않고 별도 보고하며, 대화/독립 history 단위 paired bootstrap 95% 구간을 낸다. LoCoMo의 독립 대화가 10개뿐이라는 한계도 명시한다.

데이터/평가 근거: [LoCoMo 공식 저장소](https://github.com/snap-research/locomo), [LongMemEval 공식 저장소](https://github.com/xiaowu0162/LongMemEval), [LongMemEval 공식 평가 코드](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py).

## 제출 수준으로 만들기 위해 남은 것

“뇌에서 영감을 받았다”, “sleep 단계가 있다”, “그래프가 있다”, “요약 손실이 있다” 자체는 신규성이 아니다. [LightMem](https://arxiv.org/abs/2510.18866), Hindsight, [TierMem](https://arxiv.org/abs/2602.17913), [TrustMem](https://arxiv.org/abs/2606.25161)과의 겹침을 원고에 명시했다.

채택 가능성을 좌우할 증거는 통합 구현, 실제 전체 비교 결과, 동일 예산 ablation이다. 현재 성능 수치는 없으며, 원고는 **초록부터 결론까지 작성된 결과 전 단계의 영문 초안**이다. 개선 수치·유의성·성공한 실험 그래프는 만들지 않았다.

ICLR 2027 공식 style을 사용하고, 미제출 초안에 맞춰 working manuscript임을 표시했다. [공식 작성 지침](https://iclr.cc/Conferences/2027/AuthorGuidelines)의 본문 9쪽 제한과 [AI 사용 공개 정책](https://iclr.cc/Conferences/2027/AIPolicyForAuthors)을 반영했다. 최종 제출 전에는 저자가 주장, 참고문헌, 실제 AI 사용 및 검토 내역을 확인해야 한다.
