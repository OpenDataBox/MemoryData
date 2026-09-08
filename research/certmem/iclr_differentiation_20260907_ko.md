# CertMem: ICLR를 위한 차별화 방향 검토

작성: 2026-09-07. 로컬 CertMem 구현, RetentionMem 초안, 아래에 연결한 공개 논문을 확인했다. 이 문서의 우선순위는 연구 판단이며, 채택 확률이나 실험 성능을 뜻하지 않는다. 이번 작업에서는 모델 실험을 실행하지 않았다. 검색한 로컬 경로에서 CertMem 실측 결과 파일은 확인하지 못했다.

**추천하는 중심 질문은 “제한된 메모리를 반복적으로 압축할 때, 어떤 수행 능력의 손실을 얼마나 허용할 것인가?”이다.** 현재의 adaptive compression + residual + raw expansion을 모두 독립적인 신규 기여로 내세우기보다, 하나의 손실 통제 원리와 그 원리를 실현하는 작은 보존 장치를 중심으로 묶는 편이 설득력 있다.

## 1. 현재 자산과 이미 겹치는 부분

현재 구현에는 세션별 사실 추출·함의 판정, 압축 cap 선택, 누락 사실에 대응하는 원문 문장 추가, 출처를 따라 원문 확장, query reformulation, 인물 profile, 예산별 비교 및 ablation 실행기가 있다. [CertMem v15](../../methods/certmem/scripts/run_system_v15.py)

또한 기존 [RetentionMem 초안](../retentionmem/draft_ko.md)에는 이미 수행 회귀 예측, 잔여 구절 선택, 반복 병합, 압축기·reader 이식성이라는 구상이 있다. 따라서 아래에서 이 요소를 언급하는 것은 새로 발견한 아이디어라는 뜻이 아니다. **CertMem에서 구체적인 보장·알고리즘·실험으로 완성할 후보**라는 뜻이다.

| 가까운 선행 연구 | 확인한 핵심 내용 | CertMem에서 이것만 주장하면 약한 부분 |
|---|---|---|
| [TrustMem, 2026-06](https://arxiv.org/html/2606.25161v1) | coverage/preservation/faithfulness를 평가하는 Memory Transition Verifier, transition preference를 이용한 RL | “메모리 갱신을 검증한다”, “누락·변조·환각을 줄인다”, verifier reward를 추가한다 |
| [TierMem, 2026-02](https://arxiv.org/abs/2602.17913) | summary-first retrieval, 필요시 immutable raw logs로 escalation, provenance와 verified write-back | summary/raw 두 계층, 출처 링크, 필요할 때만 원문을 다시 읽기 |
| [TriMem, 2026-05](https://arxiv.org/abs/2605.19952) | 원문·atomic facts·profiles의 공존, prompt optimization | 세 종류 메모리, profile 추가, 사실만 저장하면 세부정보가 사라진다는 주장 |
| [DeMem, 2026-05](https://arxiv.org/html/2605.10870v1) | decision-centric rate-distortion, decision conflict를 이용한 partition refinement와 regret 분석 | “설명 보존보다 decision 보존”, rate-distortion 수식, safe merging이라는 표어 |
| [Principled Context Engineering, 2025-11](https://arxiv.org/abs/2511.17908) | conformal filtering으로 supporting evidence coverage를 제어하며 RAG context 축소 | relevance threshold에 conformal calibration을 적용했다는 것만으로 신규성 주장 |
| [SafeCommit, 2026-08](https://arxiv.org/abs/2608.04289) | 가능한 world 집합을 이용한 action certificate, 불확실하면 probe/fallback, representation error를 구분한 보장 | 불확실한 메모리로 행동하기 전에 인증하고 필요시 탐색한다는 것만으로 신규성 주장 |
| [RECON, 2026-07](https://arxiv.org/abs/2607.16716) | evidence chain, cascading invalidation, conflict, counterfactual, temporal constraints 평가 | “multi-hop이나 갱신 전파를 처음 평가한다”는 주장 |

TierMem은 확인한 [ICLR 2026 MemAgent workshop 판본](https://openreview.net/attachment?id=dJgeY3Awrv&name=pdf)이 있다. 이를 ICLR main-track 채택 사례로 표현하면 안 된다. 다른 논문의 출판 상태도 여기서는 arXiv 공개 시점과 방법 내용만 사용한다.

## 2. 차별화 후보 8개

| 번호 | 방향 | 연구상 장점 | 현재 코드 재사용 | 핵심 위험 | 판단 |
|---|---|---|---|---|---|
| 1 | 실제 답변 회귀의 통계적 통제 | Cert라는 이름과 측정 대상이 일치 | 높음 | LTT 단순 적용으로 보일 수 있음 | 중심 원리 후보 |
| 2 | 원문 재접근 없이 동작하는 최소 잔여 근거 보존 | 보존 장치와 실제 저장 제약이 명확 | 높음 | 유한 용량에서 목표가 불가능할 수 있음 | 가장 현실적인 메커니즘 후보 |
| 3 | 사실 집합보다 조건·시간·관계의 보존 | coverage가 실패하는 이유를 설명 | 중간 | graph memory와 구분해야 함 | 가장 설명력 있는 강화안 |
| 4 | 반복 압축 전체의 손실 예산 | 장기 메모리 고유 문제를 직접 다룸 | 중간 | sequential 보장이 보수적일 수 있음 | 이론·장기 실험 강화안 |
| 5 | 검증기 자체의 오차를 반영한 인증 | 생성된 검사에 대한 과적합을 다룸 | 중간 | 독립 감사 비용 | 인증을 주장하면 필수에 가까움 |
| 6 | 압축의 반례를 찾는 능동 검사 | 희귀하지만 결정적인 정보 손실 탐지 | 중간 | synthetic probe 성능으로 끝날 수 있음 | 2·3을 위한 보조 알고리즘 |
| 7 | 정보 수정 시 영향받는 인증의 선택적 무효화 | 오래된 사실 보존과 올바른 갱신을 양립 | 중간 이하 | provenance/DB 기능처럼 보일 수 있음 | 별도 장기 프로젝트 후보 |
| 8 | 여러 host에 적용하는 통제 계층 | 특정 memory architecture 의존을 줄임 | 높음 | framework 논문 수준으로 보일 수 있음 | 독립 기여보다 검증 축 |

### 1) 사실 coverage 대신 실제 답변 회귀를 인증

**논문 한 문장:** “CertMem은 메모리 변경으로 추가되는 수행 손실을 목표 수준 이하로 통제하면서 저장·읽기 비용을 줄인다.”

원문의 사실 100개 중 99개를 보존해도, 빠진 한 개가 모든 평가 질문의 정답을 결정하면 coverage 99%와 QA 회귀 100%가 함께 가능하다. 추가적인 질문 분포·근거 가정 없이는 사실 누락률이 답변 손실 상한이 되지 않는다. 이를 과장된 새 불가능성 정리로 내세우기보다는, 센서 선택을 정당화하는 반례와 실제 진단 실험으로 사용한다.

정의할 대상의 예:

\[
L_\theta(H,q)=\left[\ell(A(M_\theta(H),q),y(H,q))-\ell(A(M_{\rm ref}(H),q),y(H,q))\right]_+\in[0,1].
\]

`A`에는 고정한 reader와 retrieval 절차가 포함된다. `M_ref`는 명시한 비교 정책이며 원문 기반 시스템도 완벽한 oracle은 아니다. 따라서 절대 정답률도 함께 보고한다. 새 정보가 정정을 포함하면 전후 모두 같은 최신 관측 및 해당 질문의 시간 범위에 맞는 정답으로 평가한다.

원하는 통계적 형태는 독립 calibration 표본으로 선택한 정책에 대해

\[
\Pr_{D_{\rm cal}}\left\{\mathbb E_{(H,q)\sim\mathcal D}[L_{\hat\theta}(H,q)]\le\epsilon\right\}\ge 1-\delta.
\]

이는 **정해진 분포에서 정책의 평균 손실에 대한 보장**이다. 개별 질문의 정답 보장, 임의의 분포 이동, 임의의 reader 교체에 대한 보장이 아니다. [Learn then Test](https://arxiv.org/abs/2110.01052)는 출발점이며, 이 표준 결과 자체는 신규 기여가 아니다.

구현은 개발 데이터에서 위험 예측기·잔여 정보 선택기를 학습하고, 별도 calibration에서 완성된 정책 후보를 검정하고, untouched test에서 검증하는 방식이 적절하다. adaptive cap 선택과 residual repair를 포함한 **최종 정책 전체**가 calibration 대상이어야 한다. benchmark test 질문은 write-time 입력이나 threshold 선택에 쓰지 않는다. Offline calibration에 task QA를 썼다면 그 사실은 명시한다.

**결정적 실험:** 같은 실제 저장·읽기 예산에서 coverage gate, reader confidence gate, 직접 회귀 gate를 비교한다. 평균 QA, positive regression, risk–cost curve, 인증 실패율, fallback 비율을 함께 본다. 수차례의 독립 calibration/test 재표집으로 목표 위험 위반 빈도를 추정하고 불확실성을 보고한다.

**차별화 조건:** 2번의 잔여 근거 선택이나 4번의 반복 압축처럼 memory 특유의 난점을 함께 해결해야 한다. “TrustMem + LTT”에 그치면 약하다.

### 2) 원문을 다시 읽지 않아도 되는 최소 잔여 근거

**논문 한 문장:** “삭제 전에, 요약만으로는 유지할 수 없는 답변 능력에 필요한 작은 근거 집합을 찾아 함께 보존한다.”

요약 S와 잔여 근거 Z만 다음 시점으로 넘긴다. 원문 전체를 다시 검색할 수 없는 실험에서 `(S,Z)`의 총 저장량, profile, 인덱스, 출처 metadata와 인증 상태까지 센다. source ID만 남기는 것은 근거 내용의 보존이 아니다.

현재 lost-facts-per-token 점수를 **추가 구절당 회복되는 수행 능력**으로 발전시킨다. 중복 구절의 한계 이득을 다시 계산하고, 하나의 사실에 여러 문장이 필요할 때는 문장 묶음을 선택한다. 생성한 답 자체를 기억시키는 지름길보다, 해당 답을 지지하는 원문 구절을 보존한다.

후보 목적식:

\[
\min_{S,Z} C_{\rm store}(S,Z),\quad \text{s.t. }U_{\rm cal}(S,Z)\le\epsilon.
\]

여기서 개별 메모리의 `U_cal`이 유효한지는 별도 문제다. 초기 구현에서는 개별 샘플에 점별 보장을 주장하지 않고, 이런 선택을 수행하는 정책의 평균 회귀를 1번 절차로 calibration한다.

**결정적 실험:** write 후 읽기 프로세스를 재시작하고, 직렬화된 제한 크기의 메모리만 전달한다. 동일 byte/token 한도에서 summary, all-facts truncation, recency, random spans, salience, fact-coverage residual, task-risk residual을 비교한다. 원문 접근 횟수는 0이어야 한다. 이미 개발한 residual 예산 비교를 재사용할 수 있다.

**선행과의 차이:** TierMem은 필요할 때 원문에 접근한다. DeMem의 확인한 정식화는 query-aware answer-time state와 inference-time footprint를 중심으로 한다. CertMem 후보는 **미래 질문을 보기 전의 영구 보존 선택과 이후의 원문 비가용성**을 명시한다. DeMem 전체를 원문 무제한 시스템이라고 단정할 필요는 없으며, 동일 제약 아래 비교할 구현을 정확히 정의한다.

**실패 조건:** 목표 위험과 용량이 양립하지 않으면 infeasible로 보고해야 한다. raw fallback으로 몰래 저장 한도를 넘기거나, 답변을 모두 포기해서만 위험을 낮추면 성공으로 평가하지 않는다. 모든 이력의 모든 미래 질문을 작은 고정 메모리로 보존할 수 있다는 주장은 하지 않는다. 이 설정은 privacy 보장이 아니라 **원문 재접근 금지와 저장 예산의 실험 조건**이다.

### 3) 조건·시간·관계의 결합을 보존하는 인증

**논문 한 문장:** “개별 사실의 높은 보존율 아래에서도 사라지는 추론 관계를 감지하고, 그 관계를 유지하는 최소 근거를 보존한다.”

예: “민수는 금요일에 참석할 수 있다”와 “승인된 경우에만 가능하다”에서 인물·요일·행사명을 남겼어도 조건이 빠지면 행동이 바뀐다. 또는 A가 B의 별명이고 B가 C를 예약했다는 두 근거가 함께 있어야 A의 예약 장소에 답할 수 있다.

기억 단위를 `(주체, 사건, 시간, 극성, 조건, 출처, 연결 관계)`로 검사하고, 위험을 단일 사실별 점수에만 배분하지 않는다. 현재의 window 사실 추출에 cross-window 조건/시간/참조 묶음을 추가하고, source bundle별 손실과 복구를 측정한다.

**결정적 실험:** 같은 사실 보존율·저장량에서 관계만 바꾸거나 삭제한 matched pair를 만든다. 난도를 1→2→4 hop, 조건 중첩, 시간 범위 충돌로 올린다. controlled truth graph에서 exact correctness를 먼저 측정하고, LongMemEval 및 [RECON](https://arxiv.org/abs/2607.16716)처럼 실제로 연결을 요구하는 외부 평가로 확장한다.

**알고리즘상의 진짜 난점:** 두 구절을 함께 넣어야 효과가 생기면 이득은 보완적이다. 일반적인 greedy set cover의 근사 보장을 그대로 붙일 수 없다. 작은 bundle enumeration, beam search, 작은 문제의 exact optimizer를 비교해 선택 품질을 검증한다.

**차별화 조건:** graph를 만들었다는 것이 아니라, “어떤 결합 손실이 사실 coverage에 보이지 않는가”와 “같은 예산에서 그 손실을 어떻게 복구하는가”가 결과로 나와야 한다. TriMem·TiMem·RECON과의 단순 구조 중복을 피한다.

### 4) 반복 압축 전체에 대한 손실 예산

**논문 한 문장:** “한 번의 압축을 통과한 메모리도 반복 재압축에서 망가질 수 있으므로, 장기 운용 전체의 손실 예산을 관리한다.”

오래된 요약을 새 세션과 합쳐 다시 압축하고, 매번 허용 손실을 새로 리셋하지 않는다. 예산이 부족하면 더 큰 잔여 근거를 유지하거나 압축을 연기한다. 총 저장 한도가 고정된 경우에는 연기·fallback도 한도를 지켜야 하며, 불가능한 구간을 명시한다.

현재 v15의 host는 세션을 따로 압축해 추가하고, 전체 오래된 메모리를 반복 병합하는 실험과는 다르다. 따라서 진짜 recurrent consolidation loop가 필요하다.

**결정적 실험:** 원래 기록으로 매번 새로 만드는 방식과, 직전 압축 상태만으로 1·5·20·50회 재압축하는 방식을 비교한다. 고정 저장량에서 누적 회귀, 오래된 질문 유지율, worst-group 오류, 비용, 목표 달성 불가능 빈도를 그린다. 이 숫자는 실험 설계 지점이며 예상 결과가 아니다.

**이론 범위:** 고정 평가 대상에서 단계별 변화의 누적을 다루는 결과부터 시작하고, 새 정보가 들어와 정답이 바뀌는 경우를 분리한다. 조건부 가정·정책 적응·calibration 갱신을 명시한 뒤 horizon 수준 통제를 목표로 한다. 단순 union bound만으로 신규성이 생기지는 않으며, 보수성 때문에 아무것도 압축하지 못하는지 확인해야 한다. [bounded observations의 confidence sequence 연구](https://arxiv.org/abs/2010.09686)는 도구이지 memory 성능을 자동 보장하는 정리가 아니다.

### 5) “검증기가 맞는가?”를 포함한 인증

**논문 한 문장:** “메모리 압축과 동일한 검증 점수에 반복 적응할 때 발생하는 거짓 통과를, 독립 감사와 오차 보정으로 통제한다.”

사실 추출기가 중요한 조건을 아예 뽑지 않으면 entailment 검증도 그것을 검사하지 않는다. 같은 LLM이 압축·추출·검증을 하면 공통 실패가 생길 수 있다. 검사 점수를 최대화한 후보에서 검증기 오류가 더 커지는 selection effect도 확인해야 한다.

후보 생성과 최종 감사를 분리한다. 감사에는 생성된 fact list뿐 아니라 별도로 추출·검토한 원문 근거 및 독립 task 질문을 사용한다. 숫자·날짜·부정·조건의 controlled edit에는 가능한 부분에서 결정적 checker를 붙인다. 불명확한 부분은 미검증으로 남긴다.

**결정적 실험:** 독립 judge 및 사람의 층화 표본 검토로 false-pass를 측정한다. 검사 점수를 비슷하게 유지하면서 핵심 조건을 바꾼 후보, 생성기/검증기 동일·상이 조합, 후보 탐색 횟수 증가를 비교한다. 감사 토큰·사람 검토 비용도 보고한다.

**차별화 조건:** TrustMem verifier를 단순히 더 큰 모델로 바꾸는 수준을 넘어, **최종 선택된 메모리에서 발생하는 검증 누락과 후보 선택 편향**을 규정하고 실제 task risk에 연결해야 한다. SafeCommit도 representation error를 구분하므로 “오차 항을 하나 더한 보장” 자체를 최초 기여로 내세우지 않는다.

### 6) 압축의 반례를 찾는 능동 검사

**논문 한 문장:** “일반적인 기억 질문을 무작위로 만드는 대신, 원문과 압축본의 차이를 드러내는 질문을 찾아 적은 검사 비용으로 위험을 탐지한다.”

검사 생성기에 원문과 후보 요약을 주고, 요약을 구체적으로 반박할 수 있는 날짜·예외·수량·참조 질문과 원문 근거를 요구한다. 그 질문으로 실패한 구절 묶음을 보존하고 다시 검사한다. 원문에 답이 분명히 없는 질문은 제거한다.

이는 write-time에 아직 알려지지 않은 실제 사용자 질문을 사용한다는 뜻이 아니다. 진짜 test QA는 끝까지 감춘다. 반례 생성에 사용한 검사를 final certificate용 독립 평가처럼 재사용하지 않는다.

**결정적 실험:** 같은 검사 호출 수에서 random probes, typed probes, fact enumeration, counterexample probes를 비교한다. gold 근거 삭제 감지율, 실제 QA 회귀 탐지율, 잔여 구절 효율을 평가한다. 단순히 생성된 시험문제 점수만 오르는 경우는 실패다.

**차별화 조건:** generic self-refinement로 보일 수 있어 단독 주제보다 2번 또는 3번의 효율적인 검사 알고리즘으로 쓰는 편이 낫다. 장기 목표는 학습 가능한 challenger·repairer지만, 초기부터 RL 전체를 추가할 필요는 없다.

### 7) 수정된 사실에 연결된 인증의 선택적 무효화

**논문 한 문장:** “새 정보가 들어오면 영향받는 파생 기억과 인증만 다시 검증하여, 유효한 지식의 보존과 올바른 수정을 동시에 처리한다.”

예: 3월 서울 주소가 6월 부산으로 바뀌면 현재 배송지는 바뀌어야 하지만 3월의 거주 이력까지 지우면 안 된다. 주소로부터 파생된 배송 계획의 인증은 무효화하고, 관계없는 취향 정보는 유지한다.

각 근거의 유효 시간과 의존성을 기록하고, certificate가 어느 source/version을 전제로 하는지 저장한다. 바뀐 의존 경로만 재평가하고, 독립된 근거로 여전히 지지되는 결론은 보존한다.

**결정적 실험:** LongMemEval knowledge-update, controlled correction/retraction, RECON cascade 과제에서 stale-answer 오류와 불필요한 삭제, 재검증 비용을 함께 측정한다.

**한계:** provenance·temporal graph 자체는 신규성이 약하다. retrieval 순위가 전역적으로 바뀌면 “변경 지역 밖은 영향 없음”이 깨지므로 locality 가정이나 routing 영향 검사까지 필요하다. 구현 범위가 커서 당장 주 논문에 모두 넣는 것은 권하지 않는다.

### 8) 여러 메모리 시스템에 붙는 통제 계층

**논문 한 문장:** “새 메모리 구조를 강제하지 않고, host가 제안하는 변경을 동일한 손실 계약으로 선택·보완한다.”

host 인터페이스를 propose → evaluate → repair → commit으로 맞추고, Mem0·LightMem·다른 압축기에 같은 제어기를 적용한다. host-native retrieval을 유지하는 end-to-end 실험과, 동일 retrieval 아래 compression만 바꾸는 메커니즘 실험을 분리한다.

**결정적 실험:** 실제 upstream host 두 개 이상과 reader 두 계열에서 각 host 단독 대비 개선을 측정한다. leave-one-host-out 전이와 reader 교체 후 재calibration을 별도로 보고한다. “적은 재calibration으로 작동한다”와 “재calibration 없이 통계 보장이 유지된다”는 다른 주장이다.

**차별화 조건:** 이것만으로는 middleware/framework 기여에 머물 수 있다. 1·2·3의 원리가 구조와 모델을 넘어 작동한다는 증거로 쓰는 것이 좋다.

## 3. 추천하는 논문 구성

### A. 현재 자산을 가장 잘 살리는 구성 — 우선 추천

가제: **CertMem: Risk-Controlled Memory Compaction with Minimal Evidence Residuals**

핵심은 1 + 2, 5는 보장을 위한 필수 검증, 4는 반복 압축 실험으로 사용한다. 첫 단계부터 4의 완전한 online theorem까지 약속하지 않는다.

기여를 세 개로 압축한다:

1. fact coverage와 실제 수행 회귀 사이의 간극을 정의하고, 최종 압축·보완 정책의 위험을 calibration하는 문제 설정.
2. 원문 재접근이 없는 실제 저장 제약 아래, 누락 능력을 회복시키는 작은 근거 집합 선택.
3. 독립 감사와 반복 압축에서 위험–비용 tradeoff가 유지되는지 검증하는 실험.

주요 그림은 “accuracy 최고점” 하나보다 **동일 총 저장량에서 목표 회귀 위험을 만족하는 비용 곡선**, **반복 압축 횟수에 따른 성능**, **certificate 목표와 실제 위반 빈도**가 적절하다.

### B. 사실 coverage 센서가 실제 QA에 잘 연결되지 않을 때

가제: **CertMem: Preserving Reasoning Dependencies under Memory Compression**

3 + 2를 중심으로, 6을 검사 알고리즘에 사용한다. 핵심 발견은 “사실은 남아 있는데 사실 사이의 조건·시간·연결이 사라진다”이다. 제어된 반례와 실제 multi-hop 평가가 이어져야 한다. 이 경우 통계 보장이 성립하지 않으면 calibrated라는 표현을 줄이고, relation-preserving mechanism의 효과를 중심으로 주장한다.

### C. 긴 연구 기간과 충분한 독립 데이터가 있을 때

가제: **CertMem: Controlling Cumulative Forgetting in Continually Updated Memory**

4 + 7을 중심으로 5를 포함한다. 큰 가능성이 있지만 recurrent update 구현, 동적 정답 정의, 의존 자료에서의 유효 추론이 모두 필요해 A보다 부담이 크다.

## 4. 우선 실행할 결정 실험

아래 순서는 긴 baseline sweep 전에 연구 방향을 판별하기 위한 것이다. GPU 시간 추정이나 완료 약속은 아니다.

1. **센서의 정보량:** 같은 source family 안에서 생성된 probe와 실제 QA를 구분하고, compression 피해를 예측하는지 평가한다. 학습/선택에 쓰지 않은 QA로 AUROC·calibration·risk–coverage를 본다. evidence turn을 이용한 매칭은 진단용이며 배포 시 사용 가능한 feature로 취급하지 않는다.
2. **같은 예산에서 residual의 가치:** random, recency, salience, coverage, task-risk, oracle evidence를 비교한다. oracle은 선택에 gold가 들어간 진단 상한이며 실제 정책으로 main table에 섞지 않는다. 원문도 LLM reader에게 항상 최상의 성능을 주는 oracle은 아니다.
3. **원문 비가용 실험:** writer가 제한된 메모리를 직렬화한 뒤 reader 프로세스는 그 파일만 읽는다. 전체 history나 생성한 모든 facts가 우회 채널로 남아 있지 않아야 한다. capacity 초과와 fallback을 명시한다.
4. **진짜 재압축:** 앞선 압축 결과를 다음 압축 입력으로 사용한다. 새로운 raw history에서 매번 재생성하면 누적 손실을 검증한 것이 아니다.
5. **인증과 효용의 동시 확인:** 목표 위험을 맞추더라도 거의 항상 raw/defer/abstain이면 tradeoff가 없는 것이다. 같은 비용의 단순 baseline과 비교한다.

계속할 조건은 특정 점수 하나가 아니라, held-out source에서 잔여 정보 선택이 단순 baseline보다 안정적으로 낫고, 전체 비용을 세어도 이득이 있으며, 검증 신호가 실제 실패와 연결되는 것이다. 그렇지 않으면 A에서 B로 옮기거나 보장 주장을 축소한다. 기존 README의 Spearman 0.4 또는 AUROC 0.65 같은 내부 go/no-go 기준을 ICLR 채택 기준이나 이론 증명으로 취급하지 않는다.

## 5. 제출 전 해결해야 하는 구현·평가 간극

이 항목은 이번에 확인한 로컬 코드에 대한 관찰이다. 코드는 수정하지 않았다.

| 항목 | 로컬 근거 | 논문에 미치는 영향 |
|---|---|---|
| read-time `certified`는 아직 calibrated QA 상한이 아님 | [v15](../../methods/certmem/scripts/run_system_v15.py), `R0=sum(rel*r)`, 선택 후 해당 위험 차감 | session risk와 정규화 relevance를 조합한 점수는 확률이 아니다. 일부 문장만 확장해도 해당 turn 기여를 전부 차감한다. 미검색 후보·profile 오류도 포함하지 않는다. |
| `--no_raw`는 원문 삭제 실험과 다름 | [v15](../../methods/certmem/scripts/run_system_v15.py), raw_turns/ridx 생성 후 read loop에서 확장을 skip | 메모리에 원문이 남고, write defer가 raw를 units에 넣는다. 접근 차단·직렬화·저장량 검사를 별도로 해야 한다. |
| 전역 저장 한도가 없음 | [v15](../../methods/certmem/scripts/run_system_v15.py), 세션별 cap, units 누적 | cap은 세션별 요약 줄 수다. 전체 persistent memory가 고정된 조건의 장기 성능으로 표현할 수 없다. |
| 저장량 계산이 전체 footprint와 다름 | [v15](../../methods/certmem/scripts/run_system_v15.py), `mem_total=sum(u['tok'])` | raw archive·profiles·embeddings·index·certificate metadata를 별도 보고해야 한다. 최종 prompt 토큰도 직접 센다. |
| 사실 추출은 exhaustive oracle이 아님 | [facts.py](../../methods/certmem/certmem/facts.py), window별 최대 12개, 생성 LLM | 추출 누락 자체를 감사해야 한다. 빈 추출을 risk 0으로 처리하는 분기도 거짓 통과를 만들 수 있다. |
| calibration 표본 단위 | [ltt_gate.py](../../methods/certmem/scripts/ltt_gate.py), 대화별 fold지만 HB 입력은 session rows | 같은 대화의 session 의존성은 fold 분리만으로 제거되지 않는다. 독립 trajectory/source-family 집계 또는 정당화된 의존성 추론이 필요하다. |
| e-process 설명 불일치 | [ltt_gate.py](../../methods/certmem/scripts/ltt_gate.py), 함수 주석과 마지막 출력 | `1+lambda*(eps-r)`는 낮은 risk에서 증가한다. 마지막 출력의 높은-risk 귀무가설 반박 해석과, 함수 주석의 낮은-risk 귀무가설 아래 유지된다는 설명이 반대다. 귀무가설과 조건부 평균 가정을 일치시켜야 한다. |
| host 재현 범위 | [mem0_adapter.py](../../methods/certmem/certmem/hosts/mem0_adapter.py) | 현재 Mem0-like/LightMem-like는 prompt 재구현이다. 실제 upstream baseline 결과와 구분한다. |
| answerability 평가 범위 | [longmemeval.py](../../methods/certmem/certmem/longmemeval.py), `_abs` 제외 | abstention/신뢰도를 주장하려면 answerable 질문의 accuracy 외에 abstention 항목과 오답–응답률 tradeoff가 필요하다. |
| 주장과 수치 해석 | [analyze_sensors.py](../../methods/certmem/scripts/analyze_sensors.py), 진단 결과에 theorem confirmed 표현 | 관측 상관관계·AUROC는 정리의 증명이 아니다. bootstrap 역시 질문별 IID 대신 실제 source 의존성을 고려한다. |

e-process 설명 점검의 간단한 계산: risk가 전부 0이고 eps=0.3, lambda=0.5이면 20회 후 값은 `1.15^20 = 16.3665`다. 낮은 위험에서 임계값 10을 넘는 것은 현재 식에서 자연스럽다. 이것을 위험 위반 경보로 해석하면 방향이 뒤집힌다. 단순한 unconditional mean만으로 adaptive sequence의 supermartingale 조건이 성립하는 것도 아니다.

Calibration 데이터 규모도 조기에 확인해야 한다. 예를 들어 **고정된 하나의 정책, IID Bernoulli 실패 사건, 관측 실패 0건**이라는 단순 상황에서 one-sided exact 95% 상한을 5% 이하로 만들려면 최소 59개 독립 표본이 필요하다. 1% 이하에는 299개다: `ceil(log(0.05)/log(1-eps))`. 이는 LoCoMo QA 개수를 독립 표본으로 세라는 뜻이 아니며, bounded mean loss·다중 정책 선택에는 동일 공식을 그대로 적용할 수 없다. 대화가 10개인 데이터만으로 강한 trajectory-level 보장을 주장하기 어렵다는 규모 감각이다.

전체 비용은 적어도 `write + verify + repair + read` 토큰·호출·시간과 지속 저장 footprint로 나눠 보고한다. 질문 수 Q에 대해 `C_total(Q)=C_write+Q*C_read`를 비교하면 비싼 write 검증이 어느 사용량에서 이득이 되는지도 볼 수 있다. 사용한 학습·calibration·사람 감사 비용은 별도 공개한다.

## 6. 지금 주된 신규성으로 밀지 않을 것

- profile, query reformulation, top-k 확대, reasoning mode 추가: 성능 구성 요소와 ablation으로는 유용하지만 현재 검색된 선행 대비 중심 기여로 약하다.
- 3종 메모리 또는 summary/raw hierarchy: TriMem·TierMem과 직접 겹친다.
- verifier reward로 RL 추가: TrustMem과 겹치며 계산비용도 증가한다.
- rate-distortion 수식 하나: DeMem과 관련 compaction 연구가 있으므로 정식화만으로 차별화하기 어렵다.
- 무제한 raw 보관을 제외한 “저장 압축률”: 실제 조건을 흐린다.
- 합성 benchmark만 만들고 그 위에서만 이기는 결과: 외부 benchmark와 메커니즘 진단이 함께 필요하다.
- 여덟 방향을 모두 하나의 모델에 결합: 어떤 아이디어가 효과를 냈는지 흐려진다. A를 먼저 검증하고 부족한 원인에 따라 B 또는 C로 발전시키는 편이 낫다.

**최우선 제안:** CertMem을 “검증 기능이 많은 메모리”로 설명하기보다, **실제 답변 회귀를 통제하는 작은 잔여 근거 보존 방법**으로 정리한다. 원문 재접근 금지와 반복 압축은 이 원리가 필요한 이유를 보여주는 실험 조건으로 사용한다. 이 주장이 독립 데이터·정확한 비용 계산·검증기 감사 아래 유지되는지가 논문 방향을 결정해야 한다.
