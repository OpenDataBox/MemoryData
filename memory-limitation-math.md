# 메모리 프레임워크 개선을 위한 수학 공식·이론 42선

> **정리일:** 2026-09-03  
> **대상:** MemoryData / RetentionMem 및 외부 장기 메모리 프레임워크  
> **구성:** 이론 후보 42개 · 추천 연구 조합 3개 · 참고 자료 26개

[일반 텍스트 버전(TXT)](../memory_framework_math_theories_42_ko.txt)

앞선 대화와 TXT의 전체 내용을 읽기 쉬운 Markdown으로 정리했다. 수식은
인라인 수식 `$...$`과 독립 수식 `$$...$$`으로 작성했다.
수식을 지원하는 Markdown 미리보기에서 열어 읽으면 된다.

이 문서는 **이론 검토와 설계 아이디어**를 정리한 것이다. 새로운 실험 결과,
성능 보장, 연구의 신규성을 입증한 문서는 아니다. 기존 이론의 표준식과
메모리 문제에 맞게 옮긴 설계식이 함께 있으므로, 각 항목의 **적용 조건과
주의점**을 함께 읽어야 한다. 이론을 적용한다고 원래 정리의 보장이
자동으로 따라오는 것은 아니다.

**우선 검토** 표시는 현재 문제와 코드에 특히 직접적으로 연결되는
후보라는 뜻이다. 42개를 하나의 시스템에 모두 넣으라는 뜻은 아니다.

## 목차

- [0. 해결할 문제, 우선순위, 공통 기호](#section-0)
- [A. 정보이론과 강건 최적화 — 01~08](#section-a)
- [B. 통계적 검증 — 09~16](#section-b)
- [C. 논리와 시간 — 17~24](#section-c)
- [D. 조합 최적화와 그래프 — 25~32](#section-d)
- [E. 장기 의사결정 — 33~38](#section-e)
- [F. 평가와 인과적 비교 — 39~42](#section-f)
- [G. 추천 연구 조합 3개](#section-g)
- [H. 코드 연결 및 반드시 구분할 것](#section-h)
- [I. 참고문헌과 원문 링크](#section-i)

---

<a id="section-0"></a>

## 0. 해결할 문제, 우선순위, 공통 기호

### 0.1. 앞서 다룬 다섯 가지 메모리 문제

1. 압축·요약 중 나중에 필요한 세부사항이 사라진다.
2. 새 정보를 반영하면서 여전히 유효한 옛 정보까지 훼손할 수 있다.
3. 비슷한 기억을 찾는 것과 답에 필요한 근거를 찾는 것은 다르다.
4. 발언·계획·실행 결과가 기억 속에서 섞일 수 있다.
5. 실패 원인과 전체 비용을 분리해서 평가하기 어렵다.

### 0.2. 이론을 고르는 우선순위

무엇을 잃으면 안 되는가?

- 정보이론, 비대칭 손실함수

이 변경을 승인해도 되는가?

- 선택적 예측, 통계적 위험 제어

무엇을 추가 보존해야 하는가?

- 집합 덮개, 배낭 문제, 하이퍼그래프

어떤 사실이 언제 유효한가?

- 믿음 수정, 시간 데이터베이스, 출처 대수

언제 더 검색하거나 정리해야 하는가?

- 정보의 가치, 제한된 의사결정 과정

### 0.3. 공통 기호

| 기호 | 의미 |
| --- | --- |
| $H_t$ | 지금까지 들어온 원문 기록 |
| $M^-$ | 새 관측까지 반영했지만 아직 압축·병합하지 않은 메모리 |
| $M^+$ | 압축·병합한 후보 메모리 |
| $q,\ Q$ | 질문 또는 수행할 작업 |
| $a^*(H_t,q)$ | 원문과 질문의 시간 범위에 근거한 정답 |
| $g(M,q)$ | 메모리 $M$을 사용하는 모델의 답 |
| $\ell$ | 오답 또는 작업 실패의 손실 |
| $B$ | 메모리·토큰·계산 예산 |
| $\mathbb E[\cdot]$ | 기댓값 |
| $P(\cdot)$ | 확률 |
| $I(X;Y)$ | 상호정보량 |
| $\sum$ | 합계 |
| $\arg\min,\ \arg\max$ | 목적함수를 최소·최대화하는 선택 |
| $\mathbf 1[\cdot]$ | 조건이 참이면 1, 거짓이면 0 |
| $[x]_+$ | $\max(x,0)$ |
| $\epsilon$ | 허용할 위험 또는 손실의 상한 |
| $\delta$ | 통계적 보장이 실패하도록 허용하는 확률 |

각 항목에서 새로 쓰는 기호는 해당 항목에서 설명한다.

### 0.4. 변경 때문에 발생한 성능 퇴행

**변경 때문에 발생한 성능 퇴행**

$$
\Delta_q=\left[\ell\!\left(g(M^+,q),a^*\right)-\ell\!\left(g(M^-,q),a^*\right)\right]_+
$$

**뜻**

압축·갱신 이후의 손실이 이전보다 커진 만큼을 측정한다. 개선되었거나 변화가 없으면 퇴행은 0이다.

**중요한 평가 조건**

변경 전후 모두 같은 최신 원문과 같은 시간 범위의 정답으로 평가한다. 그래야 잘못된 옛 정보를 올바르게 고친 것을 기억 손실로 오판하지 않는다.

퇴행만 줄이면 기존 오답을 그대로 유지하는 정책도 좋아 보일 수 있다. 따라서 절대 정답률·작업 성공률과 변경에 의한 퇴행을 함께 봐야 한다.

### 0.5. 피할 수 없는 용량 한계

**B비트가 표현할 수 있는 서로 다른 상태 수**

$$
N_{\mathrm{states}}=2^B
$$

구별해야 하는 기록이 이보다 많다면, 외부 원문에 접근하지 않고 모든 미래 질문에 완벽하게 답하는 것은 불가능하다.

따라서 목표는 “무손실 만능 기억”이 아니라 다음과 같다.

정해진 예산에서 중요한 정보 손실을 최소화한다.

원문 전체를 별도 보관한다면 복구 가능성은 높아지지만, 그 원문 저장량과 접근 비용도 시스템 비용에 포함해야 한다.

<a id="section-a"></a>

## A. 무엇을 남기고 압축할 것인가: 정보이론과 강건 최적화

<a id="theory-01"></a>

### 01. Rate-distortion theory: 기억 용량과 정보 손실의 교환관계

> **우선 검토**

**핵심 개념**

허용 손실 D 아래에서 기록을 표현하는 데 필요한 최소 정보량을 생각하는 틀이다. 메모리의 왜곡을 문장 유사도가 아니라 미래 질문에 답하지 못하는 정도로 정의하는 것이 핵심이다.

**핵심 수식**

$$
R(D)=\min_{p(M\mid H):\,\mathbb{E}[d(H,M,Q)]\le D} I(H;M)
$$

**실용적인 설계식**

$$
\min_M \mathbb{E}_Q\!\left[\ell(g(M,Q),a^*)\right]+\lambda\,\operatorname{tokens}(M)
$$

**메모리에 적용**

“짧고 자연스러운 요약”이 아니라 “짧으면서 답변 능력을 유지하는 요약”을 선택한다. 압축률을 늘렸을 때 QA 성능이 얼마나 나빠지는지 곡선을 만들고, 예산에 맞는 지점을 찾을 수 있다.

**주의점**

정보량의 비트와 LLM 토큰 수는 같은 단위가 아니다. 상호정보량과 미래 질문 분포를 정확히 알기 어려워 실제 구현에서는 QA 손실, 근거 보존율 등의 대리 목적함수를 사용하게 된다.

참고: [R01](#ref-r01)

<a id="theory-02"></a>

### 02. Information bottleneck: 답에 필요한 정보를 보존

> **우선 검토**

**핵심 개념**

원문을 적게 담더라도 답에 필요한 정보는 많이 담도록 한다.

**메모리용 설계식**

$$
\max_M I(M;A\mid Q)-\beta I(M;H)
$$

**기호**

- I(M;A | Q): 질문을 알 때 메모리가 정답에 제공하는 정보
- I(M;H)   : 메모리가 원문에서 담고 있는 정보
- $\beta$     : 압축 강도를 조절하는 가중치

**메모리에 적용**

반복적인 인사말은 줄이고 부정·예외·날짜·실행 전제조건은 남기는 방향으로 압축기를 학습한다.

**주의점**

위 식은 원래 information bottleneck을 질의 조건부 문제로 옮긴 설계식이다. 상호정보량의 정확한 추정은 어렵다. 임베딩 코사인 유사도를 상호정보량이나 사실 보존율과 동일시하면 안 된다.

참고: [R02](#ref-r02)

<a id="theory-03"></a>

### 03. Sufficient statistic: 메모리를 과거 기록의 충분통계량으로 보기

**핵심 개념**

메모리를 알고 나면 원문을 더 봐도 정답 예측에 추가 도움이 없는 표현을 이상적인 목표로 삼는다.

**핵심 수식**

$$
P(A\mid H,Q)=P(A\mid M,Q)
$$

**메모리에 적용**

작업형 에이전트에서는 긴 대화 전체 대신 아래 상태가 충분할 수 있다.

- 현재 목표
- 완료된 작업과 검증 결과
- 남은 의존성
- 유효한 제약조건

**주의점**

충분성은 어떤 질문·작업 집합을 대상으로 하는지에 달려 있다. 예약 업무에 충분한 상태가 과거 대화 전체의 회상에도 충분한 것은 아니다. 실용 시스템에서는 정확한 충분성보다 근사적인 충분성을 목표로 삼는다.

<a id="theory-04"></a>

### 04. Minimum description length: 요약과 잔여 정보의 합 최소화

**핵심 개념**

공통 패턴은 요약으로 저장하고, 요약으로 설명되지 않는 예외는 잔여 정보로 저장한다.

**핵심 수식**

$$
\min_C L(C)+L(H\mid C)
$$

**기호**

- C       : 요약 또는 규칙
- L(C)    : 요약 자체의 부호 길이
- L(H | C): 요약만으로 설명되지 않는 잔여 정보의 부호 길이

**메모리에 적용**

요약: “매주 화요일 회의.” 잔여 정보: “매월 첫째 주 제외.”

RetentionMem의 “요약 + residual” 구조와 잘 맞는다.

**주의점**

일부 원문 구절만 남기는 방식은 원문 전체를 복원하는 엄밀한 무손실 MDL 부호화와 구분해야 한다.

참고: [R03](#ref-r03)

<a id="theory-05"></a>

### 05. Successive refinement: 필요에 따라 기억의 해상도 높이기

**핵심 개념**

처음에는 저해상도 요약을 사용하고, 필요할 때 세부 정보를 추가한다.

**설계식**

$$
M_{k+1}=(M_k,E_{k+1})
$$

**기호**

- E_(k+1): 다음 단계에서 추가하는 정보

**메모리에 적용**

1단계: 인물·사건 요약 2단계: 날짜와 예외 3단계: 정확한 원문 4단계: 도구 실행 기록

검색 비용을 단계적으로 지출하고 필요한 해상도에서 멈춘다.

**주의점**

모든 데이터와 손실함수에서 단계별 표현이 각각 최적이 되는 것은 아니다. 계층을 만들었다고 정보이론적으로 최적인 것은 아니다.

참고: [R01](#ref-r01)

<a id="theory-06"></a>

### 06. Cost-sensitive learning: 정보 손실의 종류마다 다른 비용 부여

> **우선 검토**

**핵심 개념**

날짜 누락, 부정 누락, 인물 혼동, 조건 누락, 사실 창작을 같은 심각도의 오류로 취급하지 않는다.

**설계식**

$$
\begin{aligned}
d={}&w_{\mathrm{date}}d_{\mathrm{date}}
+w_{\mathrm{neg}}d_{\mathrm{neg}}
+w_{\mathrm{entity}}d_{\mathrm{entity}}\\
&+w_{\mathrm{condition}}d_{\mathrm{condition}}
+w_{\mathrm{fabrication}}d_{\mathrm{fabrication}}
\end{aligned}
$$

**메모리에 적용**

“좋아한다”와 “좋아하지 않는다”는 문자열 대부분이 같아도 의미 손실은 크다. 반대로 표현만 바뀌었다면 손실이 아닐 수 있다.

숫자·부정·시간 범위 특징을 손실함수와 학습 가중치에 명시적으로 연결한다. 비교적 낮은 구현 비용으로 시작할 수 있다.

**주의점**

가중치를 임의로 정하고 결과에 맞춰 바꾸면 평가가 편향될 수 있다. 가중치의 근거와 민감도 분석을 함께 제시하는 것이 좋다. [0,1] 손실을 요구하는 통계 절차와 결합할 때는 범위도 맞춰야 한다.

<a id="theory-07"></a>

### 07. CVaR: 평균 성능보다 심각한 실패를 줄이기

**핵심 개념**

평균 손실뿐 아니라 손실이 큰 쪽의 꼬리 부분을 줄인다.

**핵심 수식**

$$
\operatorname{CVaR}_{\alpha}(L)=\min_{\eta}\left[\eta+\frac{1}{1-\alpha}\mathbb{E}\!\left[(L-\eta)_+\right]\right]
$$

**메모리용 목적함수**

$$
\min_\pi\mathbb{E}[L_\pi]+\lambda\,\operatorname{CVaR}_{0.95}(L_\pi)
$$

**메모리에 적용**

$\alpha$ = 0.95이면 대략 최악의 5%에 해당하는 큰 손실에 집중한다. 평균 QA 정확도는 높지만 특정 날짜나 실행 조건을 심하게 망가뜨리는 시스템을 구분하는 데 유용하다.

**주의점**

드문 실패를 추정하려면 충분한 표본이 필요하다. 0/1 오류뿐 아니라 실패의 심각도를 반영한 손실 정의도 검토할 수 있다.

참고: [R04](#ref-r04)

<a id="theory-08"></a>

### 08. Distributionally robust optimization: 다른 미래 질문에 대비

**핵심 개념**

학습 때 예상한 질문 분포 하나에만 맞추지 않고, 허용한 분포 변화 안에서 가장 불리한 경우까지 고려한다.

**핵심 수식**

$$
\min_\pi\sup_{P\in\mathcal{U}(P_0)}
\mathbb{E}_{Q\sim P}[L_\pi(Q)]+\lambda C(\pi)
$$

**간단한 설계식**

$$
\min_\pi\max_{k\in\mathcal{K}}\mathbb{E}[L_\pi\mid \mathrm{type}=k]
$$

**기호**

- P0   : 학습 때 예상한 질문 분포
- U(P0): 그 주변에서 허용하는 분포 변화의 집합

**메모리에 적용**

일상 선호 질문으로 학습했더라도 날짜·수량·예외 질문 비중이 높아졌을 때 무너지지 않도록 한다.

**주의점**

지정한 불확실성 집합 밖의 임의 변화까지 보장하지는 않는다. 어떤 분포 변화가 현실적인지 먼저 정의해야 한다.

참고: [R05](#ref-r05)

<a id="section-b"></a>

## B. 압축·갱신을 승인해도 되는가: 통계적 검증

<a id="theory-09"></a>

### 09. Selective prediction: 위험하면 변경하지 않을 권리

> **우선 검토**

**핵심 개념**

위험 점수가 낮은 변경만 승인하고, 위험하면 원래 메모리를 유지한다.

**핵심 수식**

$$
\begin{aligned}
\kappa(\tau)&=P(s(X)\le\tau),\\
R(\tau)&=\mathbb{E}[\Delta\mid s(X)\le\tau].
\end{aligned}
$$

**선택 목표**

$$
\max_\tau\kappa(\tau)\qquad\text{subject to}\quad R(\tau)\le\epsilon
$$

**기호**

- s(X): 변경 후보 X의 위험 점수
- $\tau$ : 승인 임계값

**메모리에 적용**

허용 위험 안에서 가능한 한 많은 메모리를 압축한다. RetentionMem의 승인/유지 구조와 직접 연결된다. 단순 분류 정확도보다 “얼마나 압축하면서 얼마나 망가뜨리는가”를 평가하기에 적합하다.

**주의점**

MLP의 sigmoid 출력이 그 자체로 정확한 실패확률은 아니다. 승인된 사례의 조건부 위험을 별도로 보정·평가해야 한다.

<a id="theory-10"></a>

### 10. 이항분포 신뢰상한: 관측 오류 0개와 실제 위험 0은 다름

> **우선 검토**

**핵심 개념**

미리 고정한 정책을 독립적인 평가 사례에 적용하고, 관측한 실패 수로 실제 실패확률의 상한을 추정한다.

**단측 Clopper–Pearson 신뢰상한**

$$
U=
\begin{cases}
F^{-1}_{\operatorname{Beta}(k+1,n-k)}(1-\delta),&k<n,\\
1,&k=n.
\end{cases}
$$

**실패가 0개일 때**

$$
U=1-\delta^{1/n}
$$

**기호**

- n: 독립 평가 사례 수
- k: 퇴행 또는 실패 사례 수
- 1 - $\delta$: 신뢰수준

**95% 신뢰수준의 예**

20개 중 실패 0개 -> 위험 상한 약 13.9% 60개 중 실패 0개 -> 위험 상한 약 4.87%

**메모리에 적용**

“검증셋에서 실패가 없었다”는 결과를 과신하지 않도록 한다. 대화 당시 확인한 RetentionMem calibration의 최소 승인 표본 20개만으로 위험 5% 이하를 보장할 수는 없다. 해당 구현도 통계적 보장이 아니라고 명시하고 있었다.

**주의점**

같은 대화에서 만든 probe 60개는 독립 사례 60개와 같지 않다. 여러 임계값을 살펴본 뒤 좋은 결과만 고르면 추가 보정이 필요하다. 이 식의 실패 사건을 명확히 정의해야 한다. 예: Delta > 0.

**코드 연결**

`D:/MemoryData/methods/retentionmem/policy.py`
함수: calibrate

<a id="theory-11"></a>

### 11. Learn then Test: 여러 임계값을 골라보는 효과까지 통제

> **우선 검토**

**핵심 개념**

정책·임계값 후보를 여러 개 탐색했다는 사실까지 반영해 위험을 검정한다.

**핵심 수식**

$$
\Lambda=\{\lambda_1,\ldots,\lambda_J\},
\qquad H_{0,j}:R(\lambda_j)>\epsilon
$$

**Bonferroni 방식의 통과 기준**

$$
p_j\le\frac{\delta}{J}
$$

**메모리에 적용**

아래 후보 정책들 중 위험 기준을 통과한 것을 선택한다.

- 요약만 승인
- 위험한 구절을 추가 보존한 뒤 승인
- 더 보수적인 임계값으로 승인
- 변경하지 않고 유지

**주의점**

후보를 만드는 데이터와 최종 검정 데이터를 분리해야 한다. 승인 사례의 조건부 위험을 다룰 때는 그에 맞는 검정을 구성해야 한다. 단순히 검증셋에서 가장 좋아 보이는 임계값을 선택하는 것과 다르다.

참고: [R06](#ref-r06)

<a id="theory-12"></a>

### 12. Conformal risk control: 보존량을 위험 기준으로 보정

**핵심 개념**

손실 범위와 단조성 등의 조건이 맞을 때, 보존 범위의 보수성을 검증 데이터로 보정해 기대 위험을 제어한다.

**핵심 수식 — 손실이 [0,1]인 경우**

$$
\hat{\lambda}=
\inf\left\{\lambda:
\frac{n\widehat{R}_n(\lambda)+1}{n+1}\le\epsilon\right\}
$$

**메모리에 적용**

$\lambda$를 추가로 보존하는 근거 범위로 볼 수 있다. 보존 범위를 얼마나 넓혀야 누락 손실이 허용 범위에 들어오는지 정한다.

**주의점**

- 원래 보장은 기대 위험에 관한 것이다.
- 교환가능성 등 통계적 전제가 필요하다.
- 기본 절차는 보수성이 커질수록 사례별 손실이 감소하는 조건을 쓴다.
- 문맥을 더 넣으면 LLM 답이 나빠질 수 있어 실제 QA 손실은 단조적이지

않을 수 있다.

“원문을 더 보존하니까 conformal 보장이 성립한다”고 주장하면 안 된다. 현재 문제에서는 [11](#theory-11) Learn then Test를 먼저 검토하는 편이 다루기 쉬울 수 있다.

참고: [R07](#ref-r07)

<a id="theory-13"></a>

### 13. Union bound: 반복 갱신의 누적 위험 계산

> **우선 검토**

**핵심 개념**

한 번의 갱신 위험이 작아도 수천 번 반복하면 한 번 이상 실패할 위험은 커질 수 있다.

**핵심 수식**

$$
P\!\left(\bigcup_{t=1}^{T}E_t\right)\le\sum_{t=1}^{T}P(E_t)
$$

**위험 예산 배분**

$$
\sum_{t=1}^{T}\epsilon_t\le\epsilon_{\mathrm{total}}
$$

**기호**

- $E_t$: 갱신 t에서 잘못된 변경이 발생하는 사건

**메모리에 적용**

시스템 전체 생애 동안 허용할 위험을 정하고 각 갱신에 나누어 배분한다. 단발성 성능뿐 아니라 장기 실행 중 누적 위험을 평가한다.

**주의점**

$\epsilon_t$는 해당 실행 조건에서 정당화된 확률 상한이어야 한다. 검증셋 평균 오류율을 단계별 상한으로 그대로 대입하면 안 된다. 갱신 횟수가 매우 많으면 이 방식이 지나치게 보수적일 수 있다.

<a id="theory-14"></a>

### 14. Confidence sequences: 계속 검사해도 유효한 신뢰구간

**핵심 개념**

데이터를 계속 추가하면서 결과를 반복 확인해도 유효하도록 설계된 신뢰구간을 사용한다.

**핵심 수식**

$$
P\!\left(\forall t\ge1:\mu\in[L_t,U_t]\right)\ge1-\delta
$$

**기호**

- $\mu$       : 안정된 모집단의 위험
- $L_t$, $U_t$ : 시점 t의 하한과 상한

**메모리에 적용**

운영 중 성능을 검사하다 충분히 안전해 보이면 정책을 채택하거나, 위험해지면 중단하는 판단에 활용한다.

**주의점**

일반 신뢰구간을 반복 계산하면서 좋은 순간에만 멈추는 것과 다르다. 분포가 계속 변하면 고정된 $\mu$라는 목표를 바꾸거나 변화 탐지와 결합해야 한다. 필요한 확률적 가정을 확인해야 한다.

참고: [R08](#ref-r08)

<a id="theory-15"></a>

### 15. Counterexample-guided refinement: 실패 질문으로 기억 수리

> **우선 검토**

**핵심 개념**

현재 압축 후보를 곤란하게 만드는 진단 질문을 찾고, 그 질문에 필요한 근거를 추가해 후보를 정교화한다.

**설계식**

$$
q^*\approx\arg\max_{q\in\mathcal{Q}_{\mathrm{probe}}}\Delta_q(M^-,M^+)
$$

**반례를 이용한 보완**

$$
M^+\leftarrow\operatorname{concat}\!\left(M^+,\operatorname{evidence}(q^*)\right)
$$

**메모리에 적용**

요약 -> 실패 probe -> 필요한 원문만 보존 -> 다시 검사

CEGAR의 “추상화 -> 반례 -> 정교화”에서 착안한 구조다.

**주의점**

LLM이 만든 질문과 답은 형식 검증기의 완전한 반례 탐색이 아니다. 원래 CEGAR의 증명 보장을 그대로 가져올 수 없다. 진단 질문은 과거 원문에서 만들고 실제 미래 평가 질문은 사용하지 않는다.

참고: [R09](#ref-r09)

<a id="theory-16"></a>

### 16. Metamorphic testing: 정답을 전부 몰라도 지켜야 할 관계 검사

**핵심 개념**

의미를 보존하는 변환을 입력에 적용했을 때 출력도 대응 관계를 유지해야 한다는 성질을 검사한다.

**설계식**

$$
g\!\left(C(T(H)),T_Q(q)\right)\approx T_A\!\left(g(C(H),q)\right)
$$

**기호**

- C  : 메모리 압축 과정
- T  : 원문에 적용하는 의미 보존 변환
- T_Q: 질문의 대응 변환
- T_A: 정답의 대응 변환

**검사 예**

- 이름을 일관되게 바꾸면 답의 이름도 함께 바뀌어야 한다.
- 무관한 문장을 추가해도 관련 답은 유지되어야 한다.
- 같은 사건을 중복 입력해도 독립적인 근거 수가 늘어나면 안 된다.
- 같은 뜻의 표현으로 바꿔도 유효한 사실은 유지되어야 한다.

**주의점**

양쪽 답이 같은 방식으로 틀릴 수도 있다. 따라서 관계 일관성 검사만으로 정답성을 보장할 수는 없다. 시간이나 인과관계를 실제로 바꾸는 변환을 의미 보존 변환으로 잘못 취급하지 않도록 해야 한다.

<a id="section-c"></a>

## C. 무엇을 고치고 보존할 것인가: 논리와 시간

<a id="theory-17"></a>

### 17. AGM belief revision / 최소 변경: 필요한 부분만 믿음을 수정

> **우선 검토**

**핵심 개념**

새로 검증된 사실은 반영하되, 기존의 유효한 사실은 최대한 적게 버린다.

**유한한 사실 집합을 위한 설계식**

$$
\min_{K'}\sum_{f\in K}w_f\,\mathbf{1}[K'\not\models f]
$$

**제약조건 — 새로 검증된 사실 반영**

$$
K'\models\phi_{\mathrm{new}},
\qquad K'\text{ is consistent}
$$

**기호**

- entails: 논리적으로 해당 사실을 뒷받침한다는 뜻
- $w_f$    : 기존 사실 f를 버리는 비용

**메모리에 적용**

구조화된 사실·규칙으로 범위를 한정하면 weighted MaxSAT 또는 SMT 형태의 최소 변경 문제로 구현할 수 있다. 관련 없는 사실까지 한꺼번에 다시 쓰는 갱신을 피하는 데 유용하다.

**주의점**

AGM의 새 정보를 받아들이는 원칙을 그대로 쓰려면 그 정보를 신뢰해도 되는지 먼저 판단해야 한다. 과거 사실과 현재 사실은 시간 범위를 붙여야 불필요한 모순을 피할 수 있다.

참고: [R10](#ref-r10)

<a id="theory-18"></a>

### 18. Bitemporal model: 사실의 시간과 알게 된 시간을 분리

> **우선 검토**

**핵심 개념**

사실이 실제로 유효한 시간과, 시스템이 그 사실을 알고 있던 시간을 따로 기록한다.

**데이터 표현**

$$
f=\left(s,p,o,\,
[v_{\mathrm{from}},v_{\mathrm{to}}),\,
[t_{\mathrm{from}},t_{\mathrm{to}})\right)
$$

**기호**

- valid time      : 실제로 사실이 유효한 시간
- transaction time: 시스템이 그 사실을 기록·인식하고 있던 시간
- [a,b)           : a는 포함하고 b는 포함하지 않는 시간 구간

**메모리에 적용**

9월 3일에 “8월 20일 부산으로 이사했다”고 알게 되었다고 하자. 이사한 날짜와 시스템이 알게 된 날짜를 분리한다.

다음 질문을 다르게 답할 수 있다.

- 8월 25일 실제 거주지는 어디였는가?
- 8월 25일 당시 에이전트가 알고 있던 거주지는 어디였는가?

**주의점**

최근에 입력된 문장이 항상 가장 최근 사건은 아니다. 날짜가 모호하거나 알려지지 않았다면 임의의 정확한 시점을 만들지 않는다.

참고: [R11](#ref-r11)

<a id="theory-19"></a>

### 19. Provenance semiring: 결론의 근거 조합을 대수로 표현

> **우선 검토**

**핵심 개념**

결론이 어떤 근거들의 조합에 의존하는지를 명시적으로 기록한다.

**핵심 예제**

$$
\operatorname{prov}(f)=x_1x_2+x_3
$$

**읽는 방법**

x1 * x2: 근거 1과 근거 2가 함께 있어야 결론 f를 지원한다. + x3   : 근거 3만으로도 f를 별도로 지원할 수 있다.

**메모리에 적용**

근거 1이 무효화되면 x1 = 0을 대입해 남은 지지 경로를 확인한다. 어떤 원문이 삭제·수정되었을 때 어떤 요약과 결론을 재검토해야 하는지 추적할 수 있다.

**주의점**

이 다항식은 곧바로 확률식이 아니다. 데이터베이스 질의의 출처 추적을 자연어 추론에 적용하려면 추론의 의존관계를 제대로 추출해야 한다는 별도 문제가 남는다.

참고: [R12](#ref-r12)

<a id="theory-20"></a>

### 20. Truth maintenance system: 근거 변경을 관련 결론에만 전파

**핵심 개념**

결론을 지지하는 정당화 경로를 관리하고, 근거가 바뀌면 영향을 받는 부분만 다시 계산한다.

**핵심 예제**

$$
\mathcal{J}(c)=\{\{a,b\},\{d\}\},
\qquad\operatorname{supported}(c)=(a\land b)\lor d
$$

**메모리에 적용**

a가 무효화되어도 d가 살아 있다면 c를 즉시 삭제하면 안 된다. 메모리 전체를 다시 요약하는 대신 변경된 사실에 의존하는 부분만 재검토한다.

**다른 항목과의 관계**

[19](#theory-19)가 근거의 표현에 가깝다면, [20](#theory-20)은 그 표현을 이용하는 동적 갱신 절차에 가깝다.

**주의점**

근거 그래프 자체가 잘못되었으면 잘못된 결과를 전파할 수 있다. 의존관계 추출과 유효성 판단의 정확성을 함께 평가해야 한다.

참고: [R13](#ref-r13)

<a id="theory-21"></a>

### 21. Paraconsistent logic: 모순을 별도 상태로 보존

**핵심 개념**

서로 충돌하는 근거가 있다고 해서 아무 결론이나 허용하지 않는다. 미확인과 모순을 구분한다.

**핵심 표현**

$$
v(f)=\left(\operatorname{support}(f),\operatorname{support}(\neg f)\right)
$$

**네 가지 상태**

- (0,0): 미확인
- (1,0): 긍정 근거 존재
- (0,1): 부정 근거 존재
- (1,1): 긍정·부정 근거가 함께 존재하는 충돌 상태

**메모리에 적용**

충돌한 정보를 억지로 한 문장으로 합치지 않고 보존한 뒤, 추가 확인이나 시간·출처 분석으로 해소한다.

**주의점**

모순을 다룰 수 있게 해 주지만 어느 주장이 맞는지 자동으로 결정해 주지는 않는다. 위의 0과 1은 근거 존재 여부이지 확률이 아니다.

<a id="theory-22"></a>

### 22. Bayesian evidence aggregation: 출처 신뢰도를 반영한 근거 누적

**핵심 개념**

사용자 발언, 도구 결과, 외부 문서, 에이전트 추측을 같은 신뢰도로 취급하지 않고 근거에 따라 믿음을 갱신한다.

**핵심 수식 — 조건부 독립 증거 가정**

$$
\log\frac{P(f\mid e_{1:n})}{P(\neg f\mid e_{1:n})}
=
\log\frac{P(f)}{P(\neg f)}
+
\sum_{i=1}^{n}\log\frac{P(e_i\mid f)}{P(e_i\mid\neg f)}
$$

**메모리에 적용**

검증된 도구 결과와 에이전트의 추측에 다른 증거 가중치를 준다. 출처의 신뢰도와 관측 종류를 기억에 함께 기록한다.

**주의점**

상관된 근거를 중복 계산하면 확신을 과대평가한다. 같은 원문을 복사한 문서 열 개나 에이전트가 자기 답을 반복한 발언 열 개는 독립 증거 열 개가 아니다. 조건부 독립 가정이 맞지 않으면 의존성을 모델링해야 한다.

<a id="theory-23"></a>

### 23. Hoare logic + temporal invariants: 계획과 완료를 구분

> **우선 검토**

**핵심 개념**

행동의 전제조건과 완료 후 검증조건을 명시하고, 완료 기록이 실제 성공 증거에 연결되도록 한다.

**핵심 수식**

$$
\{P\}\;a\;\{Q\}
$$

**기호**

- P     : 실행 전제조건
- action: 행동
- Q     : 완료 후 확인해야 할 상태

**시간 불변조건의 예**

$$
\mathbf{G}\!\left(
\operatorname{emitCompleted}(j)\Rightarrow\operatorname{verifiedSuccess}(j)
\right)
$$

**메모리에 적용**

“완료”라고 기록하려면 해당 작업과 버전의 성공 증거가 있어야 한다. “저장하겠다”, “저장했다고 말했다”, “도구가 저장 성공을 반환했다”를 구분하는 작업 메모리에 적합하다.

**주의점**

자연어로 전후조건을 적어 놓은 것과 실제 검증기로 그 조건을 확인하는 것은 다르다. 예전 작업의 성공 기록이 현재 작업의 성공으로 잘못 연결되지 않도록 작업 ID와 버전도 확인해야 한다.

참고: [R14](#ref-r14)

<a id="theory-24"></a>

### 24. CRDT: 여러 에이전트의 기억 병합에서 순서 의존성 줄이기

**핵심 개념**

중복 기록과 서로 다른 동기화 순서가 있어도 복제 상태가 수렴하도록 병합 연산을 설계한다.

**병합 연산의 교환법칙·결합법칙·멱등성**

$$
\begin{aligned}
a\sqcup b&=b\sqcup a &&\text{(commutative)},\\
(a\sqcup b)\sqcup c&=a\sqcup(b\sqcup c) &&\text{(associative)},\\
a\sqcup a&=a &&\text{(idempotent)}.
\end{aligned}
$$

**메모리에 적용**

여러 에이전트가 같은 사건을 중복 기록하거나 다른 순서로 기록을 받아도 결과가 수렴하도록 한다. 불변 이벤트, 출처, 버전 같은 구조화된 층부터 적용하기에 적합하다.

**주의점**

복제 상태가 수렴하는 것과 내용이 사실인 것은 별개다. 자연어 요약 병합 자체가 이 성질을 만족한다고 가정하면 안 된다.

참고: [R15](#ref-r15)

<a id="section-d"></a>

## D. 무엇을 보존하고 검색할 것인가: 조합 최적화와 그래프

<a id="theory-25"></a>

### 25. Weighted set cover / knapsack: 최소 비용으로 필요한 근거 보존

> **우선 검토**

**핵심 개념**

필요한 정보는 확보하면서 보존 비용은 작게 만드는 조합을 선택한다.

**기호**

- $x_i$ in {0,1}: 원문 구절 i를 보존할지 여부
- $c_i$         : 구절 i의 보존 비용
- $a_{ji}$        : 구절 i가 요구사항 j를 지원하는지 여부
- v_i         : 구절 i를 보존하는 가치

**가중 집합 덮개**

$$
\begin{aligned}
\min_{x\in\{0,1\}^n}\quad&\sum_i c_i x_i\\
\text{subject to}\quad&\sum_i a_{ji}x_i\ge1
\quad\forall j\in\mathcal{J}_{\mathrm{mandatory}}.
\end{aligned}
$$

**고정 예산의 배낭 문제**

$$
\max_{x\in\{0,1\}^n}\sum_i v_i x_i
\qquad\text{subject to}\quad\sum_i c_i x_i\le B
$$

**메모리에 적용**

어떤 residual을 남길지 직접 결정한다. 모든 위험 구절을 저장하는 대신, 여러 요구사항을 함께 지원하는 구절의 조합을 찾을 수 있다.

**주의점**

여러 구절이 함께 있어야 답할 수 있다면 단순 개별 커버리지로는 부족하다. [27](#theory-27) 하이퍼그래프 모델로 확장해야 한다. 구절 중복과 연결 구분자 때문에 실제 토큰 수는 최종적으로 확인한다.

<a id="theory-26"></a>

### 26. Submodular maximization: 중복을 줄이면서 보존 범위 늘리기

> **우선 검토**

**핵심 개념**

이미 확보한 정보를 또 가져오면 추가 이득이 작아지는 구조를 이용한다.

**예시 목적함수**

$$
F(S)=\sum_j w_j\min\left(1,\sum_{i\in S}a_{ji}\right)
$$

**한계효용 감소**

$$
\Delta(i\mid A)\ge\Delta(i\mid B)\qquad(A\subseteq B)
$$

**메모리에 적용**

비슷한 구절만 반복해서 보존하지 않고 서로 다른 필요 정보를 커버하는 구절들을 선택한다.

**이론적 결과**

정규화된 단조 submodular 함수와 개수 제한에서는 greedy가 $1-1/e$의 근사보장을 갖는다는 고전 결과가 있다.

**주의점**

토큰 비용이 다른 knapsack 제약에서 단순 gain/cost greedy에 같은 보장을 그대로 붙이면 안 된다. 문제와 알고리즘에 맞는 정리가 필요하다. 여러 근거의 결합 효과가 크면 submodular 가정 자체가 깨질 수 있다.

참고: [R16](#ref-r16)

<a id="theory-27"></a>

### 27. Hypergraph optimization: 여러 근거가 함께 있어야 생기는 가치

> **우선 검토**

**핵심 개념**

단독으로는 가치가 작아도 함께 있으면 답을 완성하는 근거 묶음을 선택의 기본 단위로 본다.

**설계식**

$$
F(S)=\sum_j w_j\,\mathbf{1}[E_j\subseteq S]
$$

**기호**

- $E_j$: 질문 j에 필요한 근거 집합
- S  : 선택한 근거 집합

**정수계획의 제약 예**

$$
y_j\le x_i\quad\forall i\in E_j,
\qquad x_i,y_j\in\{0,1\}
$$

예산 안에서 $\sum_j w_jy_j$를 최대화한다.

**메모리에 적용**

A가 누구인지 알려주는 구절과 A가 참석한 날짜를 알려주는 구절이 둘 다 있어야 답할 수 있는 경우를 다룬다. 개별 문장 중요도보다 “완성된 근거 묶음”을 선택한다.

**주의점**

이런 결합 가치는 한계효용 감소 가정을 깨뜨릴 수 있다. 필요한 근거 집합을 얼마나 정확하게 추정하는지가 핵심이다. 실제 미래 평가 질문을 근거 묶음 구성에 사용하면 평가 누수가 발생한다.

**평가**

멀티홉 질문의 근거 보존에 특히 검토할 가치가 높은 후보다.

<a id="theory-28"></a>

### 28. Personalized PageRank: 연결 관계를 따라 간접 근거 검색

**핵심 개념**

질문과 직접 비슷한 노드에서 시작해 관계 그래프를 따라 관련성을 확산시켜 간접 연결 정보를 찾는다.

**핵심 수식**

$$
\mathbf p=(1-\alpha)\mathbf v+\alpha P^\top\mathbf p
$$

**기호**

- v: 질문과 직접 관련된 시작 노드의 분포
- P: 관계 그래프의 전이행렬
- p: 관계를 따라 확산한 관련성 점수
- 0 < $\alpha$ < 1

**메모리에 적용**

벡터 유사도만으로 찾기 어려운 간접 근거를 검색한다. 시간·유효성·출처 제약을 함께 넣는 방향을 검토할 수 있다.

**주의점**

HippoRAG가 이미 사용하는 핵심 요소이므로 이것 자체가 새로운 연구 기여는 아니다. 잘못된 그래프 연결은 잘못된 검색을 확산시킨다.

참고: [R17](#ref-r17)

<a id="theory-29"></a>

### 29. MMR: 관련성은 높이고 검색 결과의 중복은 줄이기

**핵심 개념**

질문과의 관련성을 높이는 동시에 이미 선택한 결과와의 중복을 줄인다.

**핵심 수식**

$$
i^*=\arg\max_{i\notin S}\left[
\lambda\,\operatorname{sim}(q,i)
-(1-\lambda)\max_{j\in S}\operatorname{sim}(i,j)
\right]
$$

**메모리에 적용**

고정 top-k가 같은 내용을 여러 번 가져오는 문제를 줄인다. 추가 학습 없이 적용하기 쉬워 저비용의 강한 비교 기준으로 적합하다.

**주의점**

다양성이 높다고 답에 필요한 근거가 완성되는 것은 아니다. 관련성과 중복 점수의 척도 및 가중치를 점검해야 한다.

참고: [R18](#ref-r18)

<a id="theory-30"></a>

### 30. Determinantal point process: 기억 집합 전체의 다양성 모델링

**핵심 개념**

서로 거의 같은 기억을 함께 고르기 어렵게 만들어, 품질과 다양성을 함께 고려하는 부분집합 선택 모델이다.

**핵심 수식**

$$
P(S)=\frac{\det(L_S)}{\det(I+L)},\qquad L\succeq0
$$

**기호**

- L  : 양의 준정부호 커널 행렬
- L_S: 선택한 집합 S에 해당하는 부분행렬
- det: 행렬식
- I  : 단위행렬

**메모리에 적용**

중복이 적은 검색 결과 집합을 구성한다. 서로 다른 유형을 검사하는 진단 probe 집합을 선택하는 데도 쓸 수 있다.

**주의점**

다양성 목적이 중요한 근거의 동시 포함과 충돌할 수 있다. [27](#theory-27)의 근거 완성도와는 목적을 구분해야 한다.

참고: [R19](#ref-r19)

<a id="theory-31"></a>

### 31. Contrastive learning / InfoNCE: 주제가 아니라 근거를 구별

**핵심 개념**

정답 근거는 가깝게, 혼동하기 쉬운 근거는 멀게 학습한다.

**핵심 수식**

$$
\mathcal L_{\mathrm{NCE}}
=-\log\frac{\exp(s(q,e^+)/T)}
{\displaystyle\sum_{e\in\{e^+\}\cup\mathcal N}\exp(s(q,e)/T)}
$$

**기호**

- e_positive: 정답을 뒷받침하는 근거
- T         : temperature, 점수 분포의 날카로움 조절

**좋은 hard negative의 예**

- 같은 사람에 관한 다른 날짜의 사실
- 현재 값과 충돌하는 옛 값
- 실행 예정과 실행 완료
- 수량이나 부정만 다른 문장

**메모리에 적용**

일반적인 주제 유사도보다 시간과 상태에 민감한 검색기를 학습한다.

**주의점**

학습 데이터 구축 비용이 비교적 크다. 어떤 시간 범위에서는 맞는 근거를 무조건 오답 근거로 취급하면 안 된다. 학습용 근거와 실제 평가용 질문·정답을 분리해야 한다.

참고: [R20](#ref-r20)

<a id="theory-32"></a>

### 32. Value of information: 추가 검색이 비용을 낼 가치가 있는가

> **우선 검토**

**핵심 개념**

추가 검색으로 최종 의사결정이 좋아질 기대 이득과 검색 비용을 비교한다.

**핵심 수식**

$$
\operatorname{VOI}
=\mathbb E_Z\!\left[\max_a\mathbb E[U(a)\mid I,Z]\right]
-\max_a\mathbb E[U(a)\mid I]
-C_{\mathrm{search}}
$$

**기호**

- Z: 추가 검색 결과
- a: 최종 답변 또는 행동

**메모리에 적용**

VOI가 양수면 더 검색하고 음수면 멈추는 정책을 고려한다.

행동 예:

- 근거가 충분하면 종료
- 날짜가 모호하면 시간 검색
- 요약에서 예외가 빠진 것 같으면 원문 검색
- 실행 여부가 불명확하면 도구 기록 확인

**주의점**

정확한 기대값 계산은 비싸다. 실제로는 이득 예측기나 근거 충분성 지표가 필요하다. 불확실성이 줄어드는 것과 실제 작업 효용이 커지는 것도 구분해야 한다.

참고: [R21](#ref-r21)

<a id="section-e"></a>

## E. 언제 쓰고 정리하고 버릴 것인가: 장기 의사결정

<a id="theory-33"></a>

### 33. POMDP: 기억 정책을 부분관측 의사결정으로 보기

**핵심 개념**

에이전트는 실제 세계 상태를 완전히 알지 못하고 관측과 기억으로 추정한다. 메모리 저장·압축·검색도 미래 추정에 영향을 주는 행동이다.

**핵심 수식**

$$
b_t(s)=P(s_t=s\mid H_t)
$$

**장기 가치**

$$
V(b)=\max_a\mathbb E\!\left[r(b,a)+\gamma V(b')\right]
$$

**기호**

- b_t  : 세계 상태에 대한 믿음 분포
- V    : 장기적으로 얻을 수 있는 기대 가치
- $\gamma$: 미래 보상의 할인율

**메모리에 적용**

지금 버린 정보가 수십 단계 뒤의 작업 실패를 일으키는 환경에서, 단기 압축 이득과 장기 성능 손실을 함께 고려한다.

**주의점**

상태·보상·전이 모델의 설계와 학습 비용이 크다. 단기 QA에는 과할 수 있으며 초기 구현보다는 장기 연구 후보에 가깝다.

참고: [R22](#ref-r22)

<a id="theory-34"></a>

### 34. Constrained MDP / Lagrangian: 정확도와 비용을 함께 제어

**핵심 개념**

장기 보상을 높이되 전체 비용 예산을 지키도록 정책을 선택한다.

**핵심 수식**

$$
\max_\pi\mathbb E_\pi\!\left[\sum_t\gamma^t r_t\right]
$$

**비용 제약**

$$
\mathbb E_\pi\!\left[\sum_t\gamma^t c_t\right]\le B
$$

**라그랑주 방식의 보상 조정**

$$
r'_t=r_t-\lambda c_t
$$

**메모리에 적용**

예산을 많이 쓰면 비용 페널티를 높여 더 강하게 압축하고, 여유가 있으면 원문을 더 보존하는 방향을 검토한다.

**주의점**

평균 비용 제약과 매 요청의 엄격한 토큰 제한은 다르다. 실제 요청 상한은 별도 hard constraint로 유지해야 한다. 비용 페널티를 높이는 것만으로 개별 요청의 상한 준수가 보장되지는 않는다.

<a id="theory-35"></a>

### 35. Contextual bandit: 상황마다 다른 메모리 방법을 선택

**핵심 개념**

현재 상황을 보고 원문 보존, 요약, 구조화, 추가 검색 등의 방법을 선택하고 관측한 보상으로 선택 정책을 개선한다.

**LinUCB 형태의 선택 규칙**

$$
a_t=\arg\max_a\left[
x_t^\top\hat\theta_a+\beta\sqrt{x_t^\top V_a^{-1}x_t}
\right]
$$

**기호**

- $x_t$        : 현재 상황의 특징
- $\hat\theta_a$: 행동 a의 보상을 예측하는 추정 계수
- V_a        : 관측 정보량을 반영하는 행렬
- $\beta$       : 불확실성을 고려한 탐색 강도

**메모리에 적용**

질문 종류, 기록 길이, 시간 정보의 양 등에 따라 적합한 메모리 처리 방식을 선택한다.

**주의점**

전체 POMDP보다 가볍지만 먼 미래 상태를 바꾸는 효과를 충분히 반영하지 못할 수 있다. 삭제를 탐색 행동으로 실행하면 위험하다. 먼저 보관된 로그의 오프라인 평가 또는 복구 가능한 동작에 한정하는 것이 좋다.

참고: [R23](#ref-r23)

<a id="theory-36"></a>

### 36. Cache optimization: 사용 가능성과 없을 때의 손해를 함께 고려

**핵심 개념**

최근에 자주 쓰인 정보만 남기는 데서 벗어나, 필요할 가능성과 없을 때 발생할 손해를 함께 고려한다.

**핵심 수식**

$$
\max_{x_i\in\{0,1\}}\sum_i p_i m_i x_i
\qquad\text{subject to}\quad\sum_i c_i x_i\le B
$$

**기호**

- $p_i$: 해당 기억이 필요할 가능성
- $m_i$: 그때 기억이 없으면 발생하는 손해
- $c_i$: 보존 비용

**메모리에 적용**

아주 드물게 필요하지만 잃으면 큰 피해가 나는 조건을 보존한다. LRU/LFU의 최근성·빈도 관점을 작업 실패 비용까지 확장한다.

**주의점**

필요 확률과 누락 손해의 추정이 필요하다. 활성 메모리에서 내보내는 것과 원문을 영구 삭제하는 것은 구분해야 한다.

<a id="theory-37"></a>

### 37. Queueing theory / Lyapunov optimization: 정리 작업의 밀림 제어

**핵심 개념**

처리되지 않은 기록의 양을 큐로 보고, 정리량과 비용을 조절한다.

**핵심 수식**

$$
\begin{aligned}
Q_{t+1}&=\max(0,Q_t-\mu_t)+A_t,\\
L(Q_t)&=\frac12Q_t^2.
\end{aligned}
$$

**줄이려는 목적**

$$
\mathbb E[L(Q_{t+1})-L(Q_t)]+V\,\mathbb E[C_t]
$$

**기호**

- $Q_t$       : 처리되지 않은 기록의 양
- $A_t$: 새로 들어온 기록의 양
- $\mu_t$ : 이번 단계에서 처리한 기록의 양
- V         : 비용과 대기량 사이의 균형 조절

**메모리에 적용**

원문 유입이 빨라지면 정리량을 늘리고, 여유가 있으면 비용을 아낀다. 대규모 스트리밍 메모리에 적합하다.

**주의점**

유입량이 처리 능력을 계속 넘으면 어떤 스케줄러도 무한히 안정적일 수는 없다. 처리 가능성 조건을 확인해야 한다.

참고: [R24](#ref-r24)

<a id="theory-38"></a>

### 38. Bayesian online changepoint detection: 기억 사용 환경 변화 탐지

**핵심 개념**

질문 유형이나 오류 패턴이 바뀌었는지 온라인으로 탐지한다.

**판단식의 예**

$$
P(r_t=0\mid z_{1:t})>\tau
$$

**기호**

- $r_t$: 마지막 변화점 이후의 길이
- observations: 오류율, 질문 유형, 검색 특징 등의 관측

**메모리에 적용**

일상 대화에서 기술 작업으로 전환되거나 특정 유형의 검색 실패가 급증하는 상황을 탐지한다. 탐지 후 압축 정책을 보수적으로 바꾸거나 재보정할 수 있다.

**주의점**

질문 분포가 바뀌었다는 것과 기존 사실이 틀렸다는 것은 다르다. 변화 탐지 결과를 곧바로 사실 삭제 신호로 사용하면 안 된다.

참고: [R25](#ref-r25)

<a id="section-f"></a>

## F. 정말 해결했는가: 평가와 인과적 비교

<a id="theory-39"></a>

### 39. Factorial experiments: 저장 개선과 검색 개선의 효과 분리

> **우선 검토**

**핵심 개념**

저장 방식과 검색 방식을 교차 실험해 각각의 효과와 결합 효과를 구분한다.

**핵심 수식**

$$
\operatorname{Interaction}=Y_{11}-Y_{10}-Y_{01}+Y_{00}
$$

**기호**

- Y_ab: 저장 방식 a와 검색 방식 b를 함께 사용했을 때의 성능
- a,b in {0,1}

**실험 예**

원문 메모리 + 기본 검색기 원문 메모리 + 개선 검색기 요약 메모리 + 기본 검색기 요약 메모리 + 개선 검색기

**메모리에 적용**

저장 표현만 좋아진 것인지, 검색기만 좋아진 것인지, 둘을 함께 바꿔야 개선되는지 구분한다. 근거가 실제 메모리에 저장되어 있는지 직접 검사하는 진단도 함께 둔다.

**주의점**

reader, 데이터, 예산 등 다른 조건을 고정해야 한다. 상호작용이 존재하면 저장·검색 효과를 단순히 독립적인 기여율로 나누기 어렵다.

<a id="theory-40"></a>

### 40. Paired comparison / cluster bootstrap: 대화 단위 불확실성 측정

> **우선 검토**

**핵심 개념**

같은 사례에 두 방법을 적용하고 차이를 계산하되, 서로 연관된 질문들을 독립 표본으로 취급하지 않는다.

**핵심 수식**

$$
d_i=Y_i^{\mathrm{new}}-Y_i^{\mathrm{base}},
\qquad \widehat\Delta=\frac1n\sum_{i=1}^{n}d_i
$$

**메모리에 적용**

같은 대화에서 나온 질문은 묶어서 다룬다. 신뢰구간을 만들 때 대화·사용자 등 실제 독립 단위로 재표집한다. 여러 seed의 실행 변동도 별도로 보고한다.

**주의점**

질문 수가 많다고 독립 사례 수가 많은 것은 아니다. 작은 메모리 벤치마크에서는 개선 폭뿐 아니라 표본 의존성을 고려했는지가 결론에 큰 영향을 준다.

<a id="theory-41"></a>

### 41. Doubly robust evaluation: 과거 로그로 새 정책 평가

**핵심 개념**

기존 정책의 행동 선택 확률과 보상 예측 모델을 함께 사용해 새로운 정책의 가치를 추정한다.

**한 단계 contextual-bandit 설정의 수식**

$$
\widehat V_{\mathrm{DR}}=
\frac1n\sum_{i=1}^{n}\left[
\sum_a\pi(a\mid x_i)\hat r(x_i,a)
+\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}
\left(r_i-\hat r(x_i,a_i)\right)
\right]
$$

**기호**

- policy_logged: 과거 로그를 만든 정책
- policy_new   : 평가할 새 정책
- reward_hat   : 보상 예측 모델

**메모리에 적용**

새 메모리 정책을 실제로 배포하기 전에 로그로 평가하는 후보 방법이다.

**주의점**

기존 정책의 선택 확률을 알아야 한다. 새 정책의 행동이 로그에서 관측될 가능성이 있어야 한다. 메모리 행동이 미래 전체를 바꾸는 환경에서는 위의 한 단계 식만으로 충분하지 않다. 순차적 평가나 에피소드 단위 실험이 필요하다.

참고: [R26](#ref-r26)

<a id="theory-42"></a>

### 42. Amortized analysis / Pareto optimization: 비용 전가 여부 확인

> **우선 검토**

**핵심 개념**

검색 비용을 줄인 대신 쓰기·정리·저장 비용을 늘린 것인지 확인한다.

**질의당 평균 비용**

$$
\overline C=
\frac{C_{\mathrm{build}}+C_{\mathrm{maintenance}}+C_{\mathrm{training}}}{N}
+C_{\mathrm{query}}
$$

**함께 비교할 지표**

$$
\mathbf J=
\left(
\mathrm{error\ rate},
\mathrm{total\ tokens},
\mathrm{p95\ latency},
\mathrm{raw\ bytes},
\mathrm{active\ bytes}
\right)
$$

**메모리에 적용**

검색 토큰은 줄었지만 정리 비용이 몇 배가 된 방법을 구분한다. 활성 메모리는 줄었지만 원문은 무한히 쌓이는 방법도 구분한다.

**Pareto 비교**

한 방법이 다른 방법보다 모든 지표에서 나쁘지 않고 적어도 하나에서 좋으면 Pareto 우위라고 본다.

**주의점**

평균 비용식에 더하는 비용들은 같은 단위로 환산해야 한다. 토큰 수, 초, 바이트를 가중치 없이 직접 더하면 안 된다. 학습 비용이 많은 시스템은 예상 질의 수 N에 따라 유불리가 달라진다.

<a id="section-g"></a>

## G. 실제 연구로 묶을 만한 세 가지 조합

42개를 모두 한 시스템에 넣기보다는, 실패 원인에 맞는 작은 조합으로 시작하는 것이 타당하다. 다음 세 조합이 특히 자연스럽다.

### 1순위. 위험을 통제하는 residual memory

**조합**

[01](#theory-01) Rate-distortion [06](#theory-06) Cost-sensitive loss [09](#theory-09) Selective prediction [11](#theory-11) Learn then Test [15](#theory-15) Counterexample-guided refinement [25](#theory-25) Set cover / knapsack 또는 [27](#theory-27) Hypergraph optimization

**처리 과정**

1. 질문에 중요한 정보 손실을 정의한다.
2. 압축 후보와 원문 근거가 있는 진단 질문을 만든다.
3. 변경으로 인한 실패 위험을 예측한다.
4. 예산 안에서 필요한 원문 구절만 추가 보존한다.
5. 보정된 정책으로 변경을 승인하거나 기존 메모리를 유지한다.

**왜 먼저인가?**

현재 RetentionMem의 구조와 가장 직접적으로 연결된다. 정보 보존, 보존 비용, 변경 승인이라는 세 요소를 구체적으로 측정하고 비교할 수 있다.

**핵심 주의점**

통계적 보정은 정책 또는 승인 사례 집단에 대한 위험을 다룬다. 개별 변경 하나하나가 반드시 안전하다는 뜻으로 해석하면 안 된다.

### 2순위. 시간과 출처를 보존하는 갱신

**조합**

[17](#theory-17) 최소 변경 / belief revision [18](#theory-18) Bitemporal model [19](#theory-19) Provenance semiring [20](#theory-20) Truth maintenance system [23](#theory-23) Hoare logic / temporal invariants

**처리 과정**

1. 현재 사실과 과거 사실을 분리한다.
2. 새 정보가 실제로 무엇을 대체하는지 판단한다.
3. 근거 의존성을 따라 관련 부분만 수정한다.
4. 계획, 발언, 검증된 실행 결과를 구분한다.

**겨냥하는 실패**

- 이사했더니 과거 주소가 사라진다.
- 아직 실행하지 않은 작업을 완료로 기억한다.
- 한 근거가 틀렸다고 관련 없는 다른 사실까지 삭제한다.

**핵심 주의점**

시간·출처 구조만 갖췄다고 사실 판정이 자동으로 정확해지는 것은 아니다. 추출, 갱신 판단, 구조적 불변조건 검증을 구분해 평가해야 한다.

### 3순위. 근거 완성도를 기준으로 한 검색

**조합**

[27](#theory-27) Hypergraph optimization [28](#theory-28) Personalized PageRank [32](#theory-32) Value of information

**처리 과정**

1. 질문에 필요한 근거 묶음을 표현한다.
2. 관계를 따라 빠진 근거를 탐색한다.
3. 추가 검색의 기대 이득이 비용보다 클 때만 검색 범위를 넓힌다.

**왜 유용한가?**

단순 top-k 개선보다 “왜 이 근거 집합이면 답할 수 있는가”를 연구 대상으로 만들 수 있다.

**핵심 주의점**

근거 완성도를 판정하는 모델 자체가 틀릴 수 있다. 추정한 근거 집합과 실제 정답 근거를 별도로 평가해야 한다.

<a id="section-h"></a>

## H. 현재 코드와의 연결 및 반드시 구분할 것

아래 연결은 대화 당시 확인한 코드 구조를 기준으로 한다. 파일은 이후 수정될 수 있으므로 실제 구현 전에는 다시 확인한다.

### 1. 위험 예측과 승인 임계값

- 파일: [policy.py](/D:/MemoryData/methods/retentionmem/policy.py)
- 함수: `calibrate`
- 연결: [09](#theory-09), [10](#theory-10), [11](#theory-11), [12](#theory-12)
- 방향: 경험적 임계값 선택을 독립 검증과 위험 제어 절차로 발전시킨다.

### 2. residual 선택과 변경 승인

- 파일: [engine.py](/D:/MemoryData/methods/retentionmem/engine.py)
- 함수: `_choose`
- 연결: [15](#theory-15), [25](#theory-25), [26](#theory-26), [27](#theory-27)
- 방향: 추가 구절의 선택을 근거 커버리지·결합 효과·비용으로 최적화한다.

### 3. 진단 질문 생성

- 파일: [generation.py](/D:/MemoryData/methods/retentionmem/generation.py)
- 연결: [06](#theory-06), [15](#theory-15), [16](#theory-16)
- 방향: 숫자·부정·예외·시간·행동 조건을 검사하는 근거 기반 probe를 만든다.

### 4. 상태와 출처 표현

- 파일: [types.py](/D:/MemoryData/methods/retentionmem/types.py), [storage.py](/D:/MemoryData/methods/retentionmem/storage.py)
- 연결: [18](#theory-18), [19](#theory-19), [20](#theory-20), [23](#theory-23)
- 방향: 사실의 유효 시간, 근거, 버전, 검증된 행동 상태를 분리한다.

### 5. 평가

- 파일: [evaluation.py](/D:/MemoryData/methods/retentionmem/evaluation.py)
- 연결: [39](#theory-39), [40](#theory-40), [42](#theory-42)
- 방향: 구성요소의 효과, 대화 단위 불확실성, 전체 비용을 함께 비교한다.

### 항상 따로 확인해야 할 세 가지

1. 원문 정보가 남아 있는가?
2. 현재 reader가 그 정보를 제대로 사용하는가?
3. 그 성능을 새로운 상황에서도 기대할 근거가 있는가?

이 세 가지는 서로 다른 주장이다. 원문을 보존했다고 reader가 항상 올바르게 답하는 것은 아니다. 현재 reader의 성능이 좋다고 정보가 완전히 보존되었다는 뜻도 아니다. 검증 데이터의 성능이 좋다고 새로운 분포에서 같은 성능이 보장되지는 않는다.

### 추가로 지켜야 할 평가 원칙

- 진단 probe는 과거 원문에서 생성한다.
- 실제 미래 평가 질문이나 정답을 보존 정책 학습에 사용하지 않는다.
- probe 생성기와 검증기가 같은 사실을 함께 놓칠 수 있음을 고려한다.
- 독립 검증과 원문 근거 보존 기준을 함께 둔다.
- 퇴행 감소와 절대 정답률을 함께 측정한다.
- 원문 보관량, 쓰기·정리 비용, 검색·답변 비용을 모두 포함한다.
- 관측된 평균 오류율과 확률적 위험 상한을 구분한다.
- 이론을 사용했다는 사실과 이론의 전제를 만족했다는 사실을 구분한다.
- 기존 연구보다 새로운지는 별도의 관련 연구 검토로 확인한다.

**최종 우선 제안**

“질의에 따른 정보 손실 정의 + 최소 근거 보존 + 통계적으로 검증된 변경 승인” 조합을 먼저 실험하는 것이 가장 직접적이다.

구현상의 병목을 프레임워크의 본질적 한계와 혼동하지 않는 것도 중요하다. 앞선 대화에서는 vendored Mem0 코드가 추출 결과를 앞의 두 사실로 자르는 부분을 확인했다. 이런 제한으로 생긴 성능 저하는 이론적 한계나 방법 자체의 성능과 분리해서 검토해야 한다.

<a id="section-i"></a>

## I. 참고문헌과 원문 링크

앞선 대화에서 참고한 원전·연구자 자료의 링크를 정리했다. 일부는 원문 PDF이며, 접속 환경이나 사이트 상태에 따라 열리지 않을 수 있다. 이 문서를 정리하면서 별도의 신규성 검증이나 추가 실험을 수행한 것은 아니다.

<a id="ref-r01"></a>

**[R01] [Successive Refinement of Information](https://isl.stanford.edu/~cover/papers/paper94.pdf)**

저자: William H. R. Equitz, Thomas M. Cover  
내용: Rate-distortion 기초와 successive refinement

<a id="ref-r02"></a>

**[R02] [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057)**

저자: Naftali Tishby, Fernando C. Pereira, William Bialek

<a id="ref-r03"></a>

**[R03] [Modeling by Shortest Data Description](https://www.sciencedirect.com/science/article/pii/0005109878900055)**

저자: Jorma Rissanen  
내용: Minimum description length

<a id="ref-r04"></a>

**[R04] [Optimization of Conditional Value-at-Risk](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf)**

저자: R. Tyrrell Rockafellar, Stanislav Uryasev

<a id="ref-r05"></a>

**[R05] [Data-driven Distributionally Robust Optimization Using the Wasserstein Metric: Performance Guarantees and Tractable Reformulations](https://arxiv.org/abs/1505.05116)**

저자: Peyman Mohajerin Esfahani, Daniel Kuhn

<a id="ref-r06"></a>

**[R06] [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052)**

저자: Anastasios N. Angelopoulos 외

<a id="ref-r07"></a>

**[R07] [Conformal Risk Control](https://arxiv.org/html/2208.02814)**

저자: Anastasios N. Angelopoulos 외

<a id="ref-r08"></a>

**[R08] [Time-uniform, Nonparametric, Nonasymptotic Confidence Sequences](https://arxiv.org/abs/1810.08240)**

저자: Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, Jasjeet Sekhon

<a id="ref-r09"></a>

**[R09] [Counterexample-Guided Abstraction Refinement for Symbolic Model Checking](https://www.cs.cmu.edu/~emc/papers/Papers%20In%20Refereed%20Journals/Counterexample-guided%20abstraction%20refinement.pdf)**

저자: Edmund Clarke, Orna Grumberg, Somesh Jha, Yuan Lu, Helmut Veith

<a id="ref-r10"></a>

**[R10] [On the Logic of Theory Change: Partial Meet Contraction and Revision Functions](https://www.cambridge.org/core/journals/journal-of-symbolic-logic/article/abs/on-the-logic-of-theory-change-partial-meet-contraction-and-revision-functions/7ED837BAD5FB6D9A7C77906D73527F9C)**

저자: Carlos E. Alchourron, Peter Gardenfors, David Makinson

<a id="ref-r11"></a>

**[R11] [Temporal Support in Standard SQL](https://www2.cs.arizona.edu/~rts/DBPD/SQL/index.html)**

자료: Richard T. Snodgrass

<a id="ref-r12"></a>

**[R12] [Provenance Semirings](https://www.cs.ucdavis.edu/~green/papers/pods07.pdf)**

저자: Todd J. Green, Grigoris Karvounarakis, Val Tannen

<a id="ref-r13"></a>

**[R13] [A Truth Maintenance System 및 관련 연구 목록](https://groups.csail.mit.edu/medg/people/doyle/publications/)**

저자: Jon Doyle

<a id="ref-r14"></a>

**[R14] [An Axiomatic Basis for Computer Programming](https://www.cs.cmu.edu/~crary/819-f09/Hoare69.pdf)**

저자: C. A. R. Hoare

<a id="ref-r15"></a>

**[R15] [Strong Eventual Consistency and Conflict-free Replicated Data Types](https://www.microsoft.com/en-us/research/video/strong-eventual-consistency-and-conflict-free-replicated-data-types/)**

발표: Marc Shapiro / Microsoft Research

<a id="ref-r16"></a>

**[R16] [An Analysis of Approximations for Maximizing Submodular Set Functions—I](https://thibaut.horel.org/submodularity/papers/nemhauser1978.pdf)**

저자: G. L. Nemhauser, L. A. Wolsey, M. L. Fisher

<a id="ref-r17"></a>

**[R17] [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831)**

저자: Bernal Jimenez Gutierrez 외

<a id="ref-r18"></a>

**[R18] [The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries](https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf)**

저자: Jaime Carbonell, Jade Goldstein

<a id="ref-r19"></a>

**[R19] [Determinantal Point Processes for Machine Learning](https://arxiv.org/abs/1207.6083)**

저자: Alex Kulesza, Ben Taskar

<a id="ref-r20"></a>

**[R20] [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)**

저자: Aaron van den Oord, Yazhe Li, Oriol Vinyals

<a id="ref-r21"></a>

**[R21] [Rational Metareasoning / Intelligence 관련 자료](https://people.eecs.berkeley.edu/~russell/papers/ptai13-intelligence.pdf)**

저자: Stuart Russell

<a id="ref-r22"></a>

**[R22] [Planning and Acting in Partially Observable Stochastic Domains](https://cs.brown.edu/research/pubs/techreports/reports/CS-96-08.html)**

저자: Leslie Pack Kaelbling, Michael L. Littman, Anthony R. Cassandra

<a id="ref-r23"></a>

**[R23] [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146)**

저자: Lihong Li, Wei Chu, John Langford, Robert E. Schapire

<a id="ref-r24"></a>

**[R24] [Stochastic Network Optimization / Backpressure 자료](https://ee.usc.edu/stochastic-nets/docs/backpressure.pdf)**

저자: Michael J. Neely

<a id="ref-r25"></a>

**[R25] [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742)**

저자: Ryan Prescott Adams, David J. C. MacKay

<a id="ref-r26"></a>

**[R26] [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601)**

저자: Miroslav Dudik, John Langford, Lihong Li
