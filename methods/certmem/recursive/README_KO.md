# LoCoMo 100문항 recursive construction pilot

## 지속 refinement — 실행 중

사용자 요청에 따라 첫 파일럿 이후에도 반복 개선을 계속한다. `continuous.py`는 고정된 기존
`refine.py`의 reader·검색·채점·2,048토큰 상한을 유지하며 메모리 구성만 바꾼다.
앞서 공개된 검증 30문항의 대화는 이 worker의 생성·평가 대상에서 제외하고, 기존 개발 70문항으로
후보를 선택한다. 이후 개발 점수 상승을 새 일반화 성능으로 보고하지 않는다.

`continuous_queue.json`의 후보를 구성 전에 고정하고 평가한다. 개선된 후보만 현재 최선으로
채택하며 `parent: best`인 다음 후보의 출발점으로 쓴다. 완료된 후보를 바꾸지 않고 새 이름으로
추가한다. 대기열이 비면 worker는 새 후보를 기다리며 실행을 유지한다.

원격 실행: `supervisorctl start locomo-continuous`. 실행 설정은 `run_continuous_4080s.sh`.
현재 결과는 `runs/continuous_v1/{protocol,history,status}.json`, 후보 계획은 `plans/`,
메모리는 `memories/`, 문항별 예측과 진단은 후보 이름으로 저장한다. 대기열 갱신은 임시 파일로
업로드한 뒤 원자적으로 교체한다. worker는 기존 실행 결과와 프로토콜이 바뀌면 재개를 거부한다.

처음 추가한 후보는 날짜 명시, 발화 중복 제거, 3·4턴 문맥, 일반적 격려 사실 제거,
주체·사건·정확한 표현을 보존한 카드, 직접적인 사실 문장 및 조합이다. 결과를 분석하며 후속
후보를 계속 추가하며, 후보 개수만으로 실험 목표를 완료 처리하지 않는다.

### 두 번째 구성 세대

`continuous_v2.py`와 `memory_ops.py`는 기존 세대 파일을 수정하지 않고 다음 연산을 추가한다.

- 상대 시간이 실제로 있는 메모리만 날짜를 명시해 다른 메모리 표현을 보존한다.
- 요약 사실을 해당 출처의 원문 블록에 붙여 중복 검색과 문맥 분산을 줄인다.
- 생성된 카드의 중복 값, 답이 없는 질문, 일반적인 감탄사를 정리한다.
- 대화에서 유래한 고정된 주제 질의로 인물별 가족·취미·직업·건강·여행·지원 정보를 모은다.
- 원문에서 정보가 없는 인사·감사·격려만 판별해 줄이고, 의미 있는 답변과 이미지 정보는 보존한다.

이 연산에도 벤치마크 질문·정답을 전달하지 않는다. 인물 프로필은 메모리 생성 단계의 고정된
주제 질의를 사용하며, 평가 시 검색과 reader는 기존 코드를 유지한다. `last week`의 날짜 범위는
달력상 전주라는 해석이며 원래 상대 표현도 보존한다. 저장된 날짜를 별도의 관측 사실로 간주하지 않는다.

`locomo-continuous-transition`은 기존 worker가 현재 대기열을 모두 마치고 실제 실행 상태에서
새 후보를 기다리는 것을 확인한 뒤 종료하고 `locomo-continuous-v2`를 시작한다. 기존 점수·메모리·
캐시를 `runs/continuous_v2/`로 이어받고 새 소스 해시와 부모 프로토콜 해시를 기록한다.
두 번째 세대 대기열은 `continuous_v2_queue.json`이다. 새 worker 시작 후 해당 소스를 수정하려면
다시 별도 세대와 output directory를 만든다.

### 세 번째 구성 세대 — 검색 키와 읽기 원문 분리

`continuous_v3.py`, `indexed_memory.py`, `evaluate_indexed.py`에서 메모리에 `index_text`를
둘 수 있도록 확장했다. 검색은 짧은 사실이나 정리된 키로 하고, reader에는 연결된 원문 `text`를
준다. BM25+dense RRF 방식·reader·2,048토큰 읽기 상한은 유지한다. 기존 메모리는 원래 평가
함수로 그대로 보내며, 57.11887 및 52.79428점이 재현되지 않으면 새 비교를 시작하지 않는다.

저장 비용은 서로 다른 검색 키와 읽기 원문을 모두 합산한다. 후보에는 출처의 앞뒤 문맥 범위,
사실·원문 동시 읽기, 아직 사실로 추출되지 않은 원문 보존, 검색 키에서 메타데이터 제거,
최선인 4턴 결합 메모리를 검색 키와 원문으로 분리하는 방식이 있다.

`locomo-continuous-transition-v3`은 두 번째 세대의 전체 대기열 완료를 확인한 뒤
`locomo-continuous-v3`를 시작한다. 이전 후보 결과는 `runs/continuous_v3/`로 이어받고,
`continuous_v3_queue.json`의 새 후보를 계속 평가한다. 기준 구성 재현 검사는 필수 통과 조건이다.

실험 대상 서버: 사용자가 임대한 Vast.ai 인스턴스 **50143064**, RTX 4080 SUPER,
`nvidia-smi`에서 32,760 MiB 확인. 원격 폴더: `/workspace/locomo-refinement`.
기존 v15 실험과 분리된 구성 실험이며, v15 전체의 재현 점수로 해석하면 안 된다.

## 2026-09-07 실측 결과

5개 후보를 개발 70문항에서 비교한 뒤 `r03_dialogue_residual`을 고정하여 별도 30문항을 평가했다.
전체 100문항 공식 F1은 **44.69 → 53.46 (+8.77점)**, 별도 검증 30문항은
**38.01 → 55.02 (+17.01점)**. 같은 읽기 예산의 원문 RAG는 전체 **46.97**이다.
누락 사실 재검사와 인접 발화 보존을 결합한 구성으로, 읽기 상한은 2,048토큰이나 저장량은
기준 대비 **7.25배**다. 이득은 주로 단일 사실 질문에 집중됐으며 다단계·시간 추론은 소폭 하락했다.

보고서: `D:\MemoryData\results\recursive_locomo_20260907\REPORT_KO.md`.
실측 결과·고정 프로토콜·메모리·예측은 같은 폴더의 `final/runs/pilot100_v3/`,
원격 소스와 환경 정보는 `final/`에 회수했다. GPU 실험은 완료 상태로 종료됐다.

## 평가 규칙

- LoCoMo 공식 데이터의 10개 대화에서 대화당 10문항, 총 100문항을 고정한다.
- 각 대화 내부에서 category 1–4의 원래 분포에 비례해 표본을 고른다. category 5는 제외한다.
- seed=20260907, 개발 7개 대화/70문항, 별도 검증 3개 대화/30문항.
- 개발 F1로 후보를 선택한 후 `selection_locked.json`을 쓰고 검증을 시작한다.
- 메모리 생성기는 대화 세션만 받는다. 벤치마크 질문·정답·근거를 생성기에 전달하지 않는다.
- Qwen3-8B, thinking off, temperature=0, 동일 reader prompt, 같은 BM25+dense RRF 검색,
  실제 tokenizer로 계산한 **최대 2,048 읽기 토큰**을 모든 후보에 적용한다.
- 주지표는 [공식 LoCoMo category F1](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py).
  기존 certmem의 날짜/숫자 정규화 F1과는 다른 지표다. 일반 token F1도 보조 기록한다.
- 메모리 **저장량은 고정하지 않는다**. 저장 토큰을 함께 보고하여 더 많은 정보를 보관한 비용을 드러낸다.
- source recall은 생성기가 명시한 출처 ID의 recall이며 사실 entailment나 정확한 보존 증명이 아니다.
- 100개 전체 점수는 개발/검증 혼합이다. 일반화 근거는 별도 30개 검증 점수로만 해석한다.
  검증 대화가 3개뿐이므로 bootstrap 구간 역시 작은 파일럿의 기술 통계다.

## 반복 구성

1. `session10`: 세션당 최대 10개 사실을 보관하는 baseline (Mem0-style prompt; upstream Mem0 아님).
2. `events`: 18턴 창에서 사건·참여자·시간·장소·이유·관계가 함께 남도록 세밀한 사실을 생성.
3. `audit`: 개발 점수가 가장 높은 메모리를 원문과 다시 비교해 누락된 사실을 추가.
4. `dialogue_residual`: 현재 최선 메모리에 인접 발화 쌍을 보관해 짧은 응답의 맥락을 보완.
5. `provenance_quotes`: 현재 최선의 사실 메모리에 해당 출처의 원문 발화를 직접 연결한다.
   같은 출처 묶음을 공유하는 사실은 합쳐 중복을 줄인다. 개발 문항에서 확인한 구체성 손실과
   paraphrase의 F1 저하를 겨냥해 검증 데이터 사용 전에 추가한 후보다.

개발 F1이 0.001 이상 개선된 후보만 새 최선으로 채택한다. 원문 RAG도 같은 reader/읽기 예산으로
별도 평가한다. 채택되지 않은 후보, 생성 실패의 원문 fallback, 프롬프트별 생성 결과도 남는다.
이 루프는 모델 가중치 학습이 아니라 **메모리 표현/보관 방식의 개선**이다.

## 실행과 산출물

```bash
cd /workspace/locomo-refinement
bash setup_4080s.sh
bash run_4080s.sh
```

장시간 실행은 `locomo-setup`, `locomo-refine` supervisor program으로 관리한다.
설치/실험 상태: `supervisorctl status locomo-setup locomo-refine`.

초기 `runs/pilot100/`은 개발 단계에서 중단했고 원본 코드 `refine_v1.py`를 보존했다.
추가 후보 실험 `runs/pilot100_v2/`에서 동일한 질문 분할과 생성/임베딩 캐시를 재사용한다.
기존 memory artifact를 무조건 복사하지 않고 각 구성기를 다시 실행하므로 달라진 프롬프트는
새로운 캐시 키로 처리한다. 초기 run은 holdout을 평가하기 전에 중단한 것을 확인했다.

v2의 원문 연결 후보 임베딩에서 batch32가 VRAM을 초과하여 v3에서 기본 batch4 및 OOM 시
2/1로 재시도하도록 수정했다. 생성 캐시는 재사용하고, 임베딩 캐시 키에 새 배치 정책을 넣어
모든 후보의 임베딩을 다시 계산한다. v2 역시 holdout을 평가하기 전에 종료됐다.

현재 `runs/pilot100_v3/` 아래 `manifest.json`, `protocol.json`, `history.json`, `status.json`,
`selection_locked.json`, `final.json`, 후보별 문항 예측과 근거 문맥, 메모리 파일을 저장한다.
프롬프트/모델 revision/설정별 캐시로 중단 후 같은 protocol을 재개한다. 프로토콜이나 데이터가
바뀌면 기존 결과를 재사용하지 않고 새 output directory를 지정해야 한다.

검증 완료 후 같은 holdout을 보고 계속 후보를 고르면 더는 미사용 검증이 아니다.
후속 개선은 기존 개발 대화 안에서 하거나 새 검증 split을 사전 고정하고 별도 실행한다.
원격 디스크는 destroy/recycle 시 유실될 수 있으므로 결과를 로컬로 회수한다.

```powershell
.venv-ollama-pilot\Scripts\python.exe -m unittest discover -s methods/certmem/recursive -p test_refine.py
```
