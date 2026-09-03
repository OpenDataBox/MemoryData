# RetentionMem

**Learning What to Retain During Memory Consolidation**

메모리를 압축하기 전에 과제 수행 능력의 손실을 예측하고, 필요한 원문 구절만 잔여 정보로 남기는 연구 구현이다. `evidence`를 시스템 이름이나 핵심 개념의 명칭으로 사용하지 않는다. 공식 데이터의 외부 필드명은 그대로 읽는다.

## 현재 상태

실행 가능한 코어, MemoryData adapter, 수집·학습·평가 CLI, AdaMEM 경험 표현 교체 도구, 공식 QA 로더가 있다. 실제 Qwen 실험과 ALFWorld 성공률은 아직 측정하지 않았다. 기본 설정의 로컬 모델 서버와 학습된 정책 체크포인트가 필요하다. 오프라인 데모와 모의 API 테스트는 연구 성능 결과가 아니다.

## 구조

| 파일 | 역할 |
|---|---|
| `types.py`, `storage.py` | 이벤트·출처 구간·메모리 계약, 변경 불가 원문, SQLite 원자적 교체 |
| `engine.py` | `observe`, `maintain`, `retrieve`; 유지/병합/보완 병합, 국소 영역 선택 |
| `generation.py` | 활성 메모리만 읽는 압축기와 원문 기반 검사 질문 생성기를 별도 호출 |
| `features.py`, `policy.py` | 고정 embedding 및 보존 특징, NumPy MLP, 별도 검증 집합 임계값 |
| `training.py` | 변경 전후 답변·독립 판정에 의한 라벨 생성, 원문 그룹 분할 검사 |
| `datasets.py`, `metrics.py`, `evaluation.py` | 공식 QA 입력, LoCoMo 범주별 F1, judge, 대화 단위 bootstrap |
| `alfworld.py` | 학습 trajectory 변환, AdaMEM 값 저장소 교체, unseen 환경 결과 집계 |

```text
observe(event) ──→ immutable archive
       └────────→ active memory
                       ↓ local region (≤8)
               compressor(active only) → summary
archive prefix → probe generator (≤8) → frozen features → small policy
                                              ↓
                    keep / merge / merge + selected source spans
                                              ↓
                               transactional active replacement
retrieve(query, budget) ← active memory only → frozen reader
```

온라인 병합은 답변 모델의 변경 전후 재실행을 하지 않는다. 압축·검사 생성에 LLM 2회, 고정 embedding, 작은 정책이 필요하다. 검증 기준 비교군은 추가 verifier 호출을 한다. 출처 원문은 질의 시 읽지 않는다. 요약과 출처 ID의 연결만으로 실제 정답 구절이 남았다고 주장하지 않는다.

## 설치와 빠른 확인

저장소 루트에서 Python 3.12를 사용한다. 이 작업에서 만든 환경은 `.runtime-cache/retentionmem-venv`다.

```powershell
uv venv .runtime-cache/retentionmem-venv --python 3.12
uv pip install --python .runtime-cache/retentionmem-venv/Scripts/python.exe -r methods/retentionmem/requirements.txt
$retentionPython = '.runtime-cache/retentionmem-venv/Scripts/python.exe'
& $retentionPython -m pytest tests/retentionmem -q
& $retentionPython -m methods.retentionmem demo --output results/retentionmem/demo
& $retentionPython -m methods.retentionmem doctor --config config/retentionmem/runtime.yaml
```

`demo`는 `heuristic_residual`과 결정적 예제 압축기를 사용한다. 출력의 `test_artifact: true`와 `scientific_accuracy: null`을 유지한다. 테스트용으로 학습한 체크포인트도 본 평가에서 거부한다. 일반 연구 모드에서 모델·embedding·정책을 다른 것으로 자동 대체하지 않는다.

## 서버 설정

`config/retentionmem/runtime.yaml`의 `llm`은 압축기, `reader`는 답변 모델, `embedding`은 고정 검색·특징 모델, `judge`는 판정 모델이다. 모델 이름은 서버가 제공하는 이름과 맞춰야 한다. 기본값은 Qwen3-8B, Qwen3-Embedding-4B, 별도 Qwen2.5-7B-Instruct judge다. 공식 판정 *rubric*을 사용하는 것과 원 논문의 judge 모델까지 동일한 재현은 구분한다.

서버에는 OpenAI 호환 `/chat/completions`, `/embeddings`, JSON 응답 지원이 필요하다. 선택적인 API 키는 지정된 환경변수에서 읽는다. `doctor`는 포트 접근 여부와 체크포인트를 확인하며 모델 정체까지 검증하지는 않는다. 모델 revision, 서버 버전, hardware, serving 옵션은 실험 기록에 추가한다.

## 수집과 학습

```powershell
# 파일럿 500개 원문 영역: train 400 / validation 100
& $retentionPython -m methods.retentionmem synthetic --train 400 --validation 100 --output results/retentionmem/pilot_sources.jsonl
& $retentionPython -m methods.retentionmem collect --config config/retentionmem/runtime.yaml --input results/retentionmem/pilot_sources.jsonl --output results/retentionmem/pilot_labels.jsonl
& $retentionPython -m methods.retentionmem train --input results/retentionmem/pilot_labels.jsonl --output results/retentionmem/pilot_policies --seeds 0 1 2

# 본 학습: train 3000 / validation 750; 테스트 benchmark는 읽지 않는다.
& $retentionPython -m methods.retentionmem synthetic --output results/retentionmem/train_sources.jsonl
& $retentionPython -m methods.retentionmem collect --config config/retentionmem/runtime.yaml --input results/retentionmem/train_sources.jsonl --output results/retentionmem/train_labels.jsonl
& $retentionPython -m methods.retentionmem train --input results/retentionmem/train_labels.jsonl --output results/retentionmem/policies --seeds 0 1 2
```

수집 재개는 동일 설정으로 `--resume`를 붙인다. 기본 합성기는 배송지·출장·시간 조건·소유 관계를 조합하는 **좁은 파일럿 분포**다. 3,750개 생성만으로 다양한 학습 분포를 확보한 것으로 간주하지 않는다. 본 실험 전 별도 합성 도메인을 추가하고 ALFWorld train 경험을 포함해야 한다. 외부에서 만든 JSONL도 다음 계약을 따르면 수집할 수 있다.

```json
{"group_id":"source-family-1","split":"train","origin":"synthetic","events":[{"id":"event-1","text":"Observed text","timestamp":"2025-01-01","actor":"user","kind":"dialogue"}]}
```

`origin`은 `synthetic` 또는 `alfworld_train`, `split`은 `train` 또는 `validation`이다. 동일 원문의 변형은 동일 `group_id`로 묶는다. ID만 바꾼 중복 원문의 분할 누수도 거부한다. 수집기는 미래 관찰·벤치마크 질문·정답 필드를 받지 않는다. 다른 표현으로 의역된 누수까지 자동으로 증명하는 검사는 아니다.

원문에서 만든 기준 답을 별도 judge가 검증하고, 변경 전 답이 맞은 질문만 회귀 학습에 사용한다. 요약 후보와 일부 잔여 구절을 보완한 후보를 모두 라벨링한다. JSON 파싱 실패는 오답으로 바꾸지 않는다. 실제 관찰이 바뀌면 현재 사실의 기준 답도 바뀐다.

MLP는 267→256→64→1, 두 은닉층, Adam이다. 기반 모델은 업데이트하지 않는다. 검증 후보별 최대 예측 위험과 실제 회귀 여부를 집계해, 기본 경험적 회귀율 5% 이하이면서 최소 20개 후보가 승인되는 가장 큰 임계값을 고른다. 조건을 만족하지 않으면 임계값 -1로 전부 유지한다. 이는 통계적인 무손실 보장이 아니다.

## QA 실행

```powershell
& $retentionPython -m methods.retentionmem download --benchmark locomo --output datasets/retentionmem
& $retentionPython -m methods.retentionmem download --benchmark longmemeval --output datasets/retentionmem
& $retentionPython -m methods.retentionmem run --config config/retentionmem/runtime.yaml --benchmark locomo --data datasets/retentionmem/locomo10.json --output results/retentionmem/locomo-main
& $retentionPython -m methods.retentionmem run --config config/retentionmem/runtime.yaml --benchmark longmemeval --data datasets/retentionmem/longmemeval_s_cleaned.json --output results/retentionmem/longmemeval-main
```

LoCoMo는 공식 10개 대화, 기본 범주 1–4의 1,540문항이다. 날짜·화자·제공된 이미지 캡션을 포함한다. `--include-adversarial`로 446개 범주5를 추가할 수 있다. 이 범주의 `adversarial_answer`는 정답이 아니라 오답 선택지다. 주 metric은 공식 범주별 F1 처리(`locomo_f1`)이며 generic `token_f1`와 구분한다. 별도 semantic judge 정확도는 자체 보조 metric이다. [공식 평가 코드](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)

LongMemEval은 공식 cleaned S JSON을 그대로 읽으며 대화나 assistant 발언을 자르지 않는다. 공식 유형별 rubric을 재사용하고 configured judge 모델을 명시한다. 실행일의 날짜를 질문에 삽입하지 않는다. [공식 데이터·평가 안내](https://github.com/xiaowu0162/LongMemEval)

`--max-cases`, `--max-questions`는 파일럿용이다. 제한을 사용한 실행은 manifest에 남는다. 전체 평가로 표시하지 않는다. `--no-judge` 실행은 정확도–비용 그림에 사용할 수 없다. `--resume`는 데이터 hash·설정·체크포인트 hash가 일치할 때만 허용한다.

## 비교군과 반복 실험

`mode`는 `raw`, `keep`, `always_merge`, `heuristic`, `heuristic_residual`, `criteria_verifier`, `learned`, `learned_residual`이다. `raw`와 `keep`은 이 엔진에서는 동일한 원문 hybrid RAG 동작이다. 별개 기여나 독립 반복으로 세지 않는다. structured/topical은 각각의 압축 방식에서 착안한 고정 프롬프트이며 SimpleMem/LightMem 전체 구현이 아니다. `criteria_verifier`도 TrustMem의 학습된 원 모델이 아니다.

```powershell
# 2K/4K/8K, 학습 seed 0/1/2의 실행 설정과 argv를 만든다. 자동 실행하지 않는다.
& $retentionPython -m methods.retentionmem matrix --config config/retentionmem/runtime.yaml --output results/retentionmem/matrix
# writer 2종 × reader 2종 × rewrite rounds 1/2/4/8까지 확장
& $retentionPython -m methods.retentionmem matrix --config config/retentionmem/runtime.yaml --output results/retentionmem/matrix-full --full
& $retentionPython -m methods.retentionmem compare --left RUN_A/predictions.jsonl --right RUN_B/predictions.jsonl
& $retentionPython -m methods.retentionmem plot --runs RUN_R1 RUN_R2 RUN_R4 RUN_R8 --output results/retentionmem/accuracy_cost.png
```

각 job의 `arguments` 배열을 저장소 루트에서 실행한다. 이식성 설정은 `reader`만 변경하고 writer/embedding/checkpoint는 고정한다. seed별 곡선을 합칠 때 독립 질문 수를 부풀리지 않는다. QA 신뢰구간은 대화 그룹을 재표집한다. LongMemEval에서는 현재 각 독립 haystack을 한 그룹으로 처리한다.

`--rounds`는 최초 온라인 구축 이후 같은 원문 prefix에 대해 추가 국소 유지보수 pass를 요청한다. 따라서 **각 개별 원문의 실제 병합 횟수와 같지 않다.** 유지 선택이나 길이 한계로 병합하지 않을 수 있고, 최초 구축에도 여러 병합이 있다. 실제 계보 깊이는 각 unit의 `version`, 실제 선택은 audit를 사용한다. 논문에서 pass 수를 실제 병합 횟수로 바꿔 쓰지 않는다.

`compression_ratio=4`는 요약 목표이고 최종 잔여 구절까지 포함한 강제 4:1 비율이 아니다. 같은 압축률 비교는 `maintain(region, budget)`의 동일 교체 상한을 이용하거나 실제 활성 토큰 기준으로 맞춰야 한다. 위험 때문에 유지한 경우 `budget_met=false`가 가능하다. 질의 예산은 강제 상한이다. 메모리 예산은 cl100k 기준이며 Qwen 서버의 실제 usage 토큰도 별도로 기록한다.

## ALFWorld / AdaMEM

ALFWorld와 AdaMEM의 환경 의존성을 설치한 Linux 실행 환경이 필요하다. 현재 Windows 검증 환경에서는 실제 에피소드를 실행하지 않았다.

```powershell
& $retentionPython -m methods.retentionmem alfworld-streams --trajectories PATH/traj_train.json --output results/retentionmem/alfworld_sources.jsonl
& $retentionPython -m methods.retentionmem adamem-prepare --config config/retentionmem/runtime.yaml --repo methods/adamem --memory-model Qwen/Qwen3-8B --subset correct_only --output results/retentionmem/adamem-residual
```

먼저 원 AdaMEM의 train trajectory와 `build_index.py`로 만든 index/meta/values가 있어야 한다. `adamem-prepare`는 원 clone 밖에 실행 디렉터리를 만들고 **검색 키와 HNSW 파일을 그대로 복사**한다. 같은 정수 ID의 경험 텍스트만 바꾼다. 원본/always_merge/learned_residual을 각각 준비한 후 같은 action/strategy 모델, retrieval top-k, 서버 설정으로 upstream runner를 실행한다. 값 저장소는 plain `list[str]`만 허용한다. 출처 hash와 변환 비용을 기록한다.

```bash
# AdaMEM 환경이 설치된 Linux에서, 준비된 디렉터리를 현재 작업 디렉터리로 사용
SPLIT=eval_out_of_distribution ENV_NAME=alfworld MEM_TYPE=adamem-low \
  MODEL_NAME=Qwen/Qwen3-8B MEMORY_MODEL_NAME=Qwen/Qwen3-8B CORRECT_ONLY=true \
  OPENAI_BASE_IP_ADDR=127.0.0.1:9909 \
  python -m examples.prompt_agent.gpt4o_alfworld
```

정확한 모델 별칭과 embedding 서버 설정은 원 AdaMEM 코드에 맞춘다. fork는 `gamefile`을 log에 추가하는 한 줄을 제외하고 원 runner를 사용한다. 공식 unseen 게임 ID 목록을 확보한 뒤 결과를 집계한다.

```powershell
& $retentionPython -m methods.retentionmem alfworld-score --trajectories PATH/traj_eval_out_of_distribution_adamem-low.json --episode-ids PATH/official_unseen_gamefiles.json --output results/retentionmem/alfworld-score.json --mode adamem-low
```

완료되지 않은 `won`, 누락된 환경, 환경 ID 불일치, 같은 환경/round 중복을 거부한다. 성공은 환경의 `won`으로 평가하며 행동 문자열 변경을 실패로 보지 않는다. 선행 조건 위반률은 아직 환경별 판정기가 없으므로 `null`이다. action log의 토큰은 cl100k 재계산 추정값이다. 원문/변경 값 저장 비용과 offline 유지보수 비용을 더해 보고한다.

## 비용과 해석

`predictions.jsonl`, `summary.json`, `manifest.json`, 각 memory의 SQLite/audit/`run_stats.json`에 결과가 남는다. 원문·활성 JSON bytes, SQLite 전체 bytes, LLM input/output, embedding input, policy 시간, query 지연, cache hit를 기록한다. 중첩 API 시간과 wall time을 합하지 않는다. API usage 누락은 표시하며 0원으로 해석하지 않는다.

수집 비용은 `labels.jsonl.costs.json`, CPU 정책 학습 시간은 `training_report.json`에 별도로 남는다. 현 plot의 x축은 **쓰기·유지보수·읽기를 합산한 질의당 LLM 토큰 수**다. 금전적 전체 비용이나 학습 비용 포함 결과로 표시하지 않는다. 전체 비용은 동일한 비용 단위로 `C_train + C_write + C_maintain + N*C_read`를 계산하고, 원문·인덱스 저장도 포함해야 한다. 단가를 설정하지 않았으므로 USD는 null이다.

추가 구축 비용 ΔW와 질의당 절감량 ΔR이 같은 단위이고 ΔR>0일 때 회수 질의 수는 `ceil((C_train+ΔW)/ΔR)`다. ΔR≤0이면 유한 회수 지점을 주장할 수 없다. 비용 집계 범위를 바꿔 절감률을 부풀리지 않는다.

시스템 baseline 실행 감사와 미완료 실험은 `research/retentionmem/protocol.md`를 참고한다. MemoryData의 기존 heavy dependency 환경에서는 `config/hybrid_retentionmem.yaml`로 adapter를 사용한다. 주 연구 평가에는 위 standalone CLI를 사용한다.
