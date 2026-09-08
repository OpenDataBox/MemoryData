# 여섯 메모리 baseline 공통 실행

LangMem, Mem0, A-MEM, LightMem, SimpleMem, E-mem을 `main.py`로 실행한다.
각 방법과 데이터셋 조합을 별도 프로세스로 순서대로 실행한다. 기본값은 6 × 2 = 12개 조합이다.

## 환경과 데이터

저장소 루트에서 Python 3.12 환경을 사용한다. 현재 이 작업 공간에는 `.venv-six` 설치와 데이터 다운로드가 완료되어 있다.
새 환경에서는 다음과 같이 준비한다.

```powershell
uv venv .venv-six --python 3.12
uv pip install --python .venv-six/Scripts/python.exe -r requirements-six-baselines.txt
& .venv-six/Scripts/python.exe scripts/download_memory_benchmarks.py
```

- LoCoMo: 원본 `locomo10.json`의 대화 10개, category 1–4의 1,540문항. Category 5 adversarial은 이 설정에서 제외한다. 기존 600문항 샘플 설정과 구별한다.
- LongMemEval-S: 공식 cleaned JSON의 500개 질문과 각 질문에 해당하는 전체 대화 이력. 기존 MemoryAgentBench 변형 대신 `longmemeval_s_official`로 실행한다.
- 정답, `has_answer`, `answer_session_ids`는 메모리 생성 입력에 넣지 않는다. 세션 날짜, 발화자, 질문 시점은 유지한다.

원본: [LoCoMo](https://github.com/snap-research/locomo), [LongMemEval](https://github.com/xiaowu0162/LongMemEval).

## 실행

OpenAI 호환 chat 서버와 embedding 서버가 필요하다. 아래 주소와 모델명은 예시이며 실제 서버의 `/v1/models`에 등록한 이름을 사용한다.
LangMem에는 tool calling 지원이 필요하다. E-mem text mode는 upstream의 `repetition_penalty` 요청을 사용하므로 이를 지원하는 vLLM 등의 서버를 사용한다.
`--embedding-dim`은 서버가 실제로 반환하는 벡터 차원에 맞춘다.
llama.cpp 임베딩 서버로 LightMem의 4,096자 청크를 처리할 때는 기본 `ubatch=512`가 부족할 수 있으므로
`--batch-size 2048 --ubatch-size 2048` 이상으로 실행한다.

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY"
$env:OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
$env:EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"

# 먼저 방법별로 한 이력·한 질문씩 확인: 여섯 방법 × 두 데이터셋
& .venv-six/Scripts/python.exe scripts/run_six_baselines.py `
  --model Qwen3-8B `
  --embedding-model Qwen3-Embedding-4B --embedding-dim 2560 `
  --max-samples 1 --max-queries 1 `
  --artifact-root results/six_smoke

# 전체 평가: LoCoMo 대화 10개 + LongMemEval-S 500개
& .venv-six/Scripts/python.exe scripts/run_six_baselines.py `
  --model Qwen3-8B `
  --embedding-model Qwen3-Embedding-4B --embedding-dim 2560
```

`--dry-run`을 추가하면 API를 호출하지 않고 설정과 실행 명령만 생성한다.
`--methods e_mem langmem`, `--datasets longmemeval`로 대상을 줄일 수 있다.
`--max-samples`는 대화 이력 개수, `--max-queries`는 각 방법·데이터셋 실행의 전체 질문 개수 제한이다.
`--max-context-chunks`는 파이프라인 연결만 확인하는 디버그용 제한이다. 이 옵션을 사용한 결과는 전체
LoCoMo/LongMemEval 비교 점수로 사용하지 않는다.
키 없는 로컬 서버에서는 `OPENAI_API_KEY`에 `EMPTY` 같은 값을 설정한다.

### Bash 공통 래퍼

저장소 루트에서 다음 명령을 실행하면 여섯 방법과 두 데이터셋을 모두 순회한다. 기본 실행은 각 조합에서 이력 1개와 질문 1개만 처리하는 smoke run이다.

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:9911/v1
export EMBEDDING_BASE_URL=http://127.0.0.1:9010/v1

bash scripts/run_six_baselines.sh
```

기본 모델은 `Qwen3.5-0.8B`, 임베딩 모델은 `Qwen3-Embedding-0.6B`, 임베딩 차원은 1,024이다. 서버 설정이 다르면 환경변수로 덮어쓴다.

```bash
MODEL=Qwen3-8B \
EMBEDDING_MODEL=Qwen3-Embedding-4B \
EMBEDDING_DIM=2560 \
bash scripts/run_six_baselines.sh --dry-run
```

전체 LoCoMo와 LongMemEval-S 평가는 반드시 `--full`을 명시한다. 매우 오래 걸리며, 기존 상태와 결과를 섞지 않으려면 실행마다 새 `ARTIFACT_ROOT`를 지정한다.

```bash
ARTIFACT_ROOT="$PWD/results/qwen35_full_$(date +%Y%m%d)" \
bash scripts/run_six_baselines.sh --full
```

일부 방법이나 데이터셋만 고르는 옵션은 공통 Python 러너로 그대로 전달된다.

```bash
bash scripts/run_six_baselines.sh \
  --methods e_mem lightmem \
  --datasets locomo
```

## 메모리와 결과

첫 실행은 대화 이력을 읽어 **메모리 생성 → 저장 → 질문별 검색·답변** 순서로 진행한다.
완성된 상태가 있으면 다음 실행에서 재사용하고, 이미 성공한 질문도 건너뛴다. 실패한 질문은 다시 시도한다.
LangMem은 저장된 메모리 텍스트를 재사용하되 로드할 때 임베딩을 다시 계산한다.
메모리 생성 도중 중단되었을 때의 부분 복구는 방법마다 다르며, 모든 방법이 청크 단위 재개를 지원하는 것은 아니다.

기본 결과 루트는 `results/six_baselines/<method>/<dataset>/`이다.
생성 설정은 `results/six_baselines/configs/`, 실행 실패 목록은 `failures.json`에 남는다.
질문 실패가 기록되면 해당 조합도 실패로 집계한다. 다른 조합은 계속 실행한다.
모델, 임베딩, 데이터 내용을 바꾼 실험에는 새 `--artifact-root`를 사용해 이전 메모리와 섞이지 않게 한다.

E-mem은 공식 **text storage mode + hybrid router + 블록별 추론 + aggregation**을 사용하고, 그 근거를 공통 답변 생성기에 전달한다.
KV-cache/GPU 모드는 이번 공통 실행 경로에 포함하지 않는다. 첫 메모리 생성 시 `Qwen/Qwen3-4B` tokenizer를 다운로드한다.
LangMem/E-mem의 `token_usage_scope: final_answer_only`는 마지막 답변 호출의 토큰만 뜻한다. 메모리 생성·검색 내부 호출까지 포함한 총비용으로 비교하면 안 된다.

## LongMemEval 판정

공통 실행기의 자동 문자열 점수와 공식 방식의 LLM judge 점수는 구분한다.
답변 생성 후 기존 judge 스크립트로 별도 판정한다. 아래 모델명과 endpoint는 실제 judge 서버로 바꾼다.

```powershell
& .venv-six/Scripts/python.exe evaluation/longmemeval/evaluate_longmemeval_judge.py `
  --result_glob "results/six_baselines/*/longmemeval/**/*_results.json" `
  --dataset-json datasets/LongMemEval/longmemeval_s_cleaned.json `
  --judge-model YOUR_JUDGE_MODEL --base-url YOUR_JUDGE_BASE_URL
```

같은 judge 모델·프롬프트를 모든 baseline에 적용해야 비교할 수 있다.

## 검증 범위

테스트는 API 호출을 대체하거나 초기화까지만 실행하는 오프라인 검증이다.
실제 모델의 답변 품질이나 논문 수치 재현을 검증한 결과는 아니다.

```powershell
uv pip install --python .venv-six/Scripts/python.exe pytest
& .venv-six/Scripts/python.exe -m pytest tests/test_longmemeval_loader.py tests/test_benchmark_adapters.py tests/test_six_baseline_runtime.py -q
```
