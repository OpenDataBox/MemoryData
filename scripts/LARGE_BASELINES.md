# 큰 모델로 여섯 baseline 실행

기존 여섯 방법과 LoCoMo/LongMemEval-S 실행기를 재사용한다.
채팅은 Qwen3.5 0.8B·4B·9B·27B, 임베딩은 Qwen3-Embedding 0.6B·4B·8B를 선택한다.
기본값은 채팅 9B + 임베딩 0.6B다. 채팅 크기를 비교할 때 임베딩을 고정한다.

## 준비

Linux/CUDA 서버의 저장소 루트에서 실행한다. 새 벤치마크 환경은 다음처럼 준비한다.

```bash
python3.12 -m venv .venv-six
.venv-six/bin/python -m pip install -r requirements-six-baselines.txt
.venv-six/bin/python scripts/download_memory_benchmarks.py
```

서버용 환경에는 Qwen3.5 및 아래 옵션을 지원하는 vLLM을 설치한다.
[공식 서버 가이드](https://recipes.vllm.ai/Qwen/Qwen3.5-27B)를 따른다.
스크립트는 설치를 수행하지 않는다. 서버는 포그라운드에서 실행되며 Ctrl+C로 종료한다.

## 9B 실행: 별도 터미널 세 개

```bash
# 터미널 1: 채팅 서버
CUDA_VISIBLE_DEVICES=0 CHAT_SIZE=9B \
  bash scripts/run_large_baselines.sh serve-chat

# 터미널 2: 임베딩 서버. GPU 1이 실제로 있는 경우의 예시
CUDA_VISIBLE_DEVICES=1 EMBEDDING_SIZE=0.6B \
  bash scripts/run_large_baselines.sh serve-embedding

# 터미널 3: 두 서버의 시작 완료 로그를 확인한 뒤 실행
CHAT_SIZE=9B bash scripts/run_large_baselines.sh --dry-run
CHAT_SIZE=9B bash scripts/run_large_baselines.sh

# 전체 평가
CHAT_SIZE=9B bash scripts/run_large_baselines.sh --full
```

기본 실행은 각 방법·데이터셋에서 이력 1개와 질문 1개를 처리한다.
`--methods mem0 e_mem`, `--datasets locomo`, `--max-samples 3`,
`--max-queries 10`도 전달할 수 있다. `--dry-run`은 서버 연결 없이 설정만 생성한다.

## 4B / 27B 및 다중 GPU

같은 포트의 이전 채팅 서버를 종료하고 새 서버를 시작한다.
서버와 평가 터미널의 CHAT_SIZE를 동일하게 설정한다.

```bash
# 4B 서버 / 평가 (별도 터미널)
CUDA_VISIBLE_DEVICES=0 CHAT_SIZE=4B bash scripts/run_large_baselines.sh serve-chat
CHAT_SIZE=4B bash scripts/run_large_baselines.sh --full

# 27B 서버: GPU 0,1에 분산. 임베딩은 다른 GPU 또는 다른 머신에서 실행
CUDA_VISIBLE_DEVICES=0,1 TP=2 CHAT_SIZE=27B \
  bash scripts/run_large_baselines.sh serve-chat
CHAT_SIZE=27B bash scripts/run_large_baselines.sh --full
```

GPU 개수만으로 실행 가능 여부가 결정되지는 않는다. 가중치 외에도 KV 캐시와 런타임 메모리가 필요하다.
서버별 기본 GPU 메모리 사용 비율은 0.9다. 같은 GPU에 두 서버를 올리면 각각
`GPU_MEMORY_UTILIZATION`을 낮추고 두 모델이 실제로 들어가는지 확인한다.
`MAX_MODEL_LEN` 기본값은 65536이며, 낮추면 기존 실험의 긴 입력이 거절될 수 있다.
LangMem에 필요한 tool calling을 활성화하고 thinking은 비활성화한다.

## 임베딩 확대

[공식 모델 카드](https://huggingface.co/Qwen/Qwen3-Embedding-8B)의 기본 차원을 적용한다:
0.6B → 1024, 4B → 2560, 8B → 4096.

```bash
# 기존 임베딩 서버를 종료한 뒤 터미널 2에서 실행
CUDA_VISIBLE_DEVICES=1 EMBEDDING_SIZE=4B \
  bash scripts/run_large_baselines.sh serve-embedding

# 터미널 3에서 실행. 8B는 두 명령 모두 EMBEDDING_SIZE=8B로 변경
CHAT_SIZE=9B EMBEDDING_SIZE=4B bash scripts/run_large_baselines.sh --full
```

## 기존 원격 서버 사용

MODEL과 EMBEDDING_MODEL은 각 서버의 /v1/models 이름과 일치해야 한다.
EMBEDDING_DIM은 실제 출력 차원이다. 다른 모델 계열은 해당 모델 설정으로 서버를 별도 실행한다.

```bash
MODEL=my-large-chat \
EMBEDDING_MODEL=my-embedding EMBEDDING_DIM=4096 \
OPENAI_BASE_URL=http://gpu-server:8000/v1 \
EMBEDDING_BASE_URL=http://gpu-server:8001/v1 \
  bash scripts/run_large_baselines.sh --full
```

인증이 필요하면 두 API에서 사용할 OPENAI_API_KEY를 설정한다.
로컬 서버 기본 바인딩은 127.0.0.1이다. 원격 예시는 외부 접근이 구성된 서버를 전제로 한다.
서버의 체크포인트는 LLM_MODEL / EMBED_MODEL로 교체할 수 있다.
모든 환경변수는 `bash scripts/run_large_baselines.sh --help`로 확인한다.

결과·메모리 상태는 `results/large_baselines/<chat>__<embedding>__d<dim>/<smoke|full>/`로 분리한다.
동일 조합은 기존 상태를 재사용할 수 있으므로 독립 반복 실험에는 새 ARTIFACT_ROOT를 지정한다.
컨텍스트·양자화·임베딩 설정을 바꾸는 실험도 새 경로를 사용한다.

```bash
CHAT_SIZE=27B ARTIFACT_ROOT="$PWD/results/qwen35_27b_repeat2" \
  bash scripts/run_large_baselines.sh --full
```

데이터·평가 상세: [SIX_BASELINES.md](SIX_BASELINES.md).