"""Initialize one baseline in isolated scratch state, without model API calls."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    entries = [item for item in plan["methods"] if item["method"] == args.method]
    if len(entries) != 1:
        raise ValueError("Unknown method in plan")
    item = entries[0]
    os.environ.update(OPENAI_API_KEY="local-preflight", OPENAI_BASE_URL="http://127.0.0.1:18082/v1",
                      EMBEDDING_BASE_URL="http://127.0.0.1:18083/v1", CUDA_VISIBLE_DEVICES="",
                      OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", BASELINE_STRICT_COMPARISON="1",
                      HF_HOME=plan["hf_home"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      MEM0_TELEMETRY="false", ANONYMIZED_TELEMETRY="false",
                      LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false")

    def forbidden_request(*unused_args, **unused_kwargs):
        raise RuntimeError("Preflight initialization unexpectedly attempted an HTTP model request")

    async def forbidden_async_request(*unused_args, **unused_kwargs):
        raise RuntimeError("Preflight initialization unexpectedly attempted an async HTTP model request")

    httpx.Client.send = forbidden_request
    httpx.AsyncClient.send = forbidden_async_request
    from utils.agent import AgentWrapper

    config = yaml.safe_load(Path(item["agent_config"]).read_text())
    dataset = yaml.safe_load(Path(item["dataset_config"]).read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=args.method + "-", dir=args.work_dir))
    config["artifact_root"] = str(scratch)
    config["output_dir"] = str(scratch / "output")
    state = scratch / "agents" / "context"
    state.mkdir(parents=True)
    agent = AgentWrapper(config, dataset, str(state))
    try:
        token_ids = agent.tokenizer.encode("LoCoMo preflight tokenizer check")
        if not token_ids:
            raise RuntimeError("Qwen tokenizer returned no tokens")
        from transformers import PreTrainedTokenizerBase
        if not isinstance(agent.tokenizer, PreTrainedTokenizerBase):
            raise RuntimeError("Comparison fell back from the native HuggingFace tokenizer")
        print(json.dumps({"method": args.method, "initialized": True, "model_api_requests": 0,
                          "tokenizer_class": type(agent.tokenizer).__name__, "scratch": str(scratch)}), flush=True)
    finally:
        agent.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
