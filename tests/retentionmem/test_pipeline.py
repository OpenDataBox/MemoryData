import ast
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import pickle
import re
import threading
import pytest

from methods.retentionmem.alfworld import load_values, score_trajectories, trajectory_streams
from methods.retentionmem.backends import OpenAIEndpoint, RemoteEmbedder, Tokenizer
from methods.retentionmem.datasets import load_cases, synthetic_streams
from methods.retentionmem.demo import DemoEmbedder
from methods.retentionmem.evaluation import run_qa, paired_comparison, clustered_interval
from methods.retentionmem.features import FeatureExtractor
from methods.retentionmem.generation import CandidateGenerator
from methods.retentionmem.policy import RetentionPolicy
from methods.retentionmem.storage import MemoryStore
from methods.retentionmem.training import collect_transitions, train_from_file, read_jsonl
from methods.retentionmem.types import Event


@pytest.fixture
def fake_api():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            if self.path.endswith("/embeddings"):
                vectors = DemoEmbedder().encode(body["input"])
                response = {"data": [{"index": i, "embedding": v.tolist()} for i, v in enumerate(vectors)], "usage": {"total_tokens": 12}}
            else:
                prompt = body["messages"][-1]["content"]
                if "Compress the following active memory" in prompt:
                    answer = {"summary": "The customer is taking a business trip to Daejeon."}
                elif "Generate at most" in prompt:
                    ids = re.findall(r"\[([^|\n]+) \|", prompt)
                    answer = {"probes": [
                        {"id": "p1", "question": "Current shipping address?", "answer": "Busan", "scope": "current", "unambiguous": True,
                         "sources": [{"event_id": ids[1].strip(), "quote": "The current shipping address is Busan."}]},
                        {"id": "p2", "question": "Business trip city?", "answer": "Daejeon", "scope": "atemporal", "unambiguous": True,
                         "sources": [{"event_id": ids[2].strip(), "quote": "Business trip to Daejeon."}]}]}
                elif prompt.startswith("Answer each diagnostic"):
                    data = json.loads(prompt[prompt.index('{"memory"'):])
                    answer = {"answers": [{"id": q["id"], "answer": ("Busan" if "Busan" in data["memory"] else "UNKNOWN") if q["id"] == "p1" else "Daejeon"} for q in data["questions"]]}
                elif prompt.startswith("Validate diagnostics"):
                    data = json.loads(prompt[prompt.index('{"source_prefix"'):])
                    answer = {"judgments": [{"id": d["id"], "valid": True, "unambiguous": True,
                               "before_correct": d["before_answer"] == d["reference_answer"],
                               "after_correct": d["after_answer"] == d["reference_answer"]} for d in data["diagnostics"]]}
                elif prompt.startswith("Judge semantic"):
                    data = json.loads(prompt[prompt.index('{"question"'):])
                    answer = {"correct": data["prediction"] == data["reference"]}
                elif prompt.startswith("Memory:"):
                    answer = "Busan" if "Busan" in prompt.split("Question:")[0] else "Unknown"
                else: raise AssertionError(prompt[:120])
                content = json.dumps(answer) if isinstance(answer, dict) else answer
                response = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 60, "completion_tokens": 12}}
            payload = json.dumps(response).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests
    server.shutdown(); server.server_close(); thread.join()


def events(group):
    filler = " This parcel contains household supplies."*30
    return [Event(group+":a", "Old shipping address Seoul. "+group+filler, "2025-03-01"),
            Event(group+":b", "The current shipping address is Busan."+filler, "2025-06-01"),
            Event(group+":c", "Business trip to Daejeon."+filler, "2025-07-01")]


def test_paired_collection_real_training_and_qa_resume(tmp_path, fake_api):
    url, calls = fake_api
    endpoint = OpenAIEndpoint(url, "test-model", retries=0)
    store = MemoryStore(tmp_path/"collection.sqlite")
    tokenizer = Tokenizer()
    embedder = RemoteEmbedder(endpoint, store)
    records = [{"group_id": f"g{i}", "origin": "synthetic", "split": "train" if i < 4 else "validation",
                "events": [e.to_dict() for e in events(f"g{i}")]} for i in range(8)]
    labels = tmp_path/"labels.jsonl"
    try:
        report = collect_transitions(records, CandidateGenerator(endpoint, store), endpoint, endpoint, embedder, tokenizer,
                                     store, labels, test_artifact=True)
        assert report["new_candidates"] == 8
        before = len(calls)
        assert collect_transitions(records, CandidateGenerator(endpoint, store), endpoint, endpoint, embedder, tokenizer,
                                   store, labels, resume=True, test_artifact=True)["new_candidates"] == 0
        assert len(calls) == before
        outcomes = {e["regression"] for r in read_jsonl(labels) for e in r["examples"] if e["eligible"]}
        assert outcomes == {0, 1}
        report = train_from_file(labels, tmp_path/"policies", seeds=[0], epochs=12, min_accepted=1)
        assert report[0]["validation_candidates"] == 8
        model = RetentionPolicy.load(tmp_path/"policies"/"policy_seed0.npz", FeatureExtractor(embedder, tokenizer).fingerprint, True)
        assert model.metadata["test_artifact"] is True
    finally: store.close()
    dataset = [{"sample_id": "c1", "conversation": {"session_1_date_time": "2025-07-01", "session_1": [
        {"dia_id": e.id, "speaker": e.actor, "text": e.text} for e in events("case")]},
        "qa": [{"question": "Current shipping address?", "answer": "Busan", "category": 4, "evidence": ["case:b"]}]}]
    path = tmp_path/"qa.json"; path.write_text(json.dumps(dataset))
    section = {"base_url": url, "model": "test-model"}
    config = {"mode": "heuristic_residual", "llm": section, "embedding": section, "judge": section, "query_budget": 128, "test_artifact": True}
    result = run_qa(config, path, "locomo", tmp_path/"run")
    assert result["questions"] == 1 and result["judge_accuracy"]["mean"] == 1
    assert result["cost_accounting"]["llm_input_tokens"] > 0
    before = len(calls)
    run_qa(config, path, "locomo", tmp_path/"run", resume=True)
    assert len(calls) == before
    # The online decision path never sends paired before/after diagnostic answers.
    online = calls[-5:]
    assert not any("Answer each diagnostic" in str(body) for route, body in online)


def test_longmemeval_loader_keeps_assistant_and_original_date(tmp_path):
    item = {"question_id": "q_abs", "question_type": "knowledge-update", "question": "Future question?", "answer": "Unknown",
            "question_date": "2023-12-25", "haystack_sessions": [[{"role": "assistant", "content": "Some past observation"}]],
            "haystack_session_ids": ["s"], "haystack_dates": ["2023-01-01"], "answer_session_ids": []}
    path = tmp_path/"long.json"; path.write_text(json.dumps([item]))
    case = list(load_cases(path, "longmemeval"))[0]
    assert case.events[0].actor == "assistant" and case.events[0].timestamp == "2023-01-01"
    assert case.questions[0].date == "2023-12-25" and case.questions[0].category == "abstention"
    assert "Future question" not in str(case.events)
    from evaluation.longmemeval.longmemeval_judge import LONGMEMEVAL_KNOWLEDGE_UPDATE_PROMPT
    assert "{question}" in LONGMEMEVAL_KNOWLEDGE_UPDATE_PROMPT


def test_clustered_pairing_and_duplicate_guard():
    a = [{"question_id": "q1", "group_id": "c1", "judge_correct": False}, {"question_id": "q2", "group_id": "c2", "judge_correct": True}]
    b = [{**r, "judge_correct": True} for r in a]
    result = paired_comparison(a, b)
    assert result["mean"] == .5 and result["clusters"] == 2
    with pytest.raises(ValueError): paired_comparison(a, b[:1])
    with pytest.raises(ValueError, match="Duplicate"): paired_comparison(a+a, b)
    assert clustered_interval(a[:1], "judge_correct")["ci95"] is None


def test_alfworld_values_and_environment_success(tmp_path):
    p = tmp_path/"train_values.pkl"; p.write_bytes(pickle.dumps(["before action", "failure condition"]))
    assert load_values(p) == ["before action", "failure condition"]
    p.write_bytes(pickle.dumps(ValueError("no executable pickle types")))
    with pytest.raises(ValueError): load_values(p)
    path = tmp_path/"traj_eval_out_of_distribution.json"
    path.write_text(json.dumps([{"gamefile": "env1", "won": True, "steps": [{"curr_action": "go north"}], "round_idx": 0},
                                {"gamefile": "env2", "won": False, "steps": [{"curr_action": "go south"}], "round_idx": 0}]))
    ids = tmp_path/"ids.json"; ids.write_text(json.dumps(["env1", "env2"]))
    report = score_trajectories(path, ids, tmp_path/"score.json")
    assert report["success_rate"]["mean"] == .5 and report["success_rate"]["clusters"] == 2
    ids.write_text(json.dumps(["env1", "env2", "env3"]))
    with pytest.raises(ValueError, match="Incomplete"): score_trajectories(path, ids, tmp_path/"score.json")


def test_memorydata_adapter_dispatch_without_heavy_imports():
    # Exercise the actual wrapper handler in isolation; the full benchmark runner
    # imports GPU dependencies not required by RetentionMem itself.
    tree = ast.parse(Path("utils/agent.py").read_text(encoding="utf-8"))
    wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AgentWrapper")
    handler = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == "_handle_memory_agent")
    module = ast.Module(body=[handler], type_ignores=[])
    namespace = {}; exec(compile(module, "utils/agent.py", "exec"), namespace)
    class Wrapper:
        def _is_agent_type(self, kind): return kind == "retentionmem"
        def _handle_retentionmem_agent(self, *args): return args
    assert namespace["_handle_memory_agent"](Wrapper(), "message", False, "q", "c") == ("message", False, "q", "c")


def test_adamem_export_keeps_index_and_original_clone(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from methods.retentionmem.alfworld import prepare_adamem
    repo = tmp_path/"upstream"
    folder = repo/"retrieval_data"/"alfworld"/"Qwen_test"
    folder.mkdir(parents=True)
    (folder/"train_hnsw_correct_only.index").write_bytes(b"unchanged-index-bytes")
    (folder/"train_hnsw_meta_correct_only.json").write_text('{"dim":64}')
    (folder/"train_values_correct_only.pkl").write_bytes(pickle.dumps(["raw0", "raw1"]))
    runner = repo/"examples"/"prompt_agent"/"gpt4o_alfworld.py"
    runner.parent.mkdir(parents=True)
    anchor = '                        gamefile = infos[i].get("extra.gamefile", "")'
    runner.write_text(anchor)
    monkeypatch.setattr("methods.retentionmem.alfworld.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="test-revision\n"))
    output = tmp_path/"prepared"
    report = prepare_adamem({"mode": "raw"}, repo, "Qwen/test", output)
    assert report["changed_values"] == 0 and report["environment_run_completed"] is False
    assert report["source_sha256"]["train_hnsw_correct_only.index"] == report["output_sha256"]["train_hnsw_correct_only.index"]
    assert runner.read_text() == anchor
    assert 'batch_trajs[i]["gamefile"]' in (output/"examples"/"prompt_agent"/runner.name).read_text()
    assert load_values(output/"retrieval_data"/"alfworld"/"Qwen_test"/"train_values_correct_only.pkl") == ["raw0", "raw1"]
