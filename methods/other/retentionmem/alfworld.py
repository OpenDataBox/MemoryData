"""AdaMEM value-only ablation and ALFWorld trajectory protocols.

The upstream retrieval keys, HNSW index, action loop and strategy prompts are retained.
All transformations finish on TRAIN experiences before unseen tasks start.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import numpy as np

from .backends import Tokenizer
from .config import build_runtime
from .evaluation import clustered_interval
from .training import append_jsonl, validate_training_streams
from .types import Event


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StringListUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("AdaMEM values must be a plain list of strings")


def load_values(path):
    result = StringListUnpickler(io.BytesIO(Path(path).read_bytes())).load()
    if not isinstance(result, list) or not result or not all(isinstance(v, str) for v in result):
        raise ValueError("Expected a nonempty AdaMEM list[str] value store")
    return result


def trajectory_streams(path, output, seed=42):
    """Split by initial task state before segmenting. Repeated rollouts stay together."""
    path = Path(path)
    if "train" not in path.name or "eval" in path.name:
        raise ValueError("Use an AdaMEM traj_train*.json export from the official train environment")
    if Path(output).exists():
        raise FileExistsError(output)
    rows = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for trajectory in rows:
        steps = [s for s in trajectory["steps"] if s.get("curr_prompt") not in {None, "None", ""} and s.get("curr_action") not in {None, "None", ""}]
        if not steps: continue
        identity = trajectory.get("gamefile") or steps[0]["curr_prompt"]
        group = hashlib.sha256(str(identity).encode()).hexdigest()
        # Same environment source across all its trajectory variants belongs to one split.
        split = "validation" if int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
        rollout = hashlib.sha256(json.dumps(steps, sort_keys=True).encode()).hexdigest()[:16]
        for start in range(0, len(steps), 8):
            events = [Event(f"{rollout}:{i}", f"Observed state and admissible actions:\n{s['curr_prompt']}\nExecuted action:\n{s['curr_action']}",
                            actor="agent", kind="trajectory", session=group).to_dict()
                      for i, s in enumerate(steps[start:start+8], start)]
            records.append({"group_id": "alfworld:"+group, "split": split, "origin": "alfworld_train", "events": events})
    validate_training_streams(records)
    for record in records: append_jsonl(output, record)
    return {"regions": len(records), "groups": len({r["group_id"] for r in records}),
            "source_sha256": digest(path), "split_counts": {s: sum(r["split"] == s for r in records) for s in ("train", "validation")},
            "note": "Filename is a guard, not proof of split provenance; retain the source rollout configuration."}


def prepare_adamem(config, repo, memory_model, output, subset="correct_only"):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if repo == output or repo in output.parents:
        raise ValueError("Prepare an isolated run directory outside the original AdaMEM clone")
    suffix = "" if subset == "all" else "_"+subset
    relative = Path("retrieval_data")/"alfworld"/memory_model.replace("/", "_")
    files = [f"train_hnsw{suffix}.index", f"train_hnsw_meta{suffix}.json", f"train_values{suffix}.pkl"]
    source = repo/relative
    for name in files:
        if not (source/name).is_file(): raise FileNotFoundError(source/name)
    values = load_values(source/files[2])
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"config": config, "upstream_revision": revision, "mode": config["mode"], "memory_model": memory_model,
                "subset": subset, "source_sha256": {name: digest(source/name) for name in files},
                "entries": len(values), "evaluation_split": "eval_out_of_distribution",
                "intervention": "trajectory value text only; index, keys, rank and agent prompts unchanged"}
    if output.exists():
        previous = output/"retentionmem_manifest.json"
        if not previous.exists() or json.loads(previous.read_text()) != manifest:
            raise ValueError("Existing AdaMEM run directory has a different or missing manifest")
    else:
        # Preserve relative imports and environment configs used by the upstream runner.
        shutil.copytree(repo, output, ignore=shutil.ignore_patterns(".git", "__pycache__", "logs", "retrieval_data", ".venv", ".cache"))
        (output/"retentionmem_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    destination = output/relative
    destination.mkdir(parents=True, exist_ok=True)
    for name in files[:2]: shutil.copy2(source/name, destination/name)
    rewritten = []
    for i, original in enumerate(values):
        if config["mode"] in {"raw", "keep"}:
            rewritten.append(original)
            continue
        memory = build_runtime(config, output/"retentionmem_stores"/str(i))
        try:
            final = memory.store.meta("export_text")
            if final is None:
                memory.observe(Event(f"train-value:{i}", original, actor="agent", kind="trajectory"))
                decisions = memory.finalize()
                # An unchanged value stays byte-identical to the original AdaMEM input.
                audits = memory.store.audits()
                changed = any(a["action"] != "keep" for a in audits)
                final = "\n\n".join(u.text for u in memory.store.units()) if changed else original
                memory.store.set_meta("export_text", final)
            rewritten.append(final)
            (output/"retentionmem_stores"/str(i)/"stats.json").write_text(json.dumps(memory.stats(), indent=2), encoding="utf-8")
        finally: memory.store.close()
    temporary = destination/(files[2]+".tmp")
    with temporary.open("wb") as stream: pickle.dump(rewritten, stream, protocol=4)
    temporary.replace(destination/files[2])
    # Save environment identities for clustering; this only adds a log field.
    runner = output/"examples"/"prompt_agent"/"gpt4o_alfworld.py"
    text = runner.read_text(encoding="utf-8")
    anchor = '                        gamefile = infos[i].get("extra.gamefile", "")'
    addition = '\n                        batch_trajs[i]["gamefile"] = gamefile'
    if addition.strip() not in text:
        if text.count(anchor) != 1: raise ValueError("Upstream trajectory logger changed; inspect before adding environment IDs")
        runner.write_text(text.replace(anchor, anchor+addition), encoding="utf-8")
    tokenizer = Tokenizer(config.get("tokenizer_encoding", "cl100k_base"))
    report = {**manifest, "output_sha256": {name: digest(destination/name) for name in files},
              "original_value_tokens": sum(map(tokenizer.count, values)), "active_value_tokens": sum(map(tokenizer.count, rewritten)),
              "changed_values": sum(a != b for a, b in zip(values, rewritten)), "environment_run_completed": False,
              "logging_change": "save gamefile for environment-level confidence intervals"}
    (output/"retentionmem_export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def score_trajectories(path, episode_ids_path, output, mode="adamem-low"):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = json.loads(Path(episode_ids_path).read_text(encoding="utf-8"))
    if not isinstance(expected, list) or len(expected) != len(set(expected)) or not expected:
        raise ValueError("episode-ids must be the unique official unseen environment list")
    if "eval_out_of_distribution" not in Path(path).name:
        raise ValueError("Main action evaluation requires the official unseen split")
    seen = set()
    scored = []
    fields = {
        "adamem-low": (["refresh_prompt", "strategy_prompt", "action_prompt"], ["refresh_response", "strategy_response", "final_response"]),
        "adamem-high": (["initial_prompt", "strategy_prompt", "action_prompt"], ["initial_response", "strategy_response", "final_response"]),
        "adamem-max": (["strategy_prompt", "action_prompt"], ["strategy_response", "final_response"]),
    }
    if mode not in fields: raise ValueError("Supported scoring flows: adamem-low/high/max")
    tokenizer = Tokenizer()
    for row in rows:
        group = row.get("gamefile")
        if not group or group not in expected: raise ValueError("Missing/unknown environment identity; use the prepared logger and exact unseen IDs")
        if type(row.get("won")) is not bool: raise ValueError("Unfinished episode: success must come from the environment")
        identity = (group, row.get("round_idx", 0))
        if identity in seen: raise ValueError("Duplicate environment/round")
        seen.add(identity)
        steps = [s for s in row["steps"] if s.get("curr_action") not in {None, "None", ""}]
        inputs, outputs = fields[mode]
        count = lambda keys: sum(tokenizer.count(s[k]) for s in steps for k in keys if isinstance(s.get(k), str))
        scored.append({"question_id": f"{group}:{row.get('round_idx', 0)}", "group_id": group, "success": row["won"],
                       "actions": len(steps), "input_tokens_estimated": count(inputs), "output_tokens_estimated": count(outputs)})
    for round_id in {r.get("round_idx", 0) for r in rows}:
        if {g for g, r in seen if r == round_id} != set(expected): raise ValueError("Incomplete unseen environment coverage")
    if not scored: raise ValueError("No episodes")
    report = {"episodes": len(scored), "success_rate": clustered_interval(scored, "success"),
              "actions": clustered_interval(scored, "actions"), "source_sha256": digest(path),
              "expected_environments_sha256": digest(episode_ids_path), "rows": scored,
              "prerequisite_violation_rate": None, "cost_note": "Token counts are cl100k estimates from logs, not server usage. Add offline export and policy training costs.",
              "success_source": "official environment won; action string differences are not failures"}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "rows"}
