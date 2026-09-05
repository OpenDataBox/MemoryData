"""Counterfactual data collection with source-group splits and held-out calibration."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .features import FeatureExtractor
from .generation import SYSTEM, render_events, render_span
from .policy import train_policy
from .types import Event, MemoryUnit


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+"\n")


def source_digest(events):
    # IDs are deliberately omitted: renaming a duplicated source cannot hide leakage.
    data = [(e.text, e.timestamp, e.actor, e.kind) for e in events]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def validate_training_streams(records):
    groups, sources = {}, {}
    for record in records:
        if set(record)-{"group_id", "split", "origin", "events"}:
            raise ValueError("Training streams may contain only group_id, split, origin and events; no benchmark QA metadata")
        if record.get("origin") not in {"synthetic", "alfworld_train"}:
            raise ValueError("Only synthetic or official ALFWorld train sources may train this policy")
        split = record["split"]
        if split not in {"train", "validation"}:
            raise ValueError("Evaluation/test streams cannot enter policy collection")
        events = [Event(**e) for e in record["events"]]
        if not events or len({e.id for e in events}) != len(events):
            raise ValueError("A training region needs unique, nonempty source records")
        group, digest = str(record["group_id"]), source_digest(events)
        for key, table in ((group, groups), (digest, sources)):
            if key in table and table[key] != split:
                raise ValueError(f"Source-group leakage across training/calibration splits: {group}")
            table[key] = split
    return groups


def answer_probes(reader, probes, memory_text, ledger, stage):
    prompt = "Answer each diagnostic using ONLY the supplied memory. If unsupported, answer UNKNOWN. Return JSON {answers:[{id,answer}]}.\n"
    prompt += json.dumps({"memory": memory_text, "questions": [{"id": p.id, "question": p.question} for p in probes]}, ensure_ascii=False)
    result = reader.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                         json_mode=True, ledger=ledger, stage=stage, max_tokens=1536)
    answers = result.get("answers", [])
    expected = {p.id for p in probes}
    if len(answers) != len(expected) or {str(a.get("id")) for a in answers} != expected or not all(isinstance(a.get("answer"), str) for a in answers):
        raise ValueError("Reader did not return exactly one answer per probe; parser failures are not regression labels")
    return {str(a["id"]): a["answer"] for a in answers}


def judge_transition(judge, probes, events, before_answers, after_answers, ledger):
    prompt = """Validate diagnostics independently against the COMPLETE source prefix. Unknown or ambiguous reference
answers are invalid. The latest explicit update controls CURRENT questions; historical questions retain their time scope.
Judge semantic correctness of the before/after answers, not lexical agreement and not whether their strings changed.
For action questions judge supported prerequisites and failure conditions, not arbitrary action wording.
Return JSON {judgments:[{id,valid,unambiguous,before_correct,after_correct}]}, using booleans only.
Source quotes establish provenance, not an automatic proof that a reference answer is correct.\n"""
    prompt += json.dumps({"source_prefix": render_events(events), "diagnostics": [
        {"id": p.id, "question": p.question, "reference_answer": p.answer, "scope": p.scope,
         "before_answer": before_answers.get(p.id, "UNKNOWN"), "after_answer": after_answers.get(p.id, "UNKNOWN")}
        for p in probes]}, ensure_ascii=False)
    result = judge.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                        json_mode=True, ledger=ledger, stage="transition_judge", max_tokens=1536)
    judgments = result.get("judgments", [])
    if len(judgments) != len(probes) or {str(j.get("id")) for j in judgments} != {p.id for p in probes}:
        raise ValueError("Judge response is missing or duplicating probes")
    if not all(all(type(j.get(k)) is bool for k in ("valid", "unambiguous", "before_correct", "after_correct")) for j in judgments):
        raise ValueError("Judge correctness and validity fields must be booleans")
    return {str(j["id"]): j for j in judgments}


def collect_transitions(records, generator, reader, judge, embedder, tokenizer, store, output,
                        max_candidates=None, resume=False, test_artifact=False):
    records = list(records)
    validate_training_streams(records)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite collected labels: {output}; use --resume")
    old = list(read_jsonl(output)) if output.exists() else []
    done = {r["candidate_id"] for r in old}
    features = FeatureExtractor(embedder, tokenizer)
    manifest = {"features": features.fingerprint, "reader": reader.model, "judge": judge.model,
                "compressor": generator.endpoint.model, "style": generator.style, "version": 2,
                "test_artifact": test_artifact}
    for row in old:
        if row["manifest"] != manifest:
            raise ValueError("Cannot resume with different reader/judge/compressor/features")
    count = 0
    for record in records:
        if max_candidates is not None and count >= max_candidates:
            break
        events = [Event(**e) for e in record["events"]]
        if len(events) > 8:
            raise ValueError("Collection region exceeds the 8-record bound; segment train streams first")
        digest = source_digest(events)
        base_id = hashlib.sha256((record["group_id"]+digest+generator.style).encode()).hexdigest()
        if base_id in done:
            continue
        start = time.perf_counter()
        units = [MemoryUnit(e.id, f"[{e.timestamp} | {e.actor}]\n{e.text}", (e.id,), e.source_ids or (e.id,)) for e in events]
        before_text = "\n\n".join(u.text for u in units)
        before_tokens = tokenizer.count(before_text)
        candidate = generator.generate(units, events, max(16, before_tokens//4))
        event_map = {e.id: e for e in events}
        if not candidate.probes:
            append_jsonl(output, {"candidate_id": base_id, "group_id": record["group_id"], "source_digest": digest,
                                 "split": record["split"], "origin": record["origin"], "manifest": manifest,
                                 "status": "no_valid_probes", "examples": []})
            count += 1
            continue
        before_answers = answer_probes(reader, candidate.probes, before_text, store, "reader_before")
        # A repaired candidate is labelled too: otherwise the predictor never sees residuals during training.
        spans = list(dict.fromkeys(s for p in candidate.probes[:max(1, len(candidate.probes)//2)] for s in p.spans))
        repaired = candidate.summary+"\n\nRetained details:\n"+"\n".join(render_span(s, event_map) for s in spans)
        variants = [("summary", candidate.summary), ("residual", repaired)]
        examples = []
        for variant, after_text in variants:
            after_answers = answer_probes(reader, candidate.probes, after_text, store, "reader_after")
            judgments = judge_transition(judge, candidate.probes, events, before_answers, after_answers, store)
            x = features.extract(candidate.probes, event_map, after_text, before_tokens)
            for probe, vector in zip(candidate.probes, x):
                j = judgments.get(probe.id, {})
                valid = j.get("valid") is True and j.get("unambiguous") is True
                typed = isinstance(j.get("before_correct"), bool) and isinstance(j.get("after_correct"), bool)
                eligible = valid and typed and j["before_correct"]
                examples.append({"probe": probe.to_dict(), "variant": variant, "candidate_id": base_id+":"+variant,
                                 "before_answer": before_answers.get(probe.id), "after_answer": after_answers.get(probe.id),
                                 "judgment": j, "eligible": bool(eligible), "features": vector.tolist(),
                                 "regression": int(not j["after_correct"]) if eligible else None,
                                 "before_tokens": before_tokens, "after_tokens": tokenizer.count(after_text)})
        append_jsonl(output, {"candidate_id": base_id, "group_id": record["group_id"], "source_digest": digest,
                             "split": record["split"], "origin": record["origin"], "manifest": manifest,
                             "status": "labelled", "summary": candidate.summary, "events": [e.to_dict() for e in events],
                             "seconds": time.perf_counter()-start, "examples": examples})
        count += 1
    return {"new_candidates": count, "output": str(output), "costs": store.costs(), "manifest": manifest}


def train_from_file(path, output, seeds=(0, 1, 2), epochs=80, max_regression_rate=.05, min_accepted=20):
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError("No collected candidates")
    features = rows[0]["manifest"]["features"]
    partitions = {"train": [], "validation": []}
    groups, sources = {}, {}
    for row in rows:
        if row["manifest"] != rows[0]["manifest"]:
            raise ValueError("Collected candidates have incompatible manifests")
        if row["origin"] not in {"synthetic", "alfworld_train"} or row["split"] not in partitions:
            raise ValueError("Only approved training origins/splits can train or calibrate")
        for key, table in ((row["group_id"], groups), (row["source_digest"], sources)):
            if key in table and table[key] != row["split"]:
                raise ValueError("Training/validation source leakage detected")
            table[key] = row["split"]
        partitions[row["split"]].extend(e for e in row["examples"] if e["eligible"] and e["regression"] in (0, 1))
    if not all(partitions.values()):
        raise ValueError("Both training and validation need eligible paired-reader examples")
    train, val = partitions["train"], partitions["validation"]
    x, y = np.asarray([e["features"] for e in train], np.float32), np.asarray([e["regression"] for e in train])
    vx, vy = np.asarray([e["features"] for e in val], np.float32), np.asarray([e["regression"] for e in val])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for seed in seeds:
        started = time.perf_counter()
        model = train_policy(x, y, vx, vy, [e["candidate_id"] for e in val], fingerprint=features, seed=seed,
                             epochs=epochs, max_regression_rate=max_regression_rate, min_accepted=min_accepted,
                             test_artifact=rows[0]["manifest"].get("test_artifact", False))
        model.metadata.update(source_file_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                              collection_manifest=rows[0]["manifest"], source_group_count=len(groups),
                              training_seconds=time.perf_counter()-started,
                              trainable_parameters=sum(v.size for v in model.p.values()))
        filename = output/f"policy_seed{seed}.npz"
        model.save(filename)
        report.append({"seed": seed, "checkpoint": str(filename), "training_seconds": model.metadata["training_seconds"],
                       "trainable_parameters": model.metadata["trainable_parameters"], **model.metadata["calibration"]})
    (output/"training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
