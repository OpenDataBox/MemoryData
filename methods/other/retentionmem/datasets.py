"""Official QA loaders and separate, deterministic synthetic training streams."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import urllib.request

from .types import Event

DATASETS = {
    "locomo": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "longmemeval": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
}


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    answer: str
    category: str
    date: str = ""
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    id: str
    events: tuple[Event, ...]
    questions: tuple[Question, ...]
    benchmark: str


def download_dataset(name, directory):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/("locomo10.json" if name == "locomo" else "longmemeval_s_cleaned.json")
    if not target.exists():
        temporary = target.with_suffix(".download")
        with urllib.request.urlopen(DATASETS[name], timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024*1024):
                stream.write(chunk)
        with temporary.open(encoding="utf-8") as stream:
            if not isinstance(json.load(stream), list):
                raise ValueError("Downloaded dataset is not a JSON array")
        temporary.replace(target)
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {"name": name, "url": DATASETS[name], "sha256": sha, "path": str(target)}
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_cases(path, benchmark, include_adversarial=False):
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError("Expected the official dataset JSON array")
    if benchmark == "locomo":
        for index, item in enumerate(data):
            sample_id = str(item.get("sample_id", index))
            conversation = item["conversation"]
            events = []
            session_keys = sorted((k for k in conversation if k.startswith("session_") and k[8:].isdigit()), key=lambda k: int(k[8:]))
            for session in session_keys:
                for turn_index, turn in enumerate(conversation[session]):
                    source_id = str(turn.get("dia_id", f"{session}:{turn_index}"))
                    text = str(turn.get("text", ""))
                    caption = str(turn.get("blip_caption", "") or "").strip()
                    if caption:
                        text = f"[Image: {caption}] {text}".strip()
                    if text.strip():
                        events.append(Event(f"{sample_id}:{source_id}", text, str(conversation.get(session+"_date_time", "")),
                                            actor=str(turn.get("speaker", "unknown")), session=session, source_ids=(source_id,)))
            questions = []
            for qindex, question in enumerate(item["qa"]):
                category = str(question["category"])
                if category == "5" and not include_adversarial:
                    continue
                # adversarial_answer is a distractor, not a valid reference answer.
                answer = "No information available" if category == "5" else question.get("answer")
                questions.append(Question(str(question.get("question_id", f"{sample_id}:q{qindex}")), question["question"],
                                          str(answer if answer is not None else "Unknown"), category,
                                          source_ids=tuple(str(s) for s in question.get("evidence", []))))
            yield Case(sample_id, tuple(events), tuple(questions), benchmark)
    elif benchmark == "longmemeval":
        for item in data:
            events = []
            sessions, ids, dates = item["haystack_sessions"], item["haystack_session_ids"], item["haystack_dates"]
            if not len(sessions) == len(ids) == len(dates):
                raise ValueError("LongMemEval session/date/ID lengths do not match")
            for session, session_id, date in zip(sessions, ids, dates):
                for turn_index, turn in enumerate(session):
                    if str(turn.get("content", "")).strip():
                        events.append(Event(f"{session_id}:{turn_index}", str(turn["content"]), str(date),
                                            actor=str(turn["role"]), session=str(session_id), source_ids=(str(session_id),)))
            qid = str(item["question_id"])
            category = "abstention" if qid.endswith("_abs") else item["question_type"]
            question = Question(qid, item["question"], str(item["answer"]), category, str(item.get("question_date", "")),
                                tuple(str(s) for s in item.get("answer_session_ids", [])))
            yield Case(qid, tuple(events), (question,), benchmark)
    else:
        raise ValueError(f"Unsupported QA benchmark {benchmark}")


def synthetic_streams(train_groups=3000, validation_groups=750, seed=42):
    """Source-only synthetic data. No official evaluation QA is sampled or paraphrased."""
    rng = random.Random(seed)
    cities = ["Seoul", "Busan", "Daejeon", "Oslo", "Kyoto", "Lima", "Perth", "Turin"]
    for index in range(train_groups+validation_groups):
        old, current, trip = rng.sample(cities, 3)
        name = f"Participant-{rng.randrange(10**8):08d}"
        order, code = rng.randrange(10000, 99999), rng.randrange(1000, 9999)
        group = f"synthetic-{seed}-{index}"
        texts = [
            f"My name is {name}. My shipping address is in {old}. My order number is {order}.",
            f"Please change my shipping address to {current}. The {old} address is historical and must not receive new orders.",
            f"I am visiting {trip} next week. This is a business trip, not another address change.",
            f"For order {order}, delivery is allowed only after 14:00. Do not leave it outside unless I explicitly confirm.",
            f"The access code is {code}. It is different from the order number. Keep the code private from other users.",
            "The support agent confirmed the address update. The courier has not yet confirmed a delivery date.",
            f"The shipment contains a blue notebook and two pens. The notebook belongs to {name}; the pens are a gift.",
            "The trip does not cancel the delivery-time restriction. I still want a confirmation before unattended delivery.",
        ]
        events = [Event(f"{group}:{i}", text, f"2025-06-{i+1:02d}", actor="user" if i != 5 else "assistant").to_dict()
                  for i, text in enumerate(texts)]
        yield {"group_id": group, "split": "train" if index < train_groups else "validation", "origin": "synthetic", "events": events}
