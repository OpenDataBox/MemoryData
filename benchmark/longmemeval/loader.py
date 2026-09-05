"""Load official LongMemEval JSON without exposing answer annotations to memory."""
import json
from pathlib import Path


def _load_json_records(path, limit=None):
    """Load a JSON array, streaming only the requested prefix when limited."""
    path = Path(path)
    if limit is None:
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle)

    limit = int(limit)
    if limit <= 0:
        return []

    decoder = json.JSONDecoder()
    records = []
    buffer = ""
    array_started = False

    with path.open(encoding="utf-8-sig") as handle:
        while len(records) < limit:
            chunk = handle.read(64 * 1024)
            at_eof = not chunk
            buffer += chunk

            while True:
                buffer = buffer.lstrip()
                if not array_started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"Expected a JSON array in {path}")
                    buffer = buffer[1:]
                    array_started = True
                    continue

                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    return records
                if not buffer:
                    break

                try:
                    record, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if at_eof:
                        raise
                    break

                records.append(record)
                buffer = buffer[end:]
                if len(records) >= limit:
                    return records

            if at_eof:
                break

    if not array_started:
        raise ValueError(f"Expected a JSON array in {path}")
    return records


def load_longmemeval_eval_data(dataset_config):
    path = Path(dataset_config.get("test_files") or
                "datasets/LongMemEval/longmemeval_s_cleaned.json")
    limit = dataset_config.get("max_test_samples")
    records = _load_json_records(path, int(limit) if limit else None)
    samples = []
    for record in records:
        qid = str(record["question_id"])
        sessions = record["haystack_sessions"]
        dates = record["haystack_dates"]
        ids = record["haystack_session_ids"]
        if not (len(sessions) == len(dates) == len(ids)):
            raise ValueError(f"{qid}: session IDs, dates and histories must align")
        # Keep complete turns and their original session time. Labels such as
        # has_answer and answer_session_ids never enter these chunks.
        chunks = []
        chunk_size = int(dataset_config.get("chunk_size", 4096))
        for sid, date, turns in zip(ids, dates, sessions):
            header = f"Session {sid} ({date})\n"
            lines = []
            size = 0
            for turn in turns:
                line = f'{turn["role"]}: {turn["content"]}'
                if lines and size + len(line) > chunk_size:
                    chunks.append(header + "\n".join(lines))
                    lines, size = [], 0
                lines.append(line)
                size += len(line)
            if lines:
                chunks.append(header + "\n".join(lines))
        context_chunk_limit = dataset_config.get("max_context_chunks")
        if context_chunk_limit:
            chunks = chunks[:int(context_chunk_limit)]
        metadata = {
            "dataset": "longmemeval", "question_id": qid, "qa_pair_id": qid,
            "question_type": record["question_type"],
            "question_date": record["question_date"],
            "question": record["question"],
            "answer_session_ids": record.get("answer_session_ids", []),
        }
        samples.append({
            "source": "longmemeval_s_official", "sample_id": qid,
            "context_chunks": [{"text": chunk} for chunk in chunks],
            "context_length": sum(map(len, chunks)),
            "questions": [f'Question date: {record["question_date"]}\nQuestion: {record["question"]}'], "answers": [record["answer"]],
            "qa_pair_ids": [qid], "question_ids": [qid],
            "question_types": [record["question_type"]],
            "question_dates": [record["question_date"]],
            "eval_metadata": [metadata],
        })
    return {"data": samples}

