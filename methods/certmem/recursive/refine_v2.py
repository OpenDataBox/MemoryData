"""Question-blind recursive memory construction with a sealed conversation holdout.

The builder accepts conversation sessions only. Development answers/evidence are
used solely by evaluation/diagnostics. Reader, retriever and read budget are fixed.
LoCoMo's category-specific F1 is the primary score (not date-normalized F1).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import string
import time

READER = (
    "Answer the question using only the provided conversation memory. "
    "Give a concise exact answer: a name, date, number, or short phrase. "
    "For a list question include every supported item, separated by commas. "
    "Use the session dates to interpret relative times, but distinguish the event date "
    "from the date someone talked about it. If unsupported, answer unknown. "
    "No explanation or introductory text."
)
WRITER = (
    "Extract long-term memory from the conversation. Output at most 10 facts, one per line. "
    "Each fact must stand alone, name the person, and preserve dates, numbers, negations, "
    "and conditions. Begin each fact with its source IDs, e.g. [D1:2,D1:3]. "
    "Use only provided IDs and facts. No commentary."
)
EVENT_WRITER = (
    "Construct detailed episodic memory from the supplied dialogue window. Output up to 24 "
    "atomic facts, one per line, each beginning with exact source IDs such as [D1:2,D1:3]. "
    "Resolve I/you/he/she/it using adjacent dialogue and name the correct person or object. "
    "Preserve specific names, titles, locations, reasons, relationships, quantities, preferences, "
    "negations and plans. Keep each event's participants, action, object, time and place together. "
    "Distinguish a plan from a completed event. Retain original relative time expressions and "
    "their session date; never invent an event date. Do not replace older events with newer ones. "
    "Include details in replies that depend on the preceding question. No general commentary."
)
AUDITOR = (
    "Audit existing memory against the ORIGINAL dialogue. Add ONLY useful facts missing from "
    "the memory, at most 16 lines. Look for omitted names/titles, precise event time or place, "
    "reasons, lists, relationships, corrections, negation, and short replies whose meaning depends "
    "on earlier turns. Each new fact must be self-contained, name its subject, and begin with "
    "exact supporting source IDs [D1:2,D1:3]. Do not invent facts or copy a fact already covered. "
    "Original dialogue is authoritative. Output NONE if nothing is missing."
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def lexical(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=100000)
def score_words(text):
    from nltk.stem import PorterStemmer
    stem = PorterStemmer()
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the|and)\b', ' ', text)
    return tuple(stem.stem(w) for w in text.split())


def match(pred, gold):
    p, g = score_words(str(pred)), score_words(str(gold))
    common = sum((Counter(p) & Counter(g)).values())
    return 2 * common / (len(p) + len(g)) if common else 0.0


def f1(pred, gold, category):
    if category == 1:
        ps, gs = str(pred).split(','), str(gold).split(',')
        return sum(max(match(p, g) for p in ps) for g in gs) / len(gs)
    if category == 3:
        gold = str(gold).split(';')[0].strip()
    if category in (2, 3, 4):
        return match(pred, gold)
    raise ValueError('Adversarial/unknown category is outside this pilot')


def generic_f1(pred, gold):
    def norm(x):
        return [w for w in re.sub(r'[^a-z0-9 ]', ' ', str(x).lower()).split()
                if w not in {'a', 'an', 'the'}]
    p, g = norm(pred), norm(gold)
    common = sum((Counter(p) & Counter(g)).values())
    return 2 * common / (len(p) + len(g)) if common else 0.0


def make_manifest(samples, seed=20260907, per_conv=10, holdout_convs=3):
    """Proportional categories within each conversation; selection independent of answers."""
    rng = random.Random(seed)
    cids = sorted(str(s['sample_id']) for s in samples)
    rng.shuffle(cids)
    holdout = set(cids[-holdout_convs:])
    records = []
    for sample in sorted(samples, key=lambda s: str(s['sample_id'])):
        cid = str(sample['sample_id'])
        buckets = defaultdict(list)
        for i, qa in enumerate(sample['qa']):
            if int(qa['category']) in (1, 2, 3, 4):
                buckets[int(qa['category'])].append(i)
        total = sum(map(len, buckets.values()))
        if total < per_conv:
            raise ValueError(f'{cid}: fewer than {per_conv} eligible questions')
        ideal = {c: len(v) * per_conv / total for c, v in buckets.items()}
        counts = {c: int(v) for c, v in ideal.items()}
        for c in sorted(counts, key=lambda c: (-(ideal[c] - counts[c]), c))[:per_conv-sum(counts.values())]:
            counts[c] += 1
        for c in sorted(buckets):
            ids = sorted(buckets[c])
            rng.shuffle(ids)
            for i in ids[:counts[c]]:
                records.append({'id': f'{cid}:{i}', 'conv_id': cid, 'qa_index': i,
                                'category': c, 'split': 'holdout' if cid in holdout else 'dev'})
    return {'seed': seed, 'per_conv': per_conv, 'holdout_convs': sorted(holdout),
            'dataset_sha256': digest(samples), 'records': records}


def session_data(sample):
    """Explicitly strip QA fields before any memory construction."""
    conv = sample['conversation']
    sessions = []
    for key in sorted((k for k in conv if re.fullmatch(r'session_\d+', k)), key=lambda k: int(k.split('_')[1])):
        num = int(key.split('_')[1])
        date = conv.get(f'{key}_date_time', '')
        turns = []
        for i, t in enumerate(conv[key]):
            body = t.get('text', '')
            if t.get('blip_caption'):
                body += ' [image: ' + t['blip_caption'] + ']'
            turns.append({'id': t.get('dia_id', f'D{num}:{i+1}'), 'speaker': t['speaker'], 'body': body,
                          'text': f"[{t.get('dia_id', f'D{num}:{i+1}')}] {t['speaker']}: {body}"})
        sessions.append({'num': num, 'date': date, 'turns': turns})
    return sessions


def windows(sessions, size=18, overlap=2):
    for session in sessions:
        for start in range(0, len(session['turns']), size - overlap):
            yield session, session['turns'][start:start+size]
            if start + size >= len(session['turns']):
                break


def parse_facts(output, turns, session, limit):
    allowed = {t['id'] for t in turns}
    units = []
    for line in output.splitlines():
        line = line.strip().lstrip('-* ').strip()
        if not line or line.upper() == 'NONE':
            continue
        m = re.search(r'\[([^\]]+)\]', line)
        if not m:
            continue
        ids = re.findall(r'D\d+:\d+', m.group(1))
        if not ids or not set(ids) <= allowed:
            continue
        body = line[m.end():].strip(' :-')
        if not body:
            continue
        units.append({'text': f"(Session date: {session['date']}) {body}",
                      'sources': list(dict.fromkeys(ids)), 'session': session['num'], 'kind': 'fact'})
        if len(units) >= limit:
            break
    return units


def dedupe(units):
    seen, kept = set(), []
    for unit in units:
        key = tuple(lexical(unit['text']))
        if key not in seen:
            seen.add(key)
            kept.append(unit)
    return kept


def raw_units(sessions, paired=False):
    units = []
    for s in sessions:
        for i, t in enumerate(s['turns']):
            ts = s['turns'][max(0, i-1):i+1] if paired else [t]
            text = f"(Session date: {s['date']}) " + '\n'.join(x['text'] for x in ts)
            units.append({'text': text, 'sources': [x['id'] for x in ts],
                          'session': s['num'], 'kind': 'dialogue_pair' if paired else 'raw'})
    return units


def quoted_memory(sessions, parent):
    """Bind fact search cues to their verbatim evidence, before any query exists."""
    turns = {t['id']: t for s in sessions for t in s['turns']}
    groups = {}
    for unit in parent or []:
        if unit['kind'] != 'fact':
            continue
        ids = tuple(sorted((sid for sid in unit['sources'] if sid in turns),
                           key=lambda sid: tuple(map(int, re.findall(r'\d+', sid)))))
        if not ids:
            continue
        group = groups.setdefault(ids, {'facts': [], 'session': unit['session']})
        group['facts'].append(unit['text'])
    result = []
    for ids, group in groups.items():
        text = '\n'.join(dict.fromkeys(group['facts']))
        text += '\nVerbatim supporting dialogue:\n' + '\n'.join(turns[sid]['text'] for sid in ids)
        result.append({'text': text, 'sources': list(ids), 'session': group['session'], 'kind': 'quoted_fact'})
    return result or raw_units(sessions, paired=True)


def pack(units, order, ntok, budget):
    """Account for separators using the actual tokenizer; never silently overflow."""
    selected, parts, total = [], [], 0
    for i in order:
        text = units[int(i)]['text']
        candidate = '\n\n'.join(parts + [text])
        cost = ntok(candidate)
        if cost <= budget:
            selected.append(int(i))
            parts.append(text)
            total = cost
    return '\n\n'.join(parts), selected, total


class Runtime:
    def __init__(self, args, environment):
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
        from vllm import LLM
        torch.set_num_threads(8)
        self.args, self.np = args, np
        self.cache = Path(args.out) / 'cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.model_meta = environment['models'][args.model]
        self.embed_meta = environment['models'][args.embed_model]
        self.embed = SentenceTransformer(self.embed_meta['path'], device='cuda')
        self.llm = LLM(model=self.model_meta['path'], dtype='bfloat16', max_model_len=8192,
                       gpu_memory_utilization=0.78, max_num_seqs=24, max_num_batched_tokens=8192,
                       enable_prefix_caching=True, enforce_eager=True, seed=args.seed)
        self.tok = self.llm.get_tokenizer()
        self.ntok = lru_cache(maxsize=100000)(lambda s: len(self.tok.encode(s, add_special_tokens=False)))

    def generate(self, system, users, max_tokens=512):
        from vllm import SamplingParams
        outputs = [None] * len(users)
        missing, positions, paths = [], [], []
        for i, user in enumerate(users):
            key = digest([self.model_meta, self.args.seed, system, user, max_tokens, False])
            path = self.cache / 'generations' / (key + '.json')
            if path.exists():
                outputs[i] = json.loads(path.read_text())['text']
            else:
                messages = [{'role':'system','content':system}, {'role':'user','content':user}]
                prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                if self.ntok(prompt) + max_tokens > 8192:
                    raise ValueError('Prompt exceeds context; refusing silent truncation')
                missing.append(prompt)
                positions.append(i)
                paths.append(path)
        for start in range(0, len(missing), 24):
            batch = missing[start:start+24]
            generated = self.llm.generate(batch, SamplingParams(temperature=0, max_tokens=max_tokens), use_tqdm=False)
            for offset, result in enumerate(generated):
                j = start + offset
                answer = result.outputs[0]
                text = answer.text.strip()
                outputs[positions[j]] = text
                save(paths[j], {'text': text, 'finish_reason': answer.finish_reason,
                                'input_tokens': len(result.prompt_token_ids), 'output_tokens': len(answer.token_ids)})
            print(f'GEN {start+len(batch)}/{len(missing)} max_tokens={max_tokens}', flush=True)
        return outputs

    def encode(self, texts, query=False):
        key = digest([self.embed_meta, texts, query])
        path = self.cache / 'embeddings' / (key + '.npy')
        if path.exists():
            return self.np.load(path)
        # Official Qwen embedding instruction applies to the query side only.
        prepared = ['Instruct: Retrieve relevant conversation memories to answer the question.\nQuery: ' + t for t in texts] if query else texts
        encoded = self.embed.encode(prepared, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.np.save(path, encoded)
        return encoded


def build(rt, sessions, strategy, parent=None):
    """No QA, gold, evidence or development failures are accepted here."""
    if strategy == 'raw_rag':
        return raw_units(sessions)
    if strategy == 'dialogue_residual':
        return dedupe((parent or []) + raw_units(sessions, paired=True))
    if strategy == 'provenance_quotes':
        return quoted_memory(sessions, parent)
    if strategy == 'session10':
        chunks = [(s, s['turns']) for s in sessions]
        sys_prompt, limit, out_tokens = WRITER, 10, 512
    else:
        chunks = list(windows(sessions))
        sys_prompt, limit, out_tokens = (AUDITOR, 16, 700) if strategy == 'audit' else (EVENT_WRITER, 24, 900)
    users = []
    for s, ts in chunks:
        text = f"Session date: {s['date']}\nORIGINAL dialogue:\n" + '\n'.join(t['text'] for t in ts)
        if strategy == 'audit':
            ids = {t['id'] for t in ts}
            existing = [u['text'] for u in (parent or []) if ids.intersection(u['sources'])]
            text += '\n\nExisting memory:\n' + '\n'.join(existing)
        users.append(text + '\n\nMemory facts:')
    outputs = rt.generate(sys_prompt, users, max_tokens=out_tokens)
    units = list(parent or []) if strategy == 'audit' else []
    bad_chunks = 0
    for output, (s, ts) in zip(outputs, chunks):
        parsed = parse_facts(output, ts, s, limit)
        # Preserve source text when an initial extraction produces no usable record.
        if not parsed and strategy != 'audit':
            bad_chunks += 1
            parsed = raw_units([dict(s, turns=ts)])
        units.extend(parsed)
    if bad_chunks:
        print(f'EXTRACTION_FALLBACK {bad_chunks}/{len(chunks)} chunks', flush=True)
    return dedupe(units)


def evaluate(rt, memories, records, samples_by_id, budget, name, outdir):
    """Question-only retrieval; answers and evidence accessed after generation."""
    import numpy as np
    from rank_bm25 import BM25Okapi
    prompts, metadata = [], []
    for cid in sorted({r['conv_id'] for r in records}):
        selected_records = [r for r in records if r['conv_id'] == cid]
        units = memories[cid]
        texts = [u['text'] for u in units]
        dense = rt.encode(texts)
        bm25 = BM25Okapi([lexical(t) or ['_empty'] for t in texts])
        questions = [samples_by_id[cid]['qa'][r['qa_index']]['question'] for r in selected_records]
        queries = rt.encode(questions, query=True)
        for record, question, query in zip(selected_records, questions, queries):
            dense_scores, sparse_scores = dense @ query, np.asarray(bm25.get_scores(lexical(question)))
            def ranks(scores):
                order = np.argsort(-scores, kind='stable')
                ranks = np.empty(len(order), dtype=np.int64)
                ranks[order] = np.arange(len(order))
                return ranks
            fused = 1 / (60 + ranks(dense_scores)) + 1 / (60 + ranks(sparse_scores))
            order = np.argsort(-fused, kind='stable')[:min(len(units), 120)]
            context, hit_ids, tokens = pack(units, order, rt.ntok, budget)
            prompts.append(f'Conversation memory:\n{context}\n\nQuestion: {question}\nAnswer:')
            metadata.append({**record, 'question': question, 'context': context, 'read_tokens': tokens,
                             'source_ids': sorted({s for i in hit_ids for s in units[i]['sources']}),
                             'memory_tokens': sum(rt.ntok(t) for t in texts), 'memory_units': len(units)})
    predictions = rt.generate(READER, prompts, max_tokens=96)
    rows = []
    for metadata_row, prediction in zip(metadata, predictions):
        qa = samples_by_id[metadata_row['conv_id']]['qa'][metadata_row['qa_index']]
        gold = str(qa['answer'])
        evidence = [str(e) for e in qa.get('evidence', []) if re.fullmatch(r'D\d+:\d+', str(e))]
        hit = len(set(evidence) & set(metadata_row['source_ids']))
        rows.append({**metadata_row, 'candidate': name, 'prediction': prediction, 'gold': gold,
                     'official_f1': f1(prediction, gold, int(qa['category'])),
                     'generic_token_f1': generic_f1(prediction, gold),
                     'cited_source_recall': hit / len(set(evidence)) if evidence else None,
                     'gold_evidence': evidence})
    splits = {r['split'] for r in records}
    split_label = next(iter(splits)) if len(splits) == 1 else 'all'
    save(Path(outdir) / f'{name}_{split_label}_items.json', rows)
    return rows


def summarize(rows):
    def mean(k, data=rows):
        v = [r[k] for r in data if r[k] is not None]
        return sum(v) / len(v) if v else None
    bycat = {str(c): {'n': len([r for r in rows if r['category'] == c]),
                     'f1': mean('official_f1', [r for r in rows if r['category'] == c])}
             for c in sorted({r['category'] for r in rows})}
    failures = [r for r in rows if r['official_f1'] < 0.5]
    return {'n': len(rows), 'official_f1': mean('official_f1'), 'generic_token_f1': mean('generic_token_f1'),
            'read_tokens': mean('read_tokens'), 'memory_tokens': mean('memory_tokens'),
            'cited_source_recall': mean('cited_source_recall'), 'by_category': bycat,
            'failures_below_half_f1': len(failures),
            'failures_missing_cited_sources': sum(r['cited_source_recall'] is not None and r['cited_source_recall'] < 1 for r in failures)}


def paired_interval(baseline, winner, seed):
    """Paired bootstrap by conversation; descriptive only with three holdout clusters."""
    import numpy as np
    b = {r['id']: r for r in baseline}
    if set(b) != {r['id'] for r in winner}:
        raise ValueError('Unpaired result comparison')
    groups = defaultdict(list)
    for r in winner:
        groups[r['conv_id']].append(r['official_f1'] - b[r['id']]['official_f1'])
    values = list(groups.values())
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(5000):
        selected = rng.integers(len(values), size=len(values))
        flat = [x for i in selected for x in values[i]]
        draws.append(float(np.mean(flat)))
    return {'delta': float(np.mean([x for v in values for x in v])),
            'ci95_cluster_bootstrap': np.quantile(draws, [0.025, 0.975]).tolist(),
            'conversation_clusters': len(values), 'note': 'Small pilot; few clusters limit inference.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/locomo10.json')
    ap.add_argument('--environment', default='environment.json')
    ap.add_argument('--out', default='runs/pilot100')
    ap.add_argument('--model', default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model', default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--seed', type=int, default=20260907)
    ap.add_argument('--budget', type=int, default=2048)
    ap.add_argument('--per-conv', type=int, default=10)
    ap.add_argument('--max-rounds', type=int, default=5)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--min-delta', type=float, default=0.001)
    ap.add_argument('--manifest-only', action='store_true')
    ap.add_argument('--parent-run', default=None, help='Record the earlier development run whose generation cache was reused')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = json.loads(Path(args.data).read_text(encoding='utf-8'))
    manifest = make_manifest(samples, args.seed, args.per_conv)
    mp = out / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('The frozen manifest changed; use a new output directory')
    save(mp, manifest)
    if args.manifest_only:
        print(json.dumps({'n':len(manifest['records']), 'holdout':manifest['holdout_convs']}))
        return
    environment = json.loads(Path(args.environment).read_text())
    protocol = {'args': vars(args), 'environment': environment,
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'reader': READER, 'selection': 'dev official category F1 only',
                'construction': 'conversation sessions only; no benchmark QA or gold access',
                'metric_source': 'https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py',
                'limits': 'pilot100; holdout is evaluated only after winner selection; memory storage is not matched'}
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed; use a new output directory')
    save(protocol_path, protocol)
    if (out / 'final.json').exists():
        print('Run already complete. Use a new preregistered split for further validation.')
        return
    status = {'phase': 'model_load', 'started_at': time.time()}
    save(out / 'status.json', status)
    rt = Runtime(args, environment)
    byid = {str(s['sample_id']): s for s in samples}
    sessions = {cid: session_data(s) for cid, s in byid.items()}
    dev = [r for r in manifest['records'] if r['split'] == 'dev']
    holdout = [r for r in manifest['records'] if r['split'] == 'holdout']
    dev_cids = sorted({r['conv_id'] for r in dev})
    history, memory_versions = [], {}
    best_name, best_score, best_rows, stale = None, -1, None, 0
    strategies = ['session10', 'events', 'audit', 'dialogue_residual', 'provenance_quotes']
    schedule = []
    for round_id in range(args.max_rounds):
        strategy = strategies[round_id] if round_id < len(strategies) else 'audit'
        name = f'r{round_id:02d}_{strategy}'
        parent_name = best_name
        schedule.append({'name': name, 'strategy': strategy, 'parent': parent_name})
        memory_versions[name] = {}
        for cid in dev_cids:
            save(out/'status.json', {**status, 'phase':'construction', 'round':name, 'conversation':cid})
            path = out / 'memories' / name / f'{cid}.json'
            if path.exists():
                units = json.loads(path.read_text())
            else:
                parent = memory_versions[parent_name][cid] if parent_name else None
                units = build(rt, sessions[cid], strategy, parent)
                save(path, units)
            memory_versions[name][cid] = units
            print(f'MEMORY {name} {cid} units={len(units)}', flush=True)
        save(out/'status.json', {**status, 'phase':'dev_evaluation', 'round':name})
        rows = evaluate(rt, memory_versions[name], dev, byid, args.budget, name, out)
        summary = summarize(rows)
        accepted = summary['official_f1'] > best_score + args.min_delta
        history.append({'name':name, 'strategy':strategy, 'parent':parent_name,
                        'accepted':accepted, 'dev':summary})
        if accepted:
            best_name, best_score, best_rows, stale = name, summary['official_f1'], rows, 0
        else:
            stale += 1
        save(out/'history.json', {'rounds':history, 'winner':best_name, 'schedule':schedule})
        print('ROUND_RESULT ' + json.dumps(history[-1]), flush=True)
        if stale >= args.patience:
            break
    # Lock selection BEFORE looking at held-out answers or predictions.
    selection = {'winner':best_name, 'dev_f1':best_score, 'schedule':schedule, 'locked_at':time.time()}
    save(out/'selection_locked.json', selection)
    required = {schedule[0]['name'], best_name}
    changed = True
    while changed:
        previous = set(required)
        for plan in schedule:
            if plan['name'] in required and plan['strategy'] in ('audit','dialogue_residual','provenance_quotes') and plan['parent']:
                required.add(plan['parent'])
        changed = required != previous
    for plan in schedule:
        name = plan['name']
        if name not in required:
            continue
        for cid in sorted({r['conv_id'] for r in holdout}):
            save(out/'status.json', {**status, 'phase':'holdout_construction', 'round':name, 'conversation':cid})
            path = out/'memories'/name/f'{cid}.json'
            if path.exists():
                units = json.loads(path.read_text())
            else:
                parent = memory_versions.get(plan['parent'], {}).get(cid)
                units = build(rt, sessions[cid], plan['strategy'], parent)
                save(path, units)
            memory_versions[name][cid] = units
    baseline_name = schedule[0]['name']
    baseline_hold = evaluate(rt, memory_versions[baseline_name], holdout, byid, args.budget, baseline_name, out)
    winner_hold = baseline_hold if best_name == baseline_name else evaluate(rt, memory_versions[best_name], holdout, byid, args.budget, best_name, out)
    # A raw-turn RAG reference at the same reader and read-token cap.
    raw_mem = {cid: raw_units(s) for cid, s in sessions.items()}
    raw_rows = evaluate(rt, raw_mem, manifest['records'], byid, args.budget, 'raw_rag_reference', out)
    baseline_dev = json.loads((out/f'{baseline_name}_dev_items.json').read_text())
    save(out/'pilot100_baseline_items.json', baseline_dev+baseline_hold)
    save(out/'pilot100_winner_items.json', best_rows+winner_hold)
    final = {'winner':best_name, 'dev_baseline':summarize(baseline_dev), 'dev_winner':summarize(best_rows),
             'holdout_baseline':summarize(baseline_hold), 'holdout_winner':summarize(winner_hold),
             'holdout_paired_delta':paired_interval(baseline_hold,winner_hold,args.seed),
             'pilot100_baseline':summarize(baseline_dev+baseline_hold),
             'pilot100_winner':summarize(best_rows+winner_hold),
             'raw_rag_reference':summarize(raw_rows),
             'raw_rag_holdout':summarize([r for r in raw_rows if r['split']=='holdout']),
             'seconds':time.time()-status['started_at'],
             'interpretation':'100 combines selected-on dev and sealed holdout; only holdout is out-of-sample.'}
    save(out/'final.json', final)
    save(out/'status.json', {**status, 'phase':'complete', 'winner':best_name})
    print('FINAL ' + json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
