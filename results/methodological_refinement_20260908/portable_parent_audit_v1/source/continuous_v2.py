"""Persistent, development-only refinement worker; recipes can be queued while running.

The previous sealed holdout is never evaluated here. The frozen reader, scorer,
retriever and actual-tokenizer read cap are imported unchanged from refine.py.
Every recipe is committed before construction and accepts dialogue sessions only.
"""
from __future__ import annotations

import argparse
import calendar
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback

import refine as core
from memory_ops import OPERATIONS, apply_operation


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def day_label(day):
    return f'{day.day} {day.strftime("%B")} {day.year}'


def parse_date(text):
    m = re.search(r'\b(\d{1,2}) ([A-Za-z]+),? (\d{4})\b', text)
    return datetime.strptime(' '.join(m.groups()), '%d %B %Y').date() if m else None


def calendar_anchor(text, date, style='annotated'):
    """Conservative calendar arithmetic. Ambiguous expressions keep their uncertainty."""
    anchor = parse_date(date)
    if anchor is None:
        return text
    number = {'a':1, 'an':1, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
              'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10}
    def count(word):
        return int(word) if word.isdigit() else number[word]
    pattern = re.compile(r'\b(the day before yesterday|yesterday|last night|tomorrow|'
                         r'last week|last month|last year|next year|'
                         r'(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+) '
                         r'(?:days?|weeks?|months?|years?) ago)\b', re.I)
    def replace(m):
        phrase = m.group(0)
        value = phrase.lower()
        if value in ('yesterday','last night','the day before yesterday','tomorrow'):
            days = {'yesterday':-1,'last night':-1,'the day before yesterday':-2,'tomorrow':1}[value]
            resolved = day_label(anchor + timedelta(days=days))
            if value == 'last night':
                resolved += ' at night'
        elif value == 'last week':
            end = anchor - timedelta(days=anchor.weekday()+1)
            start = end - timedelta(days=6)
            if style == 'month' and start.month == end.month:
                resolved = start.strftime('%B %Y') + ' (the preceding week)'
            else:
                resolved = f'the week before {day_label(anchor)} ({day_label(start)} to {day_label(end)})'
        elif value == 'last month':
            resolved = (anchor.replace(day=1)-timedelta(days=1)).strftime('%B %Y')
        elif value in ('last year','next year'):
            resolved = str(anchor.year + (-1 if value == 'last year' else 1))
        else:
            n, unit, _ = value.split()
            n = count(n)
            if unit.startswith(('day','week')):
                resolved = day_label(anchor-timedelta(days=n*(7 if unit.startswith('week') else 1)))
                resolved = 'approximately ' + resolved
            elif unit.startswith('month'):
                offset = anchor.year*12+anchor.month-1-n
                year, month = divmod(offset,12)
                resolved = f'approximately {calendar.month_name[month+1]} {year}'
            else:
                resolved = f'approximately {anchor.year-n}'
        return f'{resolved} [original relative expression: {phrase}]'
    return pattern.sub(replace, text)


def anchor_units(sessions, units, style='annotated'):
    dates = {s['num']:s['date'] for s in sessions}
    result = []
    for original in units:
        unit = copy.deepcopy(original)
        if unit.get('calendar_anchored'):
            result.append(unit)
            continue
        date = dates[unit['session']]
        body = re.sub(r'^\(Session date: [^\n]*?\)\s*', '', unit['text'])
        normalized = parse_date(date)
        unit['text'] = f"(Recorded on: {day_label(normalized) if normalized else date}) " + calendar_anchor(body,date,style)
        unit['calendar_anchored'] = True
        result.append(unit)
    return result


def dialogue_blocks(sessions, size=2, overlap=0):
    units = []
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError('Invalid dialogue block dimensions')
    for session, turns in core.windows(sessions,size=size,overlap=overlap):
        units.append({'text':f"(Session date: {session['date']}) " + '\n'.join(t['text'] for t in turns),
                      'sources':[t['id'] for t in turns], 'session':session['num'], 'kind':'dialogue_block'})
    return units


def construct(rt, sessions, parent, recipe):
    """Only sessions, existing conversation memory and a generic recipe enter this API."""
    units = copy.deepcopy(parent)
    for op in recipe['operations']:
        kind = op['op']
        if kind in OPERATIONS:
            units = apply_operation(rt,sessions,units,op)
        elif kind == 'anchor_time':
            units = anchor_units(sessions,units,op.get('style','annotated'))
        elif kind == 'facts_only':
            units = [u for u in units if u['kind'] in ('fact','extractive_fact','profile_fact')]
        elif kind == 'dialogue_blocks':
            blocks = dialogue_blocks(sessions,op.get('size',2),op.get('overlap',0))
            units = units+blocks if op.get('append',True) else blocks
        elif kind == 'filter_social_facts':
            # A generic ablation of boilerplate; it never uses benchmark questions.
            social = re.compile(r'\b(thank(?:ed|s)?|encourag(?:ed|es)|congratulat(?:ed|es)|greet(?:ed|s)|'
                                r'assur(?:ed|es)|expressed (?:gratitude|appreciation)|wished .* luck)\b',re.I)
            units = [u for u in units if u['kind'] not in ('fact','extractive_fact') or not social.search(u['text'])]
        elif kind == 'extract':
            chunks = list(core.windows(sessions,size=op.get('window',12),overlap=op.get('overlap',2)))
            users = [f"Recorded date: {s['date']}\nDialogue:\n"+'\n'.join(t['text'] for t in ts)+'\n\nMemory records:' for s,ts in chunks]
            generated = rt.generate(op['prompt'],users,max_tokens=op.get('max_tokens',1000))
            extracted = []
            for text,(s,ts) in zip(generated,chunks):
                records = core.parse_facts(text,ts,s,op.get('limit',28))
                for unit in records:
                    unit['kind'] = 'extractive_fact'
                if not records:
                    records = dialogue_blocks([dict(s,turns=ts)],size=2)
                extracted.extend(records)
            units = units+extracted if op.get('append',False) else extracted
        elif kind == 'quote_facts':
            normalized = [dict(u,kind='fact') for u in units if u['kind'] in ('fact','extractive_fact')]
            units = core.quoted_memory(sessions,normalized)
        else:
            raise ValueError(f'Unknown construction operation {kind}')
        units = core.dedupe(units)
    if not units:
        raise ValueError('Recipe produced empty memory')
    return units


def link_cache(source, target):
    """Immutable content-addressed entries; future writes always replace atomically."""
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.npy'):
            dest = target/path.relative_to(source)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                os.link(path,dest)


def diagnostics(rows):
    failures = [r for r in rows if r['official_f1'] < 0.5]
    return {'failures':len(failures),
            'unknown_answers':sum(r['prediction'].lower().strip(' .')=='unknown' for r in rows),
            'long_answers':sum(len(core.lexical(r['prediction']))>35 for r in rows),
            'failed_with_all_cited_sources':sum(r['cited_source_recall']==1 for r in failures),
            'items':[{k:r[k] for k in ('id','category','question','gold','prediction','official_f1','cited_source_recall')} for r in failures]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-run',default='runs/pilot100_v3')
    ap.add_argument('--out',default='runs/continuous_v2')
    ap.add_argument('--queue',default='continuous_v2_queue.json')
    ap.add_argument('--parent-run',default='runs/continuous_v1')
    ap.add_argument('--data',default='data/locomo10.json')
    ap.add_argument('--environment',default='environment.json')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    ap.add_argument('--min-delta',type=float,default=.001)
    args = ap.parse_args()
    out, seed = Path(args.out), Path(args.seed_run)
    out.mkdir(parents=True,exist_ok=True)
    original_manifest = read(seed/'manifest.json')
    records = [r for r in original_manifest['records'] if r['split']=='dev']
    cids = sorted({r['conv_id'] for r in records})
    samples = read(args.data)
    assert core.digest(samples) == original_manifest['dataset_sha256']
    byid = {str(s['sample_id']):s for s in samples if str(s['sample_id']) in cids}
    del samples
    sessions = {cid:core.session_data(sample) for cid,sample in byid.items()}
    assert not set(cids).intersection(original_manifest['holdout_convs'])
    environment = read(args.environment)
    source = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('continuous_v2.py','memory_ops.py','continuous.py','refine.py')}
    parent_run = Path(args.parent_run)
    parent_protocol = read(parent_run/'protocol.json')
    for key in ('model','embed_model','seed','budget','embed_batch_size'):
        assert parent_protocol['args'][key] == getattr(args,key)
    assert parent_protocol['records'] == records
    assert parent_protocol['reader'] == core.READER
    protocol = {'args':vars(args),'source_sha256':source,'environment':environment,
                'records':records,'excluded_holdout_conversations':original_manifest['holdout_convs'],
                'selection':'original dev70 only; no new generalization claim',
                'reader':core.READER,'retrieval':'unchanged core.evaluate BM25+dense RRF',
                'construction':'sessions-only, question-blind; queued generic recipes',
                'storage':'not matched; measured for each recipe',
                'previous_holdout':'already reported; not accessed or evaluated by this worker'}
    protocol['parent_protocol_sha256'] = core.digest(parent_protocol)
    if (out/'protocol.json').exists() and read(out/'protocol.json')!=protocol:
        raise ValueError('Protocol changed; use a new output directory')
    core.save(out/'protocol.json',protocol)
    status={'phase':'model_load','pid':os.getpid(),'started_at':time.time()}
    core.save(out/'status.json',status)
    link_cache(parent_run/'cache',out/'cache')
    if not (out/'history.json').exists():
        if read(parent_run/'status.json')['phase'] != 'awaiting_candidates':
            raise ValueError('Finish the current parent candidate batch before migration')
        # Completed artifacts are immutable; new results use atomic replacement.
        for path in parent_run.rglob('*.json'):
            relative = path.relative_to(parent_run)
            if relative.parts[0]=='cache' or relative.as_posix() in ('protocol.json','status.json'):
                continue
            dest=out/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                os.link(path,dest)
    rt = core.Runtime(args,environment)
    seed_name = read(seed/'selection_locked.json')['winner']
    seed_memory={cid:read(seed/'memories'/seed_name/f'{cid}.json') for cid in cids}
    if (out/'history.json').exists():
        history=read(out/'history.json')
    else:
        rows=core.evaluate(rt,seed_memory,records,byid,args.budget,'seed',out)
        expected=read(seed/'selection_locked.json')['dev_f1']
        summary=core.summarize(rows)
        assert abs(summary['official_f1']-expected)<1e-12, 'Seed result did not reproduce'
        history={'seed_name':seed_name,'seed':summary,'winner':'seed','best_f1':expected,'rounds':[]}
        core.save(out/'history.json',history)
        core.save(out/'seed_diagnostics.json',diagnostics(rows))
    def memory(name):
        if name=='seed':
            return seed_memory
        return {cid:read(out/'memories'/name/f'{cid}.json') for cid in cids}
    while True:
        queue=read(args.queue)
        names=[r['name'] for r in queue['recipes']]
        if len(names)!=len(set(names)) or any(not re.fullmatch(r'[a-z0-9_]+',n) for n in names):
            raise ValueError('Recipe names must be unique simple identifiers')
        done={r['name']:r for r in history['rounds']}
        for recipe in queue['recipes']:
            if recipe['name'] in done and core.digest(recipe)!=done[recipe['name']]['recipe_sha256']:
                raise ValueError('Completed recipe was edited')
        pending=[r for r in queue['recipes'] if r['name'] not in done]
        if not pending:
            core.save(out/'status.json',{**status,'phase':'awaiting_candidates','winner':history['winner'],
                                       'best_f1':history['best_f1'],'heartbeat_at':time.time()})
            time.sleep(10)
            continue
        recipe=pending[0]
        name=recipe['name']
        plan_path=out/'plans'/f'{name}.json'
        if plan_path.exists():
            plan=read(plan_path)
            assert plan['recipe']==recipe
        else:
            parent=recipe.get('parent','best')
            if parent=='best':
                parent=history['winner']
            plan={'recipe':recipe,'parent':parent,'committed_at':time.time(),
                  'recipe_sha256':core.digest(recipe),'source_sha256':source}
            core.save(plan_path,plan)
        started=time.time()
        try:
            parent_memory=memory(plan['parent'])
            memories={}
            for cid in cids:
                core.save(out/'status.json',{**status,'phase':'construction','round':name,'conversation':cid,'heartbeat_at':time.time()})
                path=out/'memories'/name/f'{cid}.json'
                units=read(path) if path.exists() else construct(rt,sessions[cid],parent_memory[cid],recipe)
                core.save(path,units)
                memories[cid]=units
                print(f'MEMORY {name} {cid} units={len(units)}',flush=True)
            core.save(out/'status.json',{**status,'phase':'dev_evaluation','round':name,'heartbeat_at':time.time()})
            rows=core.evaluate(rt,memories,records,byid,args.budget,name,out)
            summary=core.summarize(rows)
            accepted=summary['official_f1']>history['best_f1']+args.min_delta
            result={'name':name,'parent':plan['parent'],'recipe_sha256':plan['recipe_sha256'],
                    'accepted':accepted,'dev':summary,'seconds':time.time()-started}
            if accepted:
                history['winner'],history['best_f1']=name,summary['official_f1']
                core.save(out/'best_dev_items.json',rows)
            core.save(out/f'{name}_diagnostics.json',diagnostics(rows))
        except Exception as error:
            result={'name':name,'parent':plan['parent'],'recipe_sha256':plan['recipe_sha256'],
                    'accepted':False,'error':str(error),'traceback':traceback.format_exc(),'seconds':time.time()-started}
            print(result['traceback'],flush=True)
        history['rounds'].append(result)
        core.save(out/'history.json',history)
        print('ROUND_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
