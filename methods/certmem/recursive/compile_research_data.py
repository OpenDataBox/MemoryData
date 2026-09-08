"""Source-only literal probes and a frozen new-question audit manifest.

Probe compilation never receives benchmark QA. Audit sampling uses only category
and positional identifiers, excludes all previously evaluated questions, and does
not materialize question or answer text. Audit conversations had a prior pilot;
only these audit questions are new, not the conversations themselves.
"""
import argparse
from collections import Counter,defaultdict
import json
import gzip
from pathlib import Path
import random
import re

import refine as core


def literal_probes(sessions,cards):
    turns={t['id']:(s,t) for s in sessions for t in s['turns']}
    results=[]
    seen=set()
    for card in cards:
        match=re.search(r'\bQ:\s*(.*?)\s+A:\s*(.+)$',card['text'],re.S)
        if not match:
            continue
        question,answer=(x.strip() for x in match.groups())
        answer=answer.strip('"“”').rstrip('.')
        if not question or not 1<=len(core.lexical(answer))<=12:
            continue
        ids=[sid for sid in card['sources'] if sid in turns]
        hits=[]
        for sid in ids:
            session,turn=turns[sid]
            found=re.search(re.escape(answer),turn['body'],re.I)
            if found and (found.start()==0 or not turn['body'][found.start()-1].isalnum()) and (found.end()==len(turn['body']) or not turn['body'][found.end()].isalnum()):
                hits.append({'source_id':sid,'start':found.start(),'end':found.end(),'verbatim':found.group(0)})
        if not hits:
            continue
        signature=(card['session'],question.lower(),answer.lower())
        if signature in seen:
            continue
        seen.add(signature)
        results.append({'question':question,'answer':hits[0]['verbatim'],'source_ids':list(dict.fromkeys(h['source_id'] for h in hits)),
                        'candidate_context_ids':ids,
                        'answer_spans':hits,'session':card['session'],
                        'recorded_date':turns[hits[0]['source_id']][0]['date']})
    return results


def cards_from_generations(dump):
    cards=defaultdict(list)
    pattern=re.compile(r'(?ms)^\s*(?:\[[^\]]+\]\s*)?Q:\s*(.*?)\s+A:\s*(.*?)(?=^\s*(?:\[[^\]]+\]\s*)?Q:|\Z)')
    for row in dump['records']:
        for match in pattern.finditer(row['text']):
            question,answer=(x.strip() for x in match.groups())
            cards[row['conv_id']].append({'text':f'Q: {question} A: {answer}',
                                        'sources':row['turn_ids'],'session':row['session']})
    return cards


def split_probe_pool(sessions_by_id,cards_by_id,seed=20260908,per_session=2):
    rng=random.Random(seed)
    results=[]
    audit_sessions={}
    counts={}
    for cid in sorted(sessions_by_id):
        probes=literal_probes(sessions_by_id[cid],cards_by_id[cid])
        buckets=defaultdict(list)
        for probe in probes:
            buckets[probe['session']].append(probe)
        session_ids=sorted(buckets)
        rng.shuffle(session_ids)
        held=set(session_ids[:max(1,len(session_ids)//4)])
        audit_sessions[cid]=sorted(held)
        selected=[]
        for session in sorted(buckets):
            values=buckets[session]
            rng.shuffle(values)
            for probe in values[:per_session]:
                selected.append({**probe,'conv_id':cid,'split':'probe_audit' if session in held else 'probe_fit'})
        for i,probe in enumerate(selected):
            probe['id']=f'{cid}:source_probe:{i}'
        results+=selected
        counts[cid]={'literal_candidates':len(probes),'selected':len(selected)}
    return {'seed':seed,'construction':'source Q&A cards with literal answer spans; benchmark QA never supplied',
            'counts':counts,'audit_sessions':audit_sessions,'records':results}


def audit_manifest(samples,old_manifest,n=100,seed=20260908):
    rng=random.Random(seed)
    excluded={r['id'] for r in old_manifest['records']}
    conversations=sorted(old_manifest['holdout_convs'])
    base,remainder=divmod(n,len(conversations))
    extra=rng.sample(conversations,remainder)
    records=[]
    for sample in sorted(samples,key=lambda s:str(s['sample_id'])):
        cid=str(sample['sample_id'])
        if cid not in conversations:
            continue
        buckets=defaultdict(list)
        for i,qa in enumerate(sample['qa']):
            category=int(qa['category'])
            if category in (1,2,3,4) and f'{cid}:{i}' not in excluded:
                buckets[category].append(i)
        target=base+(cid in extra)
        total=sum(map(len,buckets.values()))
        assert total>=target
        ideal={c:len(v)*target/total for c,v in buckets.items()}
        counts={c:int(v) for c,v in ideal.items()}
        for c in sorted(counts,key=lambda c:(-(ideal[c]-counts[c]),c))[:target-sum(counts.values())]:
            counts[c]+=1
        for category,indices in sorted(buckets.items()):
            rng.shuffle(indices)
            records += [{'id':f'{cid}:{i}','conv_id':cid,'qa_index':i,'category':category,'split':'question_audit'} for i in indices[:counts[category]]]
    assert len(records)==n and not ({r['id'] for r in records}&excluded)
    return {'seed':seed,'n':n,'dataset_sha256':core.digest(samples),'records':records,
            'conversations':conversations,'excluded_prior_question_count':len(excluded),
            'usage':'Evaluate only after locking a methodological policy and its declared controls; never use for refinement.',
            'limitation':'New questions from previously used holdout conversations; not untouched conversations or an external dataset.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--old-manifest',type=Path,required=True)
    ap.add_argument('--cards',type=Path)
    ap.add_argument('--generation-dump',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    samples=json.loads(args.data.read_text(encoding='utf-8'))
    old=json.loads(args.old_manifest.read_text(encoding='utf-8'))
    assert core.digest(samples)==old['dataset_sha256']
    cids=sorted({r['conv_id'] for r in old['records'] if r['split']=='dev'})
    sessions={str(s['sample_id']):core.session_data(s) for s in samples if str(s['sample_id']) in cids}
    if args.generation_dump:
        with gzip.open(args.generation_dump,'rt',encoding='utf-8') as stream:
            dump=json.load(stream)
        assert dump['dataset_sha256']==old['dataset_sha256']
        cards=cards_from_generations(dump)
    elif args.cards:
        cards={cid:json.loads((args.cards/f'{cid}.json').read_text(encoding='utf-8')) for cid in cids}
    else:
        raise ValueError('Supply source cards or cached source-only generation outputs')
    pool=split_probe_pool(sessions,cards)
    if not pool['records']:
        raise ValueError('No grounded probes; do not freeze an empty pool')
    audit=audit_manifest(samples,old)
    for name,value in [('source_probes.json',pool),('question_audit100_manifest.json',audit)]:
        path=args.out/name
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:
            raise ValueError('Frozen research data changed; create a new version')
        core.save(path,value)
    summary={'probe_counts':dict(Counter(r['split'] for r in pool['records'])),
             'source_conversations':cids,'audit_questions':len(audit['records']),
             'audit_conversations':audit['conversations'],'no_audit_question_or_answer_text_materialized':True,
             'literal_grounding':True,'source_probe_counts':pool['counts']}
    core.save(args.out/'data_validation.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':
    main()
