"""Question-blind memory operations for the next refinement generation."""
from collections import defaultdict
import copy
import re

import refine as core
from continuous import calendar_anchor, dialogue_blocks

OPERATIONS={'clean_cards','fuse_evidence','entity_profiles','selective_anchor','salient_dialogue'}

TOPICS={
    'family and relationships':'children names spouse partner parents siblings family friends pets dogs cats names ownership relationships',
    'interests and creative activities':'hobbies interests books authors movies music instruments art sports creative activities collections',
    'work and education':'jobs career profession business store studio education school studies projects achievements future plans',
    'health and lifestyle':'health diet food eating exercise gym running hiking routine wellbeing mental health lifestyle changes',
    'travel and places':'travel trips places destinations cities countries vacations outdoors camping visited lived places',
    'support and values':'support mentors friends family beliefs values identity challenges adversity coping aspirations motivations community volunteering'
}


def clean_cards(units, render='sentence'):
    seen=set()
    result=[]
    query=re.compile(r'^(?:what|where|when|why|who|how|did|do|does|can|could|would|have|has|are|is|will)\b',re.I)
    filler=re.compile(r'^(?:wow|thanks?\b|thank you|that.s (?:really |so )?cool|you.ve got guts|you got this|so cool|good luck)',re.I)
    for original in units:
        unit=copy.deepcopy(original)
        match=re.match(r'^(\([^\n]*?\)\s*)?([^|\n]+)\|([^|\n]+)\|(.+)$',unit['text'],re.S)
        if not match:
            result.append(unit)
            continue
        prefix,subject,attribute,value=match.groups()
        subject,attribute,value=(x.strip() for x in (subject,attribute,value))
        if (value.rstrip().endswith('?') and query.search(value)) or filler.search(value):
            continue
        key=(unit['session'],tuple(core.lexical(subject)),tuple(core.lexical(value)))
        if key in seen:
            continue
        seen.add(key)
        if render=='sentence':
            unit['text']=(prefix or '')+f'{subject} — {attribute}: {value}'
        elif render=='value':
            unit['text']=(prefix or '')+f'{subject}: {value}'
        result.append(unit)
    return core.dedupe(result)


def fuse_evidence(rt,sessions,parent,op):
    blocks=dialogue_blocks(sessions,op.get('size',3),op.get('overlap',1))
    facts=[u for u in parent if u['kind'] in ('fact','extractive_fact','profile_fact')]
    if op.get('anchor',False):
        from continuous import anchor_units
        blocks=anchor_units(sessions,blocks,op.get('style','month'))
        facts=anchor_units(sessions,facts,op.get('style','month'))
    mapped=defaultdict(list)
    for i,block in enumerate(blocks):
        for sid in block['sources']:
            mapped[sid].append(i)
    assignments=defaultdict(list)
    residual=[]
    for fact in facts:
        ids=set(fact['sources'])
        possible={i for sid in ids for i in mapped[sid]}
        if not possible:
            residual.append(fact)
            continue
        best=max(possible,key=lambda i:(len(ids & set(blocks[i]['sources'])),-len(blocks[i]['sources']),-i))
        assignments[best].append(fact)
    result=[]
    cap=op.get('unit_budget',640)
    for i,block in enumerate(blocks):
        attached=[]
        sources=set(block['sources'])
        for fact in assignments[i]:
            text='Facts:\n'+'\n'.join(attached+[fact['text']])+'\nOriginal dialogue:\n'+block['text']
            if rt.ntok(text)<=cap:
                attached.append(fact['text'])
                sources.update(fact['sources'])
            else:
                residual.append(fact)
        text=('Facts:\n'+'\n'.join(attached)+'\nOriginal dialogue:\n' if attached else '')+block['text']
        result.append(dict(block,text=text,sources=sorted(sources),kind='fused_dialogue'))
    if op.get('keep_fact_index',False):
        result+=facts
    else:
        result+=residual
    return core.dedupe(result)


def entity_profiles(rt,sessions,parent,op):
    import numpy as np
    from rank_bm25 import BM25Okapi
    people=sorted({t['speaker'] for s in sessions for t in s['turns']})
    blocks=dialogue_blocks(sessions,size=3,overlap=1)
    texts=[u['text'] for u in blocks]
    embeddings=rt.encode(texts)
    sparse=BM25Okapi([core.lexical(t) for t in texts])
    specs=[(person,topic,terms) for person in people for topic,terms in TOPICS.items()]
    questions=[f'Personal facts about {person}: {terms}' for person,topic,terms in specs]
    queries=rt.encode(questions,query=True)
    lookup={t['id']:(s,t) for s in sessions for t in s['turns']}
    chronological=[t['id'] for s in sessions for t in s['turns']]
    users,allowed=[],[]
    for (person,topic,_),question,query in zip(specs,questions,queries):
        def ranks(scores):
            order=np.argsort(-scores,kind='stable')
            values=np.empty(len(order),dtype=np.int64)
            values[order]=np.arange(len(order))
            return values
        scores=1/(60+ranks(embeddings@query))+1/(60+ranks(np.asarray(sparse.get_scores(core.lexical(question)))))
        order=np.argsort(-scores,kind='stable')[:100]
        ids=set()
        parts=[]
        budget=op.get('source_budget',5000)
        for index in order:
            new=[]
            for sid in blocks[int(index)]['sources']:
                if sid in ids:
                    continue
                session,turn=lookup[sid]
                text=f"(Recorded: {session['date']}) "+calendar_anchor(turn['text'],session['date'],'month')
                new.append((sid,text))
            if new and rt.ntok('\n'.join(parts+[text for sid,text in new]))<=budget:
                ids.update(sid for sid,text in new)
                parts.extend(text for sid,text in new)
        selected=[lookup[sid][1] for sid in chronological if sid in ids]
        allowed.append(selected)
        users.append(f'Target person: {person}\nProfile topic: {topic}\nDated original evidence:\n'+'\n'.join(parts)+'\n\nProfile memory:')
    prompt=("Construct a consolidated personal memory profile from the supplied dated dialogue only. "
            "Output up to 14 concise records, one per line beginning with all exact supporting source IDs [D1:2,D4:3]. "
            "Every record must explicitly name the TARGET PERSON. Organize related facts by attribute: for example "
            "a person's children, pets, books, hobbies, workplaces, trips or sources of support. Where multiple source "
            "turns supply items of the same attribute, collect all supported items into one short record and name each "
            "item once. Do not assign the other speaker's experiences, children, pets or possessions to the target. "
            "Distinguish a completed activity from a plan, recommendation, question, hypothetical or preference. "
            "Keep changing states and distinct events separated with their relevant time. For relative dates preserve "
            "the original phrase and its observation date, or the provided calendar interpretation with uncertainty. "
            "Do not infer outside knowledge or fill gaps. Keep exact names, titles and source content words. "
            "Omit greetings, praise and generic encouragement. Return NONE if no facts about the target are supported.")
    generated=rt.generate(prompt,users,max_tokens=1100)
    profiles=[]
    for text,turns,(person,topic,_) in zip(generated,allowed,specs):
        if not turns:
            continue
        session=lookup[turns[0]['id']][0]
        meta={'num':session['num'],'date':'multiple dated observations; event dates are stated in each fact'}
        for record in core.parse_facts(text,turns,meta,14):
            if not re.search(r'\b'+re.escape(person)+r'\b',record['text'],re.I):
                continue
            record.update(kind='profile_fact',calendar_anchored=True,person=person,topic=topic)
            profiles.append(record)
    return core.dedupe((parent if op.get('append',True) else [])+profiles)


def salient_dialogue(rt,sessions,parent,op):
    chunks=list(core.windows(sessions,size=20,overlap=0))
    prompt=("Identify ONLY dialogue turns that can be safely omitted from long-term personal memory. "
            "A removable turn contains nothing but a greeting, thanks, generic encouragement or an information-free question. "
            "Keep all personal facts, names and titles, possession/relationship information, event details, times, lists, "
            "specific evaluations and reactions, meaningful short answers, plans, corrections, and factual content embedded "
            "in a question. Do not remove a turn with a substantive image description. When uncertain, KEEP the turn. "
            "Output one exact source ID per removable turn as DROP [D1:2]. Output NONE if none can be safely omitted. "
            "Do not rewrite or summarize the dialogue.")
    users=[f"Recorded: {s['date']}\n"+'\n'.join(t['text'] for t in ts) for s,ts in chunks]
    outputs=rt.generate(prompt,users,max_tokens=400)
    dropped=set()
    for text,(_,turns) in zip(outputs,chunks):
        allowed={t['id'] for t in turns if '[image:' not in t['body']}
        ids=set(re.findall(r'(?im)^\s*DROP\s+\[(D\d+:\d+)\]\s*$',text))
        dropped.update(ids & allowed)
    blocks=[]
    for session in sessions:
        turns=session['turns']
        for index,turn in enumerate(turns):
            if turn['id'] in dropped:
                continue
            prior=turns[max(0,index-op.get('previous_turns',1)):index]
            prior=[t for t in prior if t['id'] not in dropped or '?' in t['body']]
            selected=prior+[turn]
            blocks.append({'text':f"(Session date: {session['date']}) "+'\n'.join(t['text'] for t in selected),
                           'sources':[t['id'] for t in selected],'session':session['num'],'kind':'salient_dialogue'})
    if op.get('anchor',True):
        from continuous import anchor_units
        blocks=anchor_units(sessions,blocks,'month')
    if op.get('include_facts',True):
        blocks+=[u for u in parent if u['kind'] in ('fact','extractive_fact','profile_fact')]
    return core.dedupe(blocks) or core.raw_units(sessions,paired=True)


def apply_operation(rt,sessions,units,op):
    kind=op['op']
    if kind=='clean_cards':
        return clean_cards(units,op.get('render','sentence'))
    if kind=='fuse_evidence':
        return fuse_evidence(rt,sessions,units,op)
    if kind=='entity_profiles':
        return entity_profiles(rt,sessions,units,op)
    if kind=='salient_dialogue':
        return salient_dialogue(rt,sessions,units,op)
    if kind=='selective_anchor':
        from continuous import anchor_units
        result=[]
        temporal=re.compile(r'\b(yesterday|last night|last week|last month|last year|next year|tomorrow|\w+ (?:days?|weeks?|months?|years?) ago)\b',re.I)
        for unit in units:
            if temporal.search(unit['text']):
                result.extend(anchor_units(sessions,[unit],op.get('style','month')))
            else:
                result.append(copy.deepcopy(unit))
        return result
    raise ValueError(f'Unknown operation {kind}')
