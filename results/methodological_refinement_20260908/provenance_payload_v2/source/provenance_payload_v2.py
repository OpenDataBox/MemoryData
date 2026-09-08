"""Compile readable payloads with literal evidence and deletion controls.

Search keys remain exactly fixed. Source deletion is a model diagnostic, not a
causal proof, and repeated true evidence can make a supported statement fail it.
"""
import json
import re
import refine as core
from budgeted_evidence import storage_cost
from contrastive_events import original_context

SCHEMA={'type':'object','properties':{'facts':{'type':'array','maxItems':4,'items':{
    'type':'object','properties':{'text':{'type':'string'},'evidence':{'type':'array','minItems':1,'maxItems':3,'items':{
        'type':'object','properties':{'source_id':{'type':'string'},'quote':{'type':'string'}},
        'required':['source_id','quote'],'additionalProperties':False}}},
    'required':['text','evidence'],'additionalProperties':False}}},'required':['facts'],'additionalProperties':False}
WRITER=(
    'Turn a conversation focus into up to four concise, standalone factual statements. '
    'The original source is authoritative. Each statement must concern information actually asserted '
    'in a FOCUS turn; neighboring turns may only resolve references or necessary context. Use explicit '
    'person, event, object and time attribution. Do not turn questions, wishes, suggestions, plans or '
    'someone else\'s experience into completed facts about the speaker. Preserve uncertainty, negation '
    'and conditions. Do not infer identity or location from a picture description alone. Use no external '
    'knowledge. A relative date may be resolved only when uniquely determined by the session date; '
    'otherwise retain the original temporal qualification. For each statement, cite one to three source '
    'IDs and a short exact quote copied from each cited turn. At least one cited turn must be a FOCUS '
    'turn. The cited turns together must support the whole statement. Prefer informative content; omit '
    'generic greetings and compliments. Return JSON {facts:[{text,evidence:[{source_id,quote}]}]}; '
    'return an empty facts list if no statement can be grounded. Do not answer an external question.'
)
CHECKER=(
    'Determine whether the supplied original conversation alone supports the entire statement. '
    'Output exactly SUPPORTED if its person, event, object, time, quantity, polarity and conditions are '
    'all entailed. Output CONTRADICTED if the source establishes a conflicting fact. Otherwise output '
    'INSUFFICIENT. Missing evidence is not contradiction. Do not use outside knowledge or assume '
    'unstated identities. A recording date is not automatically the event date. An explicitly stated '
    'relative date can support only a uniquely determined conversion. No explanations.'
)


def window_for(sessions,focus_ids,ntok,cap=2048):
    focus=set(focus_ids)
    all_turns={t['id']:t for s in sessions for t in s['turns']}
    if not focus or not focus<=all_turns.keys():raise ValueError('Unknown focus source')
    focus_text=original_context(sessions,focus)
    if ntok(focus_text)>cap:
        return {'focus_ids':sorted(focus),'window_ids':sorted(focus),'oversized':True,'user':None}
    wanted=set(focus)
    neighbors=[]
    for session in sessions:
        for i,turn in enumerate(session['turns']):
            if turn['id'] not in focus:continue
            for distance in (1,2):
                for j in (i-distance,i+distance):
                    if 0<=j<len(session['turns']):neighbors.append(session['turns'][j]['id'])
    for sid in dict.fromkeys(neighbors):
        if sid not in wanted and ntok(original_context(sessions,wanted|{sid}))<=cap:wanted.add(sid)
    neighbor_text=original_context(sessions,wanted-focus) if wanted-focus else '(none)'
    user=f'FOCUS source IDs: {", ".join(sorted(focus))}\nFOCUS:\n{focus_text}\n\nNEIGHBORING source:\n{neighbor_text}\n\nJSON:'
    return {'focus_ids':sorted(focus),'window_ids':sorted(wanted),'oversized':False,'user':user}


def generate(rt,windows):
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    results=[None]*len(windows);pending=[]
    for i,window in enumerate(windows):
        if window['oversized']:
            results[i]={'object':None,'raw':'','finish_reason':'oversized_source'};continue
        user=window['user'];key=core.digest(['provenance_payload_v1',rt.model_meta,rt.args.seed,WRITER,user,SCHEMA,384])
        path=rt.cache/'provenance_payload_generations'/f'{key}.json'
        if path.exists():results[i]=json.loads(path.read_text())
        else:
            prompt=rt.tok.apply_chat_template([{'role':'system','content':WRITER},{'role':'user','content':user}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert rt.ntok(prompt)+384<=8192
            pending.append((i,path,prompt))
    for offset in range(0,len(pending),12):
        batch=pending[offset:offset+12]
        values=rt.llm.generate([x[2] for x in batch],SamplingParams(temperature=0,max_tokens=384,
            guided_decoding=GuidedDecodingParams(json=SCHEMA)),use_tqdm=False)
        for (i,path,_),value in zip(batch,values):
            answer=value.outputs[0]
            try:obj=json.loads(answer.text)
            except json.JSONDecodeError:obj=None
            results[i]={'object':obj,'raw':answer.text,'finish_reason':answer.finish_reason}
            core.save(path,results[i])
        print(f'PAYLOAD_GENERATE {offset+len(batch)}/{len(pending)}',flush=True)
    return results


def literal_facts(window,value,turns):
    obj=value['object'];valid=[];rejected=[]
    if not isinstance(obj,dict) or set(obj)!={'facts'} or not isinstance(obj['facts'],list) or len(obj['facts'])>4:
        return [],[{'reason':'invalid_generation','value':value}]
    for i,fact in enumerate(obj['facts']):
        okay=isinstance(fact,dict) and set(fact)=={'text','evidence'} and isinstance(fact['text'],str) and bool(fact['text'].strip())
        if okay:
            ev=fact['evidence'];okay=isinstance(ev,list) and 1<=len(ev)<=3
        if okay:
            okay=all(isinstance(e,dict) and set(e)=={'source_id','quote'} and isinstance(e['source_id'],str) and e['source_id'] in window['window_ids']
                      and isinstance(e['quote'],str) and bool(e['quote'].strip()) and e['quote'] in turns[e['source_id']]['text'] for e in ev)
        if okay:
            okay=len({e['source_id'] for e in ev})==len(ev) and bool({e['source_id'] for e in ev}&set(window['focus_ids']))
        if not okay:rejected.append({'index':i,'reason':'invalid_literal_evidence','fact':fact});continue
        item={'id':f'fact_{i}','text':fact['text'].strip(),'evidence':ev}
        valid.append(item)
    return valid,rejected


def compile_source(rt,sessions,units):
    groups={}
    for unit in units:
        key=core.digest(sorted(set(unit['sources'])))
        groups.setdefault(key,window_for(sessions,unit['sources'],rt.ntok))
    keys=sorted(groups);values=generate(rt,[groups[k] for k in keys])
    turns={t['id']:t for s in sessions for t in s['turns']};compiled={};jobs=[]
    for key,value in zip(keys,values):
        window=groups[key];facts,rejected=literal_facts(window,value,turns)
        compiled[key]={'window':window,'generation':value,'facts':facts,'rejected':rejected}
        for fact in facts:
            cited={e['source_id'] for e in fact['evidence']}
            contexts={'support':original_context(sessions,cited),
                      'deleted':original_context(sessions,set(window['window_ids'])-cited) if set(window['window_ids'])-cited else '(no conversation evidence)'}
            for mode,context in contexts.items():
                jobs.append((key,fact['id'],mode,f'Original conversation:\n{context}\n\nStatement: {fact["text"]}\n\nVerdict:'))
    verdicts=rt.generate(CHECKER,[x[3] for x in jobs],max_tokens=16)
    for (key,fid,mode,_),verdict in zip(jobs,verdicts):
        fact=next(f for f in compiled[key]['facts'] if f['id']==fid)
        fact[mode+'_verdict']=verdict
    for item in compiled.values():
        for fact in item['facts']:
            fact['supported']=bool(re.fullmatch(r'SUPPORTED[.!]?',fact['support_verdict'].strip().upper()))
            fact['deletion_sensitive']=fact['supported'] and bool(re.fullmatch(r'INSUFFICIENT[.!]?',fact['deleted_verdict'].strip().upper()))
    return compiled


def render(raw,facts):
    prefix='Statements:\n'+'\n'.join(f"[{','.join(e['source_id'] for e in f['evidence'])}] {f['text']}" for f in facts)+'\n\n' if facts else ''
    return prefix+'Original dialogue:\n'+raw


def construct(baseline,sessions,compiled,ntok,mode):
    if mode not in ('raw','normalized','supported','dependent'):raise ValueError('Unknown payload control')
    memory=[];details=[]
    for index,unit in enumerate(baseline):
        item=compiled[core.digest(sorted(set(unit['sources'])))];raw=original_context(sessions,unit['sources'])
        cap=ntok(unit['text']);key=unit.get('index_text',unit['text'])
        candidates=[] if mode=='raw' else [f for f in item['facts'] if mode=='normalized' or
            (mode=='supported' and f['supported']) or (mode=='dependent' and f['deletion_sensitive'])]
        selected=[];fallback=ntok(render(raw,[]))>cap
        if not fallback:
            for fact in candidates:
                if ntok(render(raw,selected+[fact]))<=cap:selected.append(fact)
        text=unit['text'] if fallback else render(raw,selected)
        sources=sorted(set(unit['sources'])|{e['source_id'] for f in selected for e in f['evidence']})
        result={**unit,'text':text,'index_text':key,'sources':sources,'kind':'provenance_payload_'+mode,
                'payload_focus_ids':sorted(set(unit['sources'])),'payload_fact_ids':[f['id'] for f in selected]}
        assert ntok(text)<=cap and result['index_text']==key
        memory.append(result)
        details.append({'unit_index':index,'source_group':core.digest(sorted(set(unit['sources']))),'selected_fact_ids':[f['id'] for f in selected],
                        'fallback_original_payload':fallback,'payload_tokens':ntok(text),'payload_cap':cap})
    original=sum(storage_cost(u,ntok) for u in baseline);cost=sum(storage_cost(u,ntok) for u in memory)
    assert cost<=2*original
    assert [u.get('index_text',u['text']) for u in baseline]==[u['index_text'] for u in memory]
    return memory,{'mode':mode,'storage_tokens':cost,'storage_cap':2*original,'parent_tokens':original,
                   'search_keys_identical':True,'unit_count_identical':True,'units':details,
                   'selected_statements':sum(len(d['selected_fact_ids']) for d in details),
                   'fallback_original_payloads':sum(d['fallback_original_payload'] for d in details)}


_initial_generate=generate


def generate(rt,windows):
    """Reuse complete 384-token responses; retry incomplete JSON at 768/1536."""
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    values=_initial_generate(rt,windows)
    valid=lambda v:isinstance(v['object'],dict) and set(v['object'])=={'facts'} and isinstance(v['object']['facts'],list) and len(v['object']['facts'])<=4
    attempts={i:[] for i in range(len(windows))}
    for i,w in enumerate(windows):
        if not w['oversized']:
            key=core.digest(['provenance_payload_v1',rt.model_meta,rt.args.seed,WRITER,w['user'],SCHEMA,384])
            attempts[i].append({'max_tokens':384,'cache_key':key,'complete_json':valid(values[i])})
    for cap in (768,1536):
        pending=[]
        for i,w in enumerate(windows):
            if w['oversized'] or valid(values[i]):continue
            key=core.digest(['provenance_payload_v1',rt.model_meta,rt.args.seed,WRITER,w['user'],SCHEMA,cap])
            path=rt.cache/'provenance_payload_generations'/f'{key}.json'
            if path.exists():
                values[i]=json.loads(path.read_text())
                attempts[i].append({'max_tokens':cap,'cache_key':key,'complete_json':valid(values[i])})
            else:
                prompt=rt.tok.apply_chat_template([{'role':'system','content':WRITER},{'role':'user','content':w['user']}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
                assert rt.ntok(prompt)+cap<=8192
                pending.append((i,key,path,prompt))
        for offset in range(0,len(pending),12):
            batch=pending[offset:offset+12]
            generated=rt.llm.generate([x[3] for x in batch],SamplingParams(temperature=0,max_tokens=cap,
                guided_decoding=GuidedDecodingParams(json=SCHEMA)),use_tqdm=False)
            for (i,key,path,_),value in zip(batch,generated):
                answer=value.outputs[0]
                try:obj=json.loads(answer.text)
                except json.JSONDecodeError:obj=None
                values[i]={'object':obj,'raw':answer.text,'finish_reason':answer.finish_reason,'max_tokens':cap}
                core.save(path,values[i])
                attempts[i].append({'max_tokens':cap,'cache_key':key,'complete_json':valid(values[i])})
            print(f'PAYLOAD_RETRY cap={cap} {offset+len(batch)}/{len(pending)}',flush=True)
    return [{**value,'attempts':attempts[i],'complete_json':valid(value)} for i,value in enumerate(values)]
