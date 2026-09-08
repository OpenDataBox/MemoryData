"""Source-only keys that distinguish semantically similar conversation events.

The query-independent proposal graph uses fixed embeddings. A common, conservative
cost chooses exactly the same evidence pairs for literal and contrastive controls.
Selection is greedy budgeted matching, not an optimal graph algorithm.
"""
import json
import re

import refine as core
from budgeted_evidence import storage_cost

SCHEMA={'type':'object','properties':{
    'relation':{'type':'string','enum':['distinct_events','same_event','unrelated','uncertain']},
    'key':{'type':'string'}},'required':['relation','key'],'additionalProperties':False}
WRITER=(
    'Two conversation excerpts may describe similar experiences. Write a concise retrieval key that '
    'keeps the two events distinguishable instead of blending their details. Use only claims explicitly '
    'supported by the excerpts, with clear person, event and time attribution when available. Do not '
    'invent differences, resolve uncertain identities, infer unstated dates, or claim a contradiction. '
    'Both excerpts must contribute concrete information. Do not write a question or answer an external '
    'task. If there are two related but distinguishable events, relation is distinct_events and key is '
    'a short factual comparison. If they clearly refer to the same event, use same_event; if unrelated, '
    'use unrelated; if the event distinction cannot be supported, use uncertain. For those three cases '
    'leave key empty. Return only JSON with relation and key.'
)
VERIFIER=(
    'Check a factual retrieval key against two original conversation excerpts. Output exactly SUPPORTED '
    'only if every claim and the distinction between events are supported, person/event/time attributes '
    'are assigned to the correct excerpt, and both excerpts contribute concrete facts. A source date '
    'does not by itself establish the event date. Reject invented distinctions, contradictions or '
    'identities, and any blended attribution. Output UNSUPPORTED or UNCERTAIN otherwise. No other text.'
)


def proposal_pairs(units,embeddings,limit=64,max_degree=2):
    import numpy as np
    if limit<0 or max_degree<1 or len(units)!=len(embeddings):
        raise ValueError('Invalid graph inputs')
    matrix=np.asarray(embeddings)@np.asarray(embeddings).T
    sources=[set(u['sources']) for u in units]
    edges=[]
    for i in range(len(units)):
        for j in range(i+1,len(units)):
            if units[i]['session']==units[j]['session'] or sources[i]&sources[j]:
                continue
            score=float(matrix[i,j])
            if not np.isfinite(score):
                raise ValueError('Nonfinite embedding similarity')
            if score>0:
                edges.append((score,i,j))
    degree=[0]*len(units)
    chosen=[]
    for score,i,j in sorted(edges,key=lambda e:(-e[0],e[1],e[2])):
        if len(chosen)>=limit:
            break
        if degree[i]>=max_degree or degree[j]>=max_degree:
            continue
        chosen.append({'parent_indices':[i,j],'similarity':score})
        degree[i]+=1
        degree[j]+=1
    return chosen


def generate_keys(rt,users):
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    outputs=[None]*len(users)
    pending=[]
    for index,user in enumerate(users):
        key=core.digest(['contrastive_event_v1',rt.model_meta,rt.args.seed,WRITER,user,SCHEMA,320])
        path=rt.cache/'contrastive_event_generations'/f'{key}.json'
        if path.exists():
            outputs[index]=json.loads(path.read_text())
        else:
            prompt=rt.tok.apply_chat_template([{'role':'system','content':WRITER},{'role':'user','content':user}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            if rt.ntok(prompt)+320>8192:
                raise ValueError('Event key generation prompt exceeds context')
            pending.append((index,path,prompt))
    for offset in range(0,len(pending),16):
        batch=pending[offset:offset+16]
        values=rt.llm.generate([x[2] for x in batch],SamplingParams(temperature=0,max_tokens=320,
            guided_decoding=GuidedDecodingParams(json=SCHEMA)),use_tqdm=False)
        for (index,path,prompt),value in zip(batch,values):
            answer=value.outputs[0]
            try:
                obj=json.loads(answer.text)
            except json.JSONDecodeError:
                obj=None
            outputs[index]={'object':obj,'raw':answer.text,'finish_reason':answer.finish_reason}
            core.save(path,outputs[index])
    return outputs


def original_context(sessions,source_ids):
    wanted=set(source_ids)
    found=set()
    parts=[]
    for session in sessions:
        turns=[t for t in session['turns'] if t['id'] in wanted]
        if turns:
            parts.append(f"(Session date: {session['date']}) "+'\n'.join(t['text'] for t in turns))
            found.update(t['id'] for t in turns)
    if found!=wanted:
        raise ValueError('Memory references a source absent from the original conversation')
    return '\n\n'.join(parts)


def compile_candidates(rt,sessions,units,edges):
    inputs=[]
    legal=[]
    for edge in edges:
        i,j=edge['parent_indices']
        payload=units[i]['text']+'\n\n'+units[j]['text']
        if rt.ntok(payload)>2048:
            continue
        legal.append(edge)
        # Generated backend facts are not the authority for the new key.
        left=original_context(sessions,units[i]['sources'])
        right=original_context(sessions,units[j]['sources'])
        inputs.append(f"EXCERPT A:\n{left}\n\nEXCERPT B:\n{right}\n\nJSON:")
    generated=generate_keys(rt,inputs)
    proposed=[]
    rejected=[]
    for edge,user,value in zip(legal,inputs,generated):
        obj=value['object']
        if not isinstance(obj,dict) or set(obj)!={'relation','key'} or obj['relation']!='distinct_events' or not isinstance(obj['key'],str) or not obj['key'].strip():
            rejected.append({'edge':edge,'generation':value})
            continue
        proposed.append((edge,user,obj['key'].strip(),value))
    verdicts=rt.generate(VERIFIER,[user.rsplit('\n\nJSON:',1)[0]+f'\n\nKEY:\n{key}\n\nVerdict:'
                                  for edge,user,key,value in proposed],max_tokens=16)
    candidates=[]
    for (edge,user,key,value),verdict in zip(proposed,verdicts):
        accepted=bool(re.fullmatch(r'SUPPORTED[.!]?',verdict.strip().upper()))
        i,j=edge['parent_indices']
        payload=units[i]['text']+'\n\n'+units[j]['text']
        unit={'text':payload,'index_text':key,'sources':sorted(set(units[i]['sources'])|set(units[j]['sources'])),
              'session':min(units[i]['session'],units[j]['session']),'sessions':sorted({units[i]['session'],units[j]['session']}),
              'parent_indices':[i,j],'kind':'contrastive_event_pair'}
        candidates.append({'id':f'pair_{i}_{j}','edge':edge,'unit':unit,'key_generation':value,'verdict':verdict,
                           'accepted':accepted,'common_cost':storage_cost(unit,rt.ntok)})
    return {'proposal_edges':edges,'legal_edges':legal,'rejected_generations':rejected,'candidates':candidates}


def choose_pairs(candidates,budget):
    if budget<0:
        raise ValueError('Negative pair budget')
    selected=[]
    used_sources=set()
    spent=0
    ordered=sorted([c for c in candidates if c['accepted']],
                   key=lambda c:(-c['edge']['similarity']/max(1,c['common_cost']),c['common_cost'],c['id']))
    for candidate in ordered:
        if set(candidate['unit']['sources']) & used_sources:
            continue
        if spent+candidate['common_cost']>budget:
            continue
        selected.append(candidate)
        used_sources.update(candidate['unit']['sources'])
        spent+=candidate['common_cost']
    return selected,{'common_cost':spent,'budget':budget,'selected_ids':[c['id'] for c in selected],
                     'no_source_repeated_across_added_pairs':True}


def construct(baseline,candidates,ntok,mode,budget):
    if mode not in ('literal','contrastive'):
        raise ValueError('Unknown pair key control')
    selected,details=choose_pairs(candidates,budget)
    memory=list(baseline)
    for candidate in selected:
        unit=dict(candidate['unit'])
        if mode=='literal':
            unit.pop('index_text')
            unit['kind']='literal_event_pair_control'
        memory.append(unit)
    base=sum(storage_cost(u,ntok) for u in baseline)
    cost=sum(storage_cost(u,ntok) for u in memory)
    assert cost<=base+budget
    return memory,{**details,'mode':mode,'parent_tokens':base,'storage_tokens':cost,'storage_cap':base+budget,
                   'actual_added_tokens':cost-base}
