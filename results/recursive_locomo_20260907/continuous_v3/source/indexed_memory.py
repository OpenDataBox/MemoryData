"""Separate source-derived retrieval keys from verbatim evidence payloads."""
from collections import defaultdict
import re

import refine as core
from continuous import anchor_units


def dedupe_memory(units):
    if not any('index_text' in u for u in units):
        return core.dedupe(units)
    result=[]
    seen=set()
    for unit in units:
        signature=(unit.get('index_text',unit['text']),unit['text'])
        if signature not in seen:
            result.append(unit)
            seen.add(signature)
    return result


def index_view(units,op):
    stopwords=set('a an the and or but if then so is are was were be been being am to of for at on in by with from as this that these those it its i you he she we they me him her us them my your his our their what which who whom when where why how do does did have has had can could would should will shall'.split())
    result=[]
    for original in units:
        unit=dict(original)
        text=unit.get('index_text',unit['text'])
        text=re.sub(r'\((?:Session date|Recorded on):[^\n]*?\)\s*','',text)
        text=re.sub(r'\[(?:D\d+:\d+(?:,\s*D\d+:\d+)*|original relative expression:[^\]]*)\]\s*','',text)
        if op.get('remove_stopwords',False):
            text=' '.join(word for word in text.split() if word.lower().strip('.,:;?!') not in stopwords)
        unit['index_text']=text.strip() or original['text']
        result.append(unit)
    return dedupe_memory(result)


def split_fused_index(units,op):
    result=[]
    for original in units:
        unit=dict(original)
        text=unit['text']
        if unit['kind']=='fused_dialogue' and text.startswith('Facts:\n'):
            facts,separator,source=text[len('Facts:\n'):].partition('\nOriginal dialogue:\n')
            if separator:
                unit['index_text']=facts
                unit['text']=text if op.get('read_summary',False) else source
        else:
            unit['index_text']=unit.get('index_text',text)
        result.append(unit)
    return dedupe_memory(result)


def indexed_evidence(rt,sessions,parent,op):
    lookup={t['id']:(s,i,t) for s in sessions for i,t in enumerate(s['turns'])}
    order={t['id']:i for i,t in enumerate(t for s in sessions for t in s['turns'])}
    facts=[]
    for unit in parent:
        if unit['kind'] in ('fact','extractive_fact','profile_fact'):
            facts.append(unit)
        elif unit['kind']=='indexed_evidence':
            facts.append(dict(unit,text=unit['index_text'],sources=unit['fact_sources'],kind='fact'))
        elif unit['kind']=='fused_dialogue' and unit['text'].startswith('Facts:\n'):
            summary,separator,_=unit['text'][len('Facts:\n'):].partition('\nOriginal dialogue:\n')
            if separator:
                facts.append(dict(unit,text=summary,kind='fact',calendar_anchored=True))
    if op.get('anchor',True):
        facts=anchor_units(sessions,facts,'month')
    groups=defaultdict(list)
    for fact in facts:
        ids=tuple(sorted((sid for sid in fact['sources'] if sid in lookup),key=order.get))
        if ids:
            groups[ids].append(fact['text'])
    covered=set()
    units=[]

    def source_payload(ids):
        expanded=set()
        for sid in ids:
            session,index,turn=lookup[sid]
            selected=session['turns'][max(0,index-op.get('before',1)):index+1+op.get('after',0)]
            expanded.update(t['id'] for t in selected)
        parts=[]
        for session in sessions:
            selected=[t for t in session['turns'] if t['id'] in expanded]
            if not selected:
                continue
            unit={'text':f"(Session date: {session['date']}) "+'\n'.join(t['text'] for t in selected),
                  'sources':[t['id'] for t in selected],'session':session['num'],'kind':'raw'}
            if op.get('anchor',True):
                unit=anchor_units(sessions,[unit],'month')[0]
            parts.append(unit['text'])
        return '\n'.join(parts),sorted(expanded,key=order.get)

    for ids,keys in groups.items():
        key='\n'.join(dict.fromkeys(keys))
        payload,sources=source_payload(ids)
        if op.get('include_summary',False):
            payload='Facts:\n'+key+'\nSource dialogue:\n'+payload
        index=key+'\n'+payload if op.get('index_source',False) else key
        units.append({'text':payload,'index_text':index,'sources':sources,'fact_sources':list(ids),
                      'session':lookup[ids[0]][0]['num'],'kind':'indexed_evidence','calendar_anchored':True})
        covered.update(ids)
    if op.get('residual',True):
        for sid in sorted(set(lookup)-covered,key=order.get):
            payload,sources=source_payload([sid])
            session,_,turn=lookup[sid]
            units.append({'text':payload,'index_text':payload,'sources':sources,
                          'session':session['num'],'kind':'indexed_residual','calendar_anchored':True})
    if not units:
        return core.raw_units(sessions,paired=True)
    # The retrieval key is semantically relevant even when payloads happen to match.
    seen=set()
    result=[]
    for unit in units:
        signature=(unit['index_text'],unit['text'])
        if signature not in seen:
            seen.add(signature)
            result.append(unit)
    return result
