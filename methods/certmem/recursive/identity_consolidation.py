"""Coalesce exact evidence identities before retrieval, without changing the reader.

Only equal payload, source IDs and session identify the same evidence. Overlapping
or merely similar passages never merge. This is a construction ablation, not a
claim that index concatenation implements multi-vector retrieval.
"""
import copy
from collections import defaultdict

from budgeted_evidence import storage_cost


def identity(unit):
    return (unit['text'], tuple(sorted(set(unit['sources']))), unit['session'])


def consolidate(units, ntok, mode):
    if mode not in ('payload', 'union', 'cues'):
        raise ValueError('Unknown evidence consolidation mode')
    groups=defaultdict(list)
    for i, unit in enumerate(units):
        groups[identity(unit)].append(i)
    result=[]
    merged=[]
    for key,indices in groups.items():
        original=units[indices[0]]
        unit=copy.deepcopy(original)
        if len(indices)>1:
            keys=list(dict.fromkeys(units[i].get('index_text',units[i]['text']) for i in indices))
            cues=[text for text in keys if text!=original['text']]
            if mode=='payload':
                index=original['text']
            elif mode=='union':
                index='\n\n'.join(keys)
            else:
                index='\n\n'.join(cues) if cues else original['text']
            unit.pop('index_text',None)
            if index!=unit['text']:
                unit['index_text']=index
            unit['kind']='identity_consolidated_evidence'
            unit['merged_from_indices']=indices
            unit['source_probe_ids']=sorted({units[i]['source_probe_id'] for i in indices if 'source_probe_id' in units[i]})
            merged.append({'indices':indices,'distinct_keys':len(keys),'cue_keys':len(cues)})
        result.append(unit)
    # Preserve all original evidence identities and their exact payloads.
    assert {identity(u) for u in result}==set(groups)
    assert len(result)==len(groups)
    stats={'mode':mode,'input_units':len(units),'output_units':len(result),
           'removed_duplicate_payloads':len(units)-len(result),'merged_groups':merged,
           'parent_tokens':sum(storage_cost(u,ntok) for u in units),
           'storage_tokens':sum(storage_cost(u,ntok) for u in result),
           'all_evidence_identities_preserved':True}
    return result,stats
