"""Route source-only probe cues into an existing memory without rewriting it."""
import refine as core
from budgeted_evidence import probe_options,multiple_choice_budget,storage_cost
from evidence_utility import context_for,reader_user
from evidence_fidelity import VERIFIER,accepted


def minimum_parent_cover(parent,required,ntok):
    """Minimum summed-unit-cost cover of a small source set; keep whole units.

    Concatenated payload cost is recounted by the caller for both read/storage caps.
    """
    required=list(dict.fromkeys(required))
    if not required or len(required)>18:
        raise ValueError('Source cover expects 1..18 diagnostic source IDs')
    positions={sid:i for i,sid in enumerate(required)}
    target=(1<<len(required))-1
    candidates=[]
    for i,unit in enumerate(parent):
        mask=sum(1<<positions[sid] for sid in set(unit['sources']) if sid in positions)
        if mask:
            candidates.append((i,mask,ntok(unit['text'])))
    states={0:(0,())}
    for index,mask,cost in candidates:
        future=dict(states)
        for state,(spent,chosen) in states.items():
            merged=state|mask
            if merged==state:
                continue
            proposal=(spent+cost,chosen+(index,))
            if merged not in future or proposal<future[merged]:
                future[merged]=proposal
        states=future
    if target not in states:
        return None
    indices=states[target][1]
    text='\n\n'.join(parent[i]['text'] for i in indices)
    return {'text':text,'sources':sorted({sid for i in indices for sid in parent[i]['sources']}),
            'parent_indices':list(indices)}


def make_options(rt,sessions,parent,rows):
    groups,admitted=probe_options(sessions,rows,rt.ntok,'joint',realized_only=True)
    flattened={}
    for group in groups:
        for option in group:
            flattened[option['id']]=option
    result={}
    for key,option in flattened.items():
        # Exact same required source set may occur as two named controls; retain
        # the option identity, while model generation caches deduplicate prompts.
        cover=minimum_parent_cover(parent,option['unit']['sources'],rt.ntok)
        if cover is None or rt.ntok(cover['text'])>2048:
            continue
        unit={**option['unit'],'text':cover['text'],'sources':cover['sources'],
              'kind':'parent_routed_evidence','parent_indices':cover['parent_indices']}
        result[key]={**option,'unit':unit,'cost':storage_cost(unit,rt.ntok),
                     'required_sources':option['unit']['sources']}
    return groups,result


def verify_parent_options(rt,sessions,rows,mapped):
    byid={row['id']:row for row in rows}
    turns={t['id']:t for s in sessions for t in s['turns']}
    keys=sorted(mapped)
    prompts=[reader_user(byid[mapped[key]['unit']['source_probe_id']]['question'],mapped[key]['unit']['text']) for key in keys]
    answers=rt.generate(core.READER,prompts,max_tokens=96)
    judge=[]
    for key,answer in zip(keys,answers):
        option=mapped[key]
        row=byid[option['unit']['source_probe_id']]
        original=context_for(row,turns,row['candidate_context_ids'])
        reference=row['generations']['full']['text']
        judge.append(f"ORIGINAL source:\n{original}\n\nQuestion: {row['question']}\nReference answer: {reference}"
                     f"\n\nCANDIDATE fragment:\n{option['unit']['text']}\nCandidate answer: {answer}\n\nVerdict:")
    verdicts=rt.generate(VERIFIER,judge,max_tokens=16)
    return {key:{'answer':answer,'verdict':verdict,'accepted':accepted(verdict)}
            for key,answer,verdict in zip(keys,answers,verdicts)}


def construct(parent,groups,mapped,mapped_verification,raw_verification,ntok,family,extra_budget):
    mapping=family.startswith('parent_')
    verified=family.endswith('_verified')
    single='_single' in family
    choices=[]
    for group in groups:
        options=[]
        for original in group:
            if single and original['kind']=='pair':
                continue
            option=mapped.get(original['id']) if mapping else original
            if option is None:
                continue
            if verified:
                if mapping:
                    keep=mapped_verification.get(option['id'],{}).get('accepted',False)
                else:
                    probe_id=option['unit']['source_probe_id']
                    ids='|'.join(option['unit']['sources'])
                    keep=option['kind']=='full' or raw_verification.get(probe_id,{}).get(ids,{}).get('accepted',False)
                if not keep:
                    continue
            options.append(option)
        choices.append(options)
    selected,details=multiple_choice_budget(choices,extra_budget,quantum=8)
    memory=parent+[o['unit'] for o in selected]
    baseline=sum(storage_cost(u,ntok) for u in parent)
    total=sum(storage_cost(u,ntok) for u in memory)
    assert total<=baseline+extra_budget
    return memory,{'family':family,'parent_tokens':baseline,'extra_budget':extra_budget,
                   'storage_tokens':total,'storage_cap':baseline+extra_budget,
                   'selected_options':[o['id'] for o in selected],'optimizer':details}
