"""Source-only behavior contracts evaluated through the actual retrieval path."""
from collections import defaultdict

import refine as core
from evaluate_indexed import evaluate
from evidence_fidelity import VERIFIER,accepted
from evidence_utility import context_for,POLICY
from budgeted_evidence import multiple_choice_budget,storage_cost


def qualified(rows):
    if any(r['split']!='probe_fit' for r in rows):
        raise ValueError('Source-fit probes only; no audit may enter construction')
    return [r for r in rows if 'analysis' in r and r['analysis']['source_dependent']
            and r['generations']['full']['source_answer_f1']>=POLICY['minimum_full_answer_f1']]


def assess(rt,memories,sessions,rows,name,out):
    """Pseudo QA below is generated from sources, never benchmark QA."""
    pseudo=defaultdict(lambda:{'qa':[]})
    records=[]
    byid={r['id']:r for r in rows}
    for row in sorted(rows,key=lambda r:r['id']):
        cid=row['conv_id']
        index=len(pseudo[cid]['qa'])
        pseudo[cid]['qa'].append({'question':row['question'],'answer':row['generations']['full']['text'],
                                  'category':4,'evidence':row['source_ids']})
        records.append({'id':row['id'],'conv_id':cid,'qa_index':index,'category':4,'split':'source_fit'})
    values=evaluate(rt,memories,records,dict(pseudo),2048,name,out)
    prompts=[]
    for value in values:
        row=byid[value['id']]
        turns={t['id']:t for s in sessions[row['conv_id']] for t in s['turns']}
        original=context_for(row,turns,row['candidate_context_ids'])
        prompts.append(f"ORIGINAL source:\n{original}\n\nQuestion: {row['question']}\nReference answer: {row['generations']['full']['text']}"
                       f"\n\nCANDIDATE fragment:\n{value['context']}\nCandidate answer: {value['prediction']}\n\nVerdict:")
    verdicts=rt.generate(VERIFIER,prompts,max_tokens=16)
    return {row['id']:{'preserved':accepted(verdict),'verdict':verdict,'prediction':row['prediction'],
                       'context_tokens':row['read_tokens']} for row,verdict in zip(values,verdicts)}


def compare_contract(before,after):
    if before.keys()!=after.keys():
        raise ValueError('Contract sets changed during comparison')
    regressions=sorted(k for k in before if before[k]['preserved'] and not after[k]['preserved'])
    repairs=sorted(k for k in before if not before[k]['preserved'] and after[k]['preserved'])
    return {'regressions':regressions,'repairs':repairs,'feasible':not regressions and bool(repairs)}


def propose(current,option_groups,mapped_checks,assessment,ntok,family,step_budget,global_cap):
    verified=family.endswith('_verified')
    single=family.startswith('single')
    existing={(u.get('index_text',u['text']),u['text']) for u in current}
    groups=[]
    for group in option_groups:
        options=[]
        for option in group:
            probe_id=option['unit']['source_probe_id']
            if assessment[probe_id]['preserved'] or (single and option['kind']=='pair'):
                continue
            if verified and not mapped_checks.get(option['id'],{}).get('accepted',False):
                continue
            signature=(option['unit'].get('index_text',option['unit']['text']),option['unit']['text'])
            if signature in existing:
                continue
            options.append(option)
        groups.append(options)
    spent=sum(storage_cost(u,ntok) for u in current)
    available=min(step_budget,global_cap-spent)
    if available<0:
        raise ValueError('Current memory already violates the global storage cap')
    selected,details=multiple_choice_budget(groups,available,quantum=8)
    memory=current+[o['unit'] for o in selected]
    assert sum(storage_cost(u,ntok) for u in memory)<=global_cap
    return memory,details
