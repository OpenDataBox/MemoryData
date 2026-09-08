"""Source-only actual-reader assessment of a stored-vector memory index."""
from collections import defaultdict
import refine as core
from evaluate_vector_memory import evaluate
from evidence_utility import context_for
from evidence_fidelity import VERIFIER,accepted


def assess(rt,memories,vectors,sessions,rows,name,out,view='q0'):
    if any(r['split']!='probe_fit' for r in rows) or view not in ('q0','fit_a'):
        raise ValueError('Only source-fit supervision is permitted')
    if view=='fit_a' and any(r.get('view')!='fit' for r in rows):
        raise ValueError('Only fit view A is permitted')
    pseudo=defaultdict(lambda:{'qa':[]})
    records=[]
    byid={r['id']:r for r in rows}
    for row in sorted(rows,key=lambda r:r['id']):
        cid=row['conv_id'];index=len(pseudo[cid]['qa'])
        pseudo[cid]['qa'].append({'question':row['question'],'answer':row['generations']['full']['text'],
                                  'category':4,'evidence':row['source_ids']})
        records.append({'id':row['id'],'conv_id':cid,'qa_index':index,'category':4,
                        'split':'source_fit' if view=='q0' else 'source_view_fit'})
    values=evaluate(rt,memories,vectors,records,dict(pseudo),2048,name,out)
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
