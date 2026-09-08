"""Independent query wording for source-only memory construction contracts.

Native schema-constrained generation uses the installed vLLM 0.10.2 API.
View quality is judged against original sources, never against a memory candidate.
"""
import copy
from collections import defaultdict
import json
from pathlib import Path

import refine as core
from evidence_utility import context_for,reader_user
from evidence_fidelity import VERIFIER,accepted
from evaluate_indexed import evaluate

SCHEMA={'type':'object','properties':{'question_1':{'type':'string'},'question_2':{'type':'string'}},
        'required':['question_1','question_2'],'additionalProperties':False}
WRITER=(
    'Produce two independently worded natural questions requesting the SAME information as the original '
    'source question. Preserve its person, event, time scope, requested granularity, and required facts. '
    'Use substantially different sentence structures from the original and each other. Make each question '
    'understandable within the whole conversation using source-supported event context when necessary. '
    'Do not give away the answer, add answer hints, invent facts, or mention source IDs. '
    'Return only a JSON object with string fields question_1 and question_2.'
)
QUESTION_CHECK=(
    'Judge two questions in the context of the original dialogue. Output exactly EQUIVALENT if they '
    'request the same information at the same granularity about the same person/event/time scope, '
    'and the new question does not newly reveal or imply the reference answer. Faithful rewording and '
    'source-supported clarification of the event are allowed. Output DIFFERENT if the requested fact '
    'changes, becomes leading, or needs a different answer. Output UNKNOWN if uncertain. No other text.'
)


def fingerprint(text):
    return tuple(core.lexical(text))


def assign_views(probe_id,generated,forbidden):
    if set(generated)!={'question_1','question_2'} or any(not isinstance(v,str) or not v.strip() for v in generated.values()):
        raise ValueError('Invalid question-view object')
    values=[generated['question_1'].strip(),generated['question_2'].strip()]
    keys=[fingerprint(v) for v in values]
    if not all(keys) or keys[0]==keys[1] or any(k in forbidden for k in keys):
        raise ValueError('A generated view repeats an existing question cue')
    # Assignment is independent of wording and any model outcome.
    flip=int(core.digest(['crossview_v1',probe_id])[-1],16)%2
    return {'fit':values[flip],'audit':values[1-flip]}


def generate_json(rt,users):
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    outputs=[None]*len(users)
    pending=[]
    for index,user in enumerate(users):
        key=core.digest(['crossview_json_v1',rt.model_meta,rt.args.seed,WRITER,user,SCHEMA,384])
        path=rt.cache/'structured_generations'/(key+'.json')
        if path.exists():
            outputs[index]=json.loads(path.read_text())['object']
        else:
            prompt=rt.tok.apply_chat_template([{'role':'system','content':WRITER},{'role':'user','content':user}],
                                              tokenize=False,add_generation_prompt=True,enable_thinking=False)
            if rt.ntok(prompt)+384>8192:
                raise ValueError('Question-view prompt exceeds context')
            pending.append((index,path,prompt))
    for start in range(0,len(pending),16):
        batch=pending[start:start+16]
        params=SamplingParams(temperature=0,max_tokens=384,guided_decoding=GuidedDecodingParams(json=SCHEMA))
        generated=rt.llm.generate([p[2] for p in batch],params,use_tqdm=False)
        for (index,path,prompt),result in zip(batch,generated):
            value=json.loads(result.outputs[0].text)
            outputs[index]=value
            core.save(path,{'object':value,'finish_reason':result.outputs[0].finish_reason})
        print(f'CROSSVIEW_JSON {start+len(batch)}/{len(pending)}',flush=True)
    return outputs


def prepare(rt,sessions,rows,out):
    forbidden={fingerprint(row['question']) for row in rows}
    originals={}
    users=[]
    for row in rows:
        turns={t['id']:t for s in sessions[row['conv_id']] for t in s['turns']}
        originals[row['id']]=context_for(row,turns,row['candidate_context_ids'])
        users.append(f"Original dialogue:\n{originals[row['id']]}\n\nOriginal question: {row['question']}"
                     f"\nReference answer (do not leak into the questions): {row['generations']['full']['text']}\n\nJSON:")
    generated=generate_json(rt,users)
    planned=[]
    rejected=[]
    for row,value in zip(rows,generated):
        try:
            views=assign_views(row['id'],value,forbidden)
            planned.append((row,views))
        except ValueError as error:
            rejected.append({'id':row['id'],'reason':str(error),'generated':value})
    tasks=[]
    for row,views in planned:
        for view in ('fit','audit'):
            question=views[view]
            tasks.append((row,view,question))
    questions=[f"Original dialogue:\n{originals[row['id']]}\nOriginal question: {row['question']}"
               f"\nReference answer: {row['generations']['full']['text']}\nNew question: {question}\nVerdict:"
               for row,view,question in tasks]
    question_checks=rt.generate(QUESTION_CHECK,questions,max_tokens=16)
    answers=rt.generate(core.READER,[reader_user(question,originals[row['id']]) for row,view,question in tasks],max_tokens=96)
    fidelity=[f"ORIGINAL source:\n{originals[row['id']]}\n\nQuestion: {row['question']}"
              f"\nReference answer: {row['generations']['full']['text']}\n\nCANDIDATE fragment:\n{originals[row['id']]}"
              f"\nCandidate answer: {answer}\n\nVerdict:" for (row,view,question),answer in zip(tasks,answers)]
    answer_checks=rt.generate(VERIFIER,fidelity,max_tokens=16)
    calibrated={}
    for (row,view,question),answer,qcheck,acheck in zip(tasks,answers,question_checks,answer_checks):
        calibrated.setdefault(row['id'],{})[view]={'question':question,'reference_answer':answer,
            'question_verdict':qcheck,'answer_verdict':acheck,'accepted':accepted(qcheck) and accepted(acheck)}
    byid={row['id']:row for row in rows}
    result={'fit':[],'audit':[]}
    for probe_id,views in sorted(calibrated.items()):
        if not all(views[v]['accepted'] for v in ('fit','audit')):
            continue
        for view in ('fit','audit'):
            row=copy.deepcopy(byid[probe_id])
            row['original_question']=row['question']
            row['question']=views[view]['question']
            row['view']=view
            if view=='audit':
                row['split']='probe_view_audit'
            row['generations']['full']['text']=views[view]['reference_answer']
            row['generations']['full']['source_answer_f1']=core.generic_f1(views[view]['reference_answer'],row['answer'])
            row['crossview_quality']=views[view]
            result[view].append(row)
    core.save(out/'quality.json',{'rejected_structure':rejected,'calibration':calibrated,
                                 'input_probes':len(rows),'paired_valid':len(result['fit'])})
    for view in ('fit','audit'):
        core.save(out/f'{view}_views.json',result[view])
    if len(result['fit'])<30 or {r['conv_id'] for r in result['fit']}!={r['conv_id'] for r in rows}:
        raise ValueError('Insufficient paired, calibrated query views; do not silently reuse original questions')
    return result


def require_fit(rows):
    if any(r.get('view')!='fit' or r['split']!='probe_fit' for r in rows):
        raise ValueError('Only fit query views may affect construction')


def assess_view(rt,memories,sessions,rows,name,out,selection_lock=None):
    views={r['view'] for r in rows}
    if len(views)!=1:
        raise ValueError('Do not mix fit and audit query views')
    view=next(iter(views))
    if view=='audit':
        if selection_lock is None or not Path(selection_lock).exists():
            raise ValueError('Audit memory evaluation requires an existing source selection lock')
    else:
        require_fit(rows)
    pseudo=defaultdict(lambda:{'qa':[]})
    records=[]
    byid={r['id']:r for r in rows}
    for row in sorted(rows,key=lambda r:r['id']):
        cid=row['conv_id']
        index=len(pseudo[cid]['qa'])
        pseudo[cid]['qa'].append({'question':row['question'],'answer':row['generations']['full']['text'],
                                  'category':4,'evidence':row['source_ids']})
        records.append({'id':row['id'],'conv_id':cid,'qa_index':index,'category':4,'split':'source_view_'+view})
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
