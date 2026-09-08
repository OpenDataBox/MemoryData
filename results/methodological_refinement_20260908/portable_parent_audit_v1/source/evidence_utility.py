"""Source-only counterfactual evidence utility; no benchmark QA is accepted.

Enumerate all singleton/pair subsets, including low-marginal pairs. This is a
diagnostic and construction primitive, not a claim of semantic certification.
"""
from itertools import combinations
import math
import random

import refine as core

POLICY = {'seed':20260908, 'fit_per_conversation':4, 'audit_per_conversation':2,
          'read_budget':2048, 'preservation_tolerance_nats':0.2,
          'minimum_full_advantage_nats':0.1, 'minimum_full_answer_f1':0.8,
          'joint_lift_nats':0.2, 'max_answer_tokens':96,
          'selection':'lowest token cost among full, all singleton and all pair subsets within 0.2 nats/token of full',
          'budget_control':'same candidate set, identical tokenizer/read cap, paired single-vs-pair comparisons; storage frontier reported separately'}


def select_probes(pool, policy=POLICY):
    rng=random.Random(policy['seed'])
    selected=[]
    for cid in sorted({p['conv_id'] for p in pool['records']}):
        for split,key in [('probe_fit','fit_per_conversation'),('probe_audit','audit_per_conversation')]:
            candidates=sorted([p for p in pool['records'] if p['conv_id']==cid and p['split']==split],key=lambda p:p['id'])
            rng.shuffle(candidates)
            selected.extend(candidates[:policy[key]])
    return selected


def context_for(probe, turns, ids):
    if not ids:
        return ''
    return f"(Session date: {probe['recorded_date']}) "+'\n'.join(turns[sid]['text'] for sid in ids)


def reader_user(question, context):
    return f'Conversation memory:\n{context}\n\nQuestion: {question}\nAnswer:'


def subset_jobs(probe, turns, ntok, budget=2048):
    ids=list(dict.fromkeys(probe['candidate_context_ids']))
    if not set(ids)<=turns.keys():
        raise ValueError('Candidate source ID not present in source-only corpus')
    if not set(probe['source_ids'])<=set(ids):
        raise ValueError('Literal answer source not inside candidate window')
    full=context_for(probe,turns,ids)
    if ntok(full)>budget:
        return [],'full_source_window_exceeds_read_budget'
    choices=[('empty',()),('full',tuple(ids))]
    choices += [('single',x) for x in combinations(ids,1)]
    choices += [('pair',x) for x in combinations(ids,2)]
    choices += [('answer_removed',tuple(sid for sid in ids if sid not in probe['source_ids']))]
    jobs=[]
    for kind,subset in choices:
        context=context_for(probe,turns,subset)
        jobs.append({'kind':kind,'source_ids':list(subset),'context_tokens':ntok(context),
                     'user':reader_user(probe['question'],context),'answer':probe['answer']})
    return jobs,None


def answer_token_request(tokenizer, user, answer):
    prefix=tokenizer.apply_chat_template([{'role':'system','content':core.READER},
              {'role':'user','content':user}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    target=tokenizer.encode(answer,add_special_tokens=False)
    if not prefix or not target:
        raise ValueError('Empty prompt/answer tokens')
    # Explicit token concatenation prevents BPE boundary changes from shifting
    # the answer span. The same target tokenization is used for all contexts.
    return {'prompt_token_ids':list(prefix)+target},len(prefix),target


def extract_answer_logprob(output, start, targets):
    expected=list(targets)
    if list(output.prompt_token_ids[start:])!=expected:
        raise ValueError('Scored answer token IDs changed')
    values=[]
    for offset,token_id in enumerate(expected):
        row=output.prompt_logprobs[start+offset]
        if row is None or token_id not in row:
            raise ValueError('Missing actual target-token log probability')
        value=float(row[token_id].logprob)
        if not math.isfinite(value) or value>1e-5:
            raise ValueError('Invalid target log probability')
        values.append(value)
    return {'mean_logprob':sum(values)/len(values),'sum_logprob':sum(values),
            'answer_tokens':len(values),'token_logprobs':values}


def subset_analysis(jobs, policy=POLICY):
    empty=next(j for j in jobs if j['kind']=='empty')
    full=next(j for j in jobs if j['kind']=='full')
    singles=[j for j in jobs if j['kind']=='single']
    pairs=[j for j in jobs if j['kind']=='pair']
    by_single={j['source_ids'][0]:j for j in singles}
    lp=lambda j:j['score']['mean_logprob']
    best_single=max(singles,key=lp)
    best_pair=max(pairs,key=lp) if pairs else best_single
    eligible=[j for j in singles+pairs+[full] if lp(j)>=lp(full)-policy['preservation_tolerance_nats']]
    minimum=min(eligible,key=lambda j:(j['context_tokens'],-lp(j),j['source_ids']))
    single_eligible=[j for j in singles+[full] if lp(j)>=lp(full)-policy['preservation_tolerance_nats']]
    single_minimum=min(single_eligible,key=lambda j:(j['context_tokens'],-lp(j),j['source_ids']))
    interactions=[]
    for pair in pairs:
        a,b=(by_single[sid] for sid in pair['source_ids'])
        interactions.append({'source_ids':pair['source_ids'],
             'joint_lift':lp(pair)-max(lp(a),lp(b)),
             'interaction':lp(pair)-lp(a)-lp(b)+lp(empty),
             'advantage':lp(pair)-lp(empty)})
    source_advantage=lp(full)-lp(empty)
    return {'full_advantage':source_advantage,
            'source_dependent':source_advantage>=policy['minimum_full_advantage_nats'],
            'best_single_sources':best_single['source_ids'],'best_pair_sources':best_pair['source_ids'],
            'minimum_sources':minimum['source_ids'],'minimum_tokens':minimum['context_tokens'],
            'single_policy_sources':single_minimum['source_ids'],'single_policy_tokens':single_minimum['context_tokens'],
            'full_tokens':full['context_tokens'],'best_pair_lift_over_best_single':lp(best_pair)-lp(best_single),
            'pair_interactions':interactions,
            'has_preserving_pair_without_preserving_single':any(lp(j)>=lp(full)-policy['preservation_tolerance_nats'] for j in pairs)
                 and not any(lp(j)>=lp(full)-policy['preservation_tolerance_nats'] for j in singles)}


def generation_controls(jobs, analysis):
    controls={}
    for name in ('empty','full','answer_removed'):
        controls[name]=next(j for j in jobs if j['kind']==name)
    for name,key in [('best_single','best_single_sources'),('best_pair','best_pair_sources'),
                     ('minimum_bundle','minimum_sources'),('single_policy','single_policy_sources')]:
        controls[name]=next(j for j in jobs if j['source_ids']==analysis[key])
    return controls
