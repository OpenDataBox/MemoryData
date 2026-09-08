"""Budgeted source-probe evidence bundles, constructed before benchmark queries.

The joint policy considers complementary pairs even with weak singleton utility.
Its single/full and full-only controls use the same utility and storage accounting.
This module does not claim that source-probe likelihood guarantees factuality.
"""
import math
import refine as core
from evidence_utility import POLICY,context_for


def storage_cost(unit,ntok):
    key=unit.get('index_text',unit['text'])
    return ntok(unit['text'])+(ntok(key) if key!=unit['text'] else 0)


def pareto_options(options):
    """Keep only alternatives that improve utility at increasing token cost."""
    result=[]
    best=-math.inf
    for option in sorted(options,key=lambda o:(o['cost'],-o['gain'],o['id'])):
        if option['gain']>best+1e-12:
            result.append(option)
            best=option['gain']
    return result


def multiple_choice_budget(groups,budget,quantum=8):
    """Dynamic program: at most one representation per source probe.

    Costs round UP, budget rounds DOWN, so the discretization never overspends.
    The returned solution is optimal for these discretized, additive costs.
    """
    if budget<0 or quantum<1:
        raise ValueError('Invalid storage budget')
    limit=budget//quantum
    states={0:(0.0,())}
    all_options={}
    for group in groups:
        options=pareto_options(group)
        for option in options:
            if option['id'] in all_options:
                raise ValueError('Repeated option identifier')
            all_options[option['id']]=option
        future=dict(states)
        for spent,(value,path) in states.items():
            for option in options:
                cost=math.ceil(option['cost']/quantum)
                total=spent+cost
                if total>limit:
                    continue
                proposal=(value+option['gain'],path+(option['id'],))
                prior=future.get(total)
                if prior is None or proposal[0]>prior[0]+1e-12:
                    future[total]=proposal
        # A higher-cost state with no more utility cannot help later groups.
        states={}
        best=-math.inf
        for cost,(value,path) in sorted(future.items()):
            if value>best+1e-12:
                states[cost]=(value,path)
                best=value
    rounded,(value,path)=max(states.items(),key=lambda x:(x[1][0],-x[0]))
    chosen=[all_options[name] for name in path]
    cost=sum(o['cost'] for o in chosen)
    assert cost<=rounded*quantum<=budget
    return chosen,{'gain':value,'actual_tokens':cost,'rounded_tokens':rounded*quantum,
                   'budget':budget,'quantum':quantum,'unused_tokens':budget-cost}


def probe_options(sessions,rows,ntok,mode,verification=None,realized_only=False):
    verified=mode.endswith('_verified')
    family=mode.removesuffix('_verified')
    kinds={'joint':{'single','pair','full'},'single':{'single','full'},'full':{'full'}}[family]
    turns={t['id']:t for s in sessions for t in s['turns']}
    groups=[]
    admitted=[]
    for row in sorted(rows,key=lambda r:r['id']):
        if row['split']!='probe_fit':
            raise ValueError('Audit probes cannot enter refinement construction')
        if 'analysis' not in row:
            continue
        if not row['analysis']['source_dependent'] or row['generations']['full']['source_answer_f1']<POLICY['minimum_full_answer_f1']:
            continue
        empty=next(s for s in row['subsets'] if s['kind']=='empty')['score']['mean_logprob']
        full=next(s for s in row['subsets'] if s['kind']=='full')['score']['mean_logprob']
        advantage=full-empty
        if advantage<POLICY['minimum_full_advantage_nats']:
            continue
        options=[]
        realized={tuple(g['source_ids']) for g in row['generations'].values()}
        for i,subset in enumerate(row['subsets']):
            if subset['kind'] not in kinds:
                continue
            ids=tuple(subset['source_ids'])
            if realized_only and ids not in realized:
                continue
            if verified and subset['kind']!='full':
                if verification is None:
                    raise ValueError('Verified policy requires source-only verification records')
                if not verification.get(row['id'],{}).get('|'.join(ids),{}).get('accepted',False):
                    continue
            score=subset['score']['mean_logprob']
            gain=min(1.0,max(0.0,(score-empty)/advantage))
            if gain<=0:
                continue
            context=context_for(row,turns,subset['source_ids'])
            unit={'text':context,'index_text':row['question'],'sources':subset['source_ids'],
                  'session':row['session'],'kind':'utility_evidence_bundle','source_probe_id':row['id']}
            options.append({'id':row['id']+':'+str(i),'gain':gain,'cost':storage_cost(unit,ntok),
                            'unit':unit,'kind':subset['kind'],'mean_logprob':score})
        groups.append(options)
        admitted.append(row['id'])
    return groups,admitted


def construct(sessions,rows,ntok,mode,budget,retain_raw=True,quantum=8,verification=None,realized_only=False):
    """Only source sessions, source-fit diagnostics and frozen policy enter."""
    baseline=core.raw_units(sessions) if retain_raw else []
    baseline_cost=sum(storage_cost(unit,ntok) for unit in baseline)
    if baseline_cost>budget:
        raise ValueError('Budget cannot retain the declared raw residual')
    groups,admitted=probe_options(sessions,rows,ntok,mode,verification,realized_only)
    selected,stats=multiple_choice_budget(groups,budget-baseline_cost,quantum)
    memory=baseline+[option['unit'] for option in selected]
    if not memory:
        raise ValueError('No source-supported units fit the construction budget')
    actual=sum(storage_cost(unit,ntok) for unit in memory)
    assert actual<=budget
    return memory,{'mode':mode,'retain_raw':retain_raw,'budget':budget,'storage_tokens':actual,
                   'realized_only':realized_only,
                   'raw_residual_tokens':baseline_cost,'source_fit_probes':len(rows),'admitted_probes':len(admitted),
                   'selected_probes':len(selected),'selected_options':[o['id'] for o in selected],
                   'utility':stats,'limitation':'Utility measured on source probes, not unseen benchmark questions; no semantic guarantee.'}
