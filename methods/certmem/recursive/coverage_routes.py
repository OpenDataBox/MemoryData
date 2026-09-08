"""Budgeted factual routes that preserve the strongest readable memory.

Facility-location gain covers supported source statements not already close to
the original search keys. It is a query-free proxy, not an answer-utility oracle.
"""
import numpy as np
import refine as core
from budgeted_evidence import storage_cost


def harvest(baseline,compiled,ntok):
    candidates=[];seen=set()
    existing={(u.get('index_text',u['text']),u['text']) for u in baseline}
    for index,unit in enumerate(baseline):
        group=core.digest(sorted(set(unit['sources'])))
        for fact in compiled[group]['facts']:
            cited=sorted({e['source_id'] for e in fact['evidence']})
            signature=(fact['text'],tuple(cited))
            if not fact['supported'] or not set(cited)<=set(unit['sources']) or signature in seen:continue
            seen.add(signature)
            route={'text':unit['text'],'index_text':fact['text'],'sources':list(unit['sources']),
                   'session':unit['session'],'kind':'supported_source_route','parent_index':index,
                   'source_group':group,'source_fact_id':fact['id']}
            if (route['index_text'],route['text']) in existing:continue
            candidates.append({'id':core.digest([group,fact['id'],index]),'unit':route,'cited_source_ids':cited,
                               'deletion_sensitive':fact['deletion_sensitive'],'cost':storage_cost(route,ntok)})
    return candidates


def geometry(fact_vectors,baseline_vectors):
    facts=np.asarray(fact_vectors,dtype=np.float32);base=np.asarray(baseline_vectors,dtype=np.float32)
    if facts.ndim!=2 or base.ndim!=2 or facts.shape[1]!=base.shape[1]:raise ValueError('Invalid route embedding shapes')
    return {'similarities':np.clip(facts@facts.T,0,1),'initial_coverage':np.clip((facts@base.T).max(axis=1),0,1),
            'fact_vectors':facts,'baseline_vectors':base}


def select(candidates,geom,budget,pool):
    if budget<0 or pool not in ('supported','dependent'):raise ValueError('Invalid route policy')
    sim=np.asarray(geom['similarities'],dtype=np.float64);covered=np.asarray(geom['initial_coverage'],dtype=np.float64).copy()
    if sim.shape!=(len(candidates),len(candidates)) or covered.shape!=(len(candidates),):raise ValueError('Invalid route geometry')
    remaining={i for i,c in enumerate(candidates) if pool=='supported' or c['deletion_sensitive']}
    chosen=[];used_sources=set();spent=0;trace=[]
    while remaining:
        eligible=[i for i in remaining if candidates[i]['cost']+spent<=budget and not set(candidates[i]['unit']['sources'])&used_sources]
        if not eligible:break
        gains={i:float(np.maximum(sim[:,i]-covered,0).mean()) for i in eligible}
        index=max(eligible,key=lambda i:(gains[i]/max(1,candidates[i]['cost']),gains[i],-candidates[i]['cost'],candidates[i]['id']))
        gain=gains[index]
        if gain<=1e-12:break
        candidate=candidates[index];chosen.append(index);spent+=candidate['cost'];used_sources.update(candidate['unit']['sources'])
        covered=np.maximum(covered,sim[:,index]);remaining.remove(index)
        trace.append({'candidate_id':candidate['id'],'marginal_mean_coverage':gain,'cumulative_cost':spent})
    return chosen,{'pool':pool,'budget':budget,'spent':spent,'selected_ids':[candidates[i]['id'] for i in chosen],
                   'initial_mean_coverage':float(np.mean(geom['initial_coverage'])) if len(candidates) else 0.,
                   'final_mean_coverage':float(covered.mean()) if len(candidates) else 0.,'trace':trace,
                   'no_source_repeated_across_added_routes':True}


def construct(baseline,candidates,geom,ntok,budget,pool):
    chosen,details=select(candidates,geom,budget,pool)
    memory=list(baseline)+[dict(candidates[i]['unit']) for i in chosen]
    parent=sum(storage_cost(u,ntok) for u in baseline);cost=sum(storage_cost(u,ntok) for u in memory)
    assert memory[:len(baseline)]==baseline and cost==parent+details['spent']<=parent+budget
    return memory,{**details,'storage_tokens':cost,'storage_cap':parent+budget,'parent_tokens':parent,
                   'baseline_prefix_identical':True,'added_routes':len(chosen)}
