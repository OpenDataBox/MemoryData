"""Matched fact-route construction with one readable identity per original unit.

All three controls select the same source facts under the maximum of their
incremental storage costs. Construction never sees a question or answer.
"""
from copy import deepcopy
import refine as core
from budgeted_evidence import storage_cost
from coverage_routes import select


def patch_key(parent,route,mode):
    if mode not in ('union','replace'):raise ValueError('Invalid key patch mode')
    unit=dict(parent)
    old=parent.get('index_text',parent['text'])
    unit['index_text']=old+'\n'+route['index_text'] if mode=='union' else route['index_text']
    unit['route_patch']={'source_group':route['source_group'],'source_fact_id':route['source_fact_id'],
                         'old_key_sha256':core.digest(old),'mode':mode}
    return unit


def common_candidates(baseline,candidates,ntok):
    result=deepcopy(candidates)
    for item in result:
        route=item['unit'];parent=baseline[route['parent_index']]
        assert all(route[k]==parent[k] for k in ('text','sources','session'))
        original=storage_cost(parent,ntok)
        costs={'copy':storage_cost(route,ntok),
               **{mode:storage_cost(patch_key(parent,route,mode),ntok)-original for mode in ('union','replace')}}
        item['mode_incremental_costs']=costs
        item['cost']=max(1,*costs.values())
    return result


def construct(baseline,candidates,geom,ntok,budget,mode):
    if mode not in ('copy','union','replace'):raise ValueError('Invalid route identity control')
    paired=common_candidates(baseline,candidates,ntok)
    chosen,selection=select(paired,geom,budget,'supported')
    memory=[dict(u) for u in baseline];parents=[]
    for index in chosen:
        route=paired[index]['unit'];parent_index=route['parent_index']
        assert parent_index not in parents;parents.append(parent_index)
        if mode=='copy':memory.append(dict(route))
        else:memory[parent_index]=patch_key(baseline[parent_index],route,mode)
    original=sum(storage_cost(u,ntok) for u in baseline)
    actual=sum(storage_cost(u,ntok) for u in memory)
    expected_delta=sum(paired[i]['mode_incremental_costs'][mode] for i in chosen)
    assert actual==original+expected_delta<=original+selection['spent']<=original+budget
    assert all(all(memory[i][k]==u[k] for k in ('text','sources','session')) for i,u in enumerate(baseline))
    if mode!='copy':assert len(memory)==len(baseline)
    return memory,{**selection,'mode':mode,'storage_tokens':actual,'storage_cap':original+budget,
                   'parent_tokens':original,'actual_added_tokens':actual-original,
                   'common_candidate_cost_sha256':core.digest(paired),'patched_parent_indices':parents,
                   'selected_route_count':len(chosen),'new_readable_copies':len(chosen) if mode=='copy' else 0,
                   'all_original_readable_payloads_preserved':True}
