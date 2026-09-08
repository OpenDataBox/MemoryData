"""Create a new controlled runner; completed payload sources are immutable."""
from pathlib import Path


def main():
    here=Path(__file__).parent;source=(here/'run_payload_refinement.py').read_text(encoding='utf-8')
    def change(old,new):
        nonlocal source
        assert old in source,old;source=source.replace(old,new)
    change('Source-only payload construction with fixed retrieval keys and deletion controls.', 'Query-free factual routes preserving the strongest original readable memory.')
    change('from provenance_payload import compile_source,construct,WRITER,CHECKER,SCHEMA','from coverage_routes import harvest,geometry,construct')
    change("default='runs/provenance_payload_v1'","default='runs/coverage_routes_v1'")
    change("previous=Path('runs/transfer_index_v1')","previous=Path('runs/provenance_payload_v2')")
    start=source.index('    source_names=');end=source.index('    hashes=',start)
    source=source[:start]+'''    source_names=('run_route_refinement.py','coverage_routes.py','retrieval_contract.py','crossview_probes.py',
                  'budgeted_evidence.py','evidence_fidelity.py','evidence_utility.py','run_evidence_utility.py','evaluate_indexed.py','refine.py')
''' +source[end:]
    change("recipes=[{'name':'aa_'+mode,'family':'source_grounded_payload','mode':mode} for mode in ('raw','normalized','supported','dependent')]",
           "recipes=[{'name':f'ab_{pool}_{budget}','family':'source_factual_routes','pool':pool,'extra_budget':budget} for budget in (2000,4000) for pool in ('supported','dependent')]")
    change(",'writer':WRITER,'checker':CHECKER,'schema':SCHEMA",'')
    start=source.index("              'construction':");end=source.index("    if (out/'protocol.json').exists():",start)
    source=source[:start]+'''              'construction':'Use supported statements from the completed source-only compiler; require every cited source ID to belong to a whole baseline unit. New route search key is the statement; readable payload is that baseline unit unchanged. Keep the entire original memory as an identical prefix.',
              'objective':'Query-free facility-location coverage of all eligible supported statement embeddings. Initial coverage is maximum cosine to existing memory search keys. Greedily choose positive marginal mean coverage per actual key+payload token, excluding repeated sources across added routes.',
              'controls':'Supported versus deletion-sensitive candidate pools, each at 2000/4000 added tokens per conversation. Both use the same all-supported coverage targets. No benchmark question or source probe question enters route construction.',
              'invariants':'Original memory texts, keys, unit order and reader are preserved. New factual routes are appended, so retrieval ranks and packing can change. Each added payload is an exact copy of an original unit; no new readable statement or date rule.',
              'budget':'Actual new key and copied readable payload both count. Storage cap equals baseline plus declared added tokens. Dense index array bytes are recorded separately.',
              'selection':'After every route memory is locked, assess matched 75 facts in Q0/A. Require positive net preserved answers in both; maximize worst gain, sum, smaller total storage, name; otherwise baseline. Lock before new dev evaluation.',
              'limits':'Facility location and factual query expansion are established methods; no novelty claim. Semantic coverage is a proxy, not answer utility. Source support and deletion are same-model judgments. No audit100/B outcomes used.'}
''' +source[end:]
    start=source.index('    compiled={}');end=source.index("    state('source_baseline_assessment')",start)
    source=source[:start]+'''    source_payloads={cid:read(previous/'source_payloads'/f'{cid}.json') for cid in cids}
    source_lock=read(previous/'source_payloads_locked.json')
    assert core.digest(source_payloads)==source_lock['compiled_sha256']
    core.save(out/'source_payloads_locked.json',source_lock)
    candidates_by_id={};geometry_by_id={}
    for cid in cids:
        state('source_route_construction',conversation=cid)
        core.save(out/'base_memories'/f'{cid}.json',baseline[cid])
        core.save(out/'source_payloads'/f'{cid}.json',source_payloads[cid])
        base_vectors=rt.encode([u.get('index_text',u['text']) for u in baseline[cid]])
        path=out/'indexes/baseline'/f'{cid}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,base_vectors)
        candidates=harvest(baseline[cid],source_payloads[cid],rt.ntok)
        fact_vectors=rt.encode([c['unit']['index_text'] for c in candidates]) if candidates else np.zeros((0,base_vectors.shape[1]),dtype=np.float32)
        geom=geometry(fact_vectors,base_vectors)
        candidates_by_id[cid]=candidates;geometry_by_id[cid]=geom
        core.save(out/'route_candidates'/f'{cid}.json',candidates)
        path=out/'route_geometry'/f'{cid}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**geom)
        print('ROUTE_CANDIDATES '+str({'conv_id':cid,'supported':len(candidates),'dependent':sum(c['deletion_sensitive'] for c in candidates)}),flush=True)
    core.save(out/'routes_locked.json',{'locked_at':time.time(),'candidates_sha256':core.digest(candidates_by_id),'used_probe_questions':False,'used_benchmark_questions':False})
    memories={}
    for recipe in recipes:
        name=recipe['name'];memory={};state('construction',round=name)
        core.save(out/'plans'/f'{name}.json',{'recipe':recipe,'source_sha256':hashes,'committed_at':time.time()})
        for cid in cids:
            memory[cid],details=construct(baseline[cid],candidates_by_id[cid],geometry_by_id[cid],rt.ntok,recipe['extra_budget'],recipe['pool'])
            vectors=rt.encode([u.get('index_text',u['text']) for u in memory[cid]])
            path=out/'indexes'/name/f'{cid}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,vectors)
            details['dense_index_bytes']=vectors.nbytes
            core.save(out/'memories'/name/f'{cid}.json',memory[cid]);core.save(out/'construction'/name/f'{cid}.json',details)
        memories[name]=memory
    core.save(out/'route_memories_locked.json',{'locked_at':time.time(),'memory_sha256':{n:core.digest(m) for n,m in memories.items()},'baseline_prefix_identical':True})
''' +source[end:]
    change("print('PAYLOAD_ASSESS '","print('ROUTE_ASSESS '");change("print('PAYLOAD_DEV '","print('ROUTE_DEV '")
    change("'source_payload_assessment'","'source_route_assessment'")
    target=here/'run_route_refinement.py'
    if target.exists():assert target.read_text(encoding='utf-8')==source
    else:target.write_text(source,encoding='utf-8',newline='\n')
    print(target)


if __name__=='__main__':main()
