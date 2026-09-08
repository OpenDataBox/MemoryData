"""Create a new matched route-identity experiment without changing frozen runs."""
from pathlib import Path


def main():
    here=Path(__file__).parent;source=(here/'run_route_refinement.py').read_text(encoding='utf-8')
    def change(old,new):
        nonlocal source
        assert old in source,old;source=source.replace(old,new)
    change('Query-free factual routes preserving the strongest original readable memory.','Matched factual keys with and without additional readable copies.')
    change('from coverage_routes import harvest,geometry,construct','from coverage_routes import harvest,geometry\nfrom identity_routes import construct')
    change("default='runs/coverage_routes_v1'","default='runs/identity_routes_v1'")
    change("previous=Path('runs/provenance_payload_v2')","previous=Path('runs/coverage_routes_v1')")
    change("('run_route_refinement.py','coverage_routes.py'","('run_identity_route_refinement.py','identity_routes.py','coverage_routes.py'")
    change("recipes=[{'name':f'ab_{pool}_{budget}','family':'source_factual_routes','pool':pool,'extra_budget':budget} for budget in (2000,4000) for pool in ('supported','dependent')]",
           "recipes=[{'name':f'ac_{mode}_{budget}','family':'source_route_identity','mode':mode,'extra_budget':budget} for budget in (2000,4000) for mode in ('copy','union','replace')]")
    start=source.index("              'construction':");end=source.index("    if (out/'protocol.json').exists():",start)
    source=source[:start]+'''              'construction':'Use the exact supported self-contained fact candidates of coverage_routes_v1. Every fact is linked to an existing whole payload. Compare appended key+payload copies, original key plus newline plus factual key in the parent, and factual key replacing the parent search key. Original readable text, sources, session, unit order and reader are preserved in all controls.',
              'objective':'Use the same all-supported facility-location geometry as the preceding experiment. Greedily select marginal mean coverage divided by the maximum incremental storage cost of copy/union/replace. Exclude repeated original sources across selections.',
              'controls':'Three identity treatments at 2000/4000 common added-token budgets. All three treatments within a budget select exactly the same facts and parents. Union/replace introduce no new readable unit; copy deliberately introduces a new readable duplicate as the matched control.',
              'invariants':'No readable payload is rewritten or removed. Union preserves the original key as an exact prefix followed by a newline and the fact key. Replace deliberately removes that old key as an ablation. Retrieval embeddings and BM25 scores change in union/replace; the reader and packer do not.',
              'budget':'Each candidate common cost is max(1, copy cost, union incremental cost, replacement incremental cost), counted with the exact reader tokenizer. Same selected IDs and common budget across controls; actual storage may differ and is reported. Both distinct key and payload count; dense bytes separately.',
              'selection':'Lock all six memories before matched 75 facts in Q0/A. Require positive net preserved answers in both; maximize worst gain, sum, smaller actual total storage, name; otherwise baseline. Lock before six dev evaluations.',
              'limits':'This is a matched mechanism comparison, not proof of academic novelty or generalization. Facility location and document key expansion are established. No benchmark question or source probe enters construction. Previously consumed audit100/B are not inputs. The public JSON is parsed before QA is stripped; QA is not passed to construction.'}
''' +source[end:]
    change("recipe['extra_budget'],recipe['pool'])","recipe['extra_budget'],recipe['mode'])")
    change("        geom=geometry(fact_vectors,base_vectors)","        geom=geometry(fact_vectors,base_vectors)\n        assert candidates==read(previous/'route_candidates'/f'{cid}.json')\n        old_geom=np.load(previous/'route_geometry'/f'{cid}.npz')\n        assert all(np.array_equal(geom[key],old_geom[key]) for key in geom)")
    change("        memories[name]=memory", "        memories[name]=memory\n    for budget in (2000,4000):\n        for cid in cids:\n            controls=[read(out/'construction'/f'ac_{mode}_{budget}'/f'{cid}.json') for mode in ('copy','union','replace')]\n            assert all(d['selected_ids']==controls[0]['selected_ids'] and d['spent']==controls[0]['spent'] for d in controls)")
    change("'baseline_prefix_identical':True","'original_readable_payloads_preserved':True")
    change("'ROUTE_","'IDENTITY_ROUTE_")
    source=source.replace("state('source_route_construction'","state('source_identity_route_construction'")
    source=source.replace("state('source_route_assessment'","state('source_identity_route_assessment'")
    target=here/'run_identity_route_refinement.py'
    if target.exists():assert target.read_text(encoding='utf-8')==source
    else:target.write_text(source,encoding='utf-8',newline='\n')
    shell=(here/'run_routes_4080s.sh').read_text(encoding='utf-8').replace('test_coverage_routes.py','test_identity_routes.py').replace('run_route_refinement.py','run_identity_route_refinement.py')
    (here/'run_identity_routes_4080s.sh').write_text(shell,encoding='utf-8',newline='\n')
    conf=(here/'routes-supervisor.conf').read_text(encoding='utf-8').replace('locomo-routes-refinement','locomo-identity-routes-refinement').replace('run_routes_4080s.sh','run_identity_routes_4080s.sh').replace('/routes_refinement.log','/identity_routes_refinement.log')
    (here/'identity-routes-supervisor.conf').write_text(conf,encoding='utf-8',newline='\n')
    print(target)


if __name__=='__main__':main()
