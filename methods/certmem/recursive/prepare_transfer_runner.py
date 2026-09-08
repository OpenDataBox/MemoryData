"""Create a separately frozen runner from the completed indexed comparison."""
from pathlib import Path


def main():
    here=Path(__file__).parent
    source=(here/'run_index_refinement.py').read_text(encoding='utf-8')
    def change(old,new):
        nonlocal source
        assert old in source,old
        source=source.replace(old,new)
    change('Source-supervised bounded optimization of stored dense memory vectors.',
           'Source-only rigid memory alignment with cross-conversation transfer controls.')
    change('from source_index_learning import support_mask,optimize_index',
           'from source_index_learning import support_mask\nfrom transfer_index_alignment import support_centroids,fit_rotation,apply_rotation,training_conversations')
    change("default='runs/source_index_v1'","default='runs/transfer_index_v1'")
    change("previous=Path('runs/contrastive_events_v1')","previous=Path('runs/source_index_v1')")
    change("sources=('run_index_refinement.py','source_index_learning.py','evaluate_vector_memory.py','assess_vector_memory.py',",
           "sources=('run_transfer_refinement.py','transfer_index_alignment.py','source_index_learning.py','evaluate_vector_memory.py','assess_vector_memory.py',")
    start=source.index('    recipes=');end=source.index('    expected=',start)
    source=source[:start]+'''    recipes=[{'name':f'z_{scope}_{label}','family':'source_supervised_memory_index','method':'rigid_cayley_alignment','radius':radius,'scope':scope}
             for label,radius in [('010',.1),('030',.3)] for scope in ('within','transfer')]
    optimizer={'method':'Cayley rotation from the skew first-order support-to-query alignment direction',
               'supervision':'Each source query pairs with the unit-normalized centroid of source-ID-positive memory vectors.',
               'constraint':'One orthogonal map per target index; all document-document inner products preserved. Spectral movement bounded by radius. Identity on orthogonal complement of training span.',
               'scopes':'within uses target Q0; transfer excludes all target-conversation Q0 and uses only six other development conversations.',
               'parameters':'stored document vectors only; query/document encoders and reader weights are frozen; no iterative gradient training'}
''' +source[end:]
    change("'limits':'Positive source overlap is weak supervision, not entailment. Finite source training may overfit. This is index construction rather than a new reader or embedding encoder. No independent audit100 or exposed B outcomes/caches used.'",
           "'limits':'Positive source overlap is weak supervision. Orthogonal alignment and Cayley maps are established methods, no novelty claim. Transfer excludes every target-conversation training query, but development conversations are not unseen globally. No audit100 or B outcomes enter fitting/selection.'")
    change("'After every index is optimized on original Q0 only, assess matched 75 facts on Q0/A.",
           "'After every index is aligned using original Q0 only under its declared within/transfer scope, assess matched 75 facts on Q0/A.")
    change("'invariants':'Memory texts, sparse keys, sources, query encoder, rank-fusion formula, packing and reader unchanged.",
           "'invariants':'Memory texts, sparse keys, sources, query encoder, rank-fusion formula, packing and reader unchanged; document-document geometry preserved.")
    start=source.index('            queries,positive=training_data[cid]');end=source.index("            assert vectors[cid].shape",start)
    source=source[:start]+'''            train_cids=training_conversations(cid,sorted(cids),recipe['scope'])
            support=np.concatenate([support_centroids(original_vectors[k],training_data[k][1]) for k in train_cids])
            query=np.concatenate([training_data[k][0] for k in train_cids])
            transform,fit_details=fit_rotation(support,query,recipe['radius'])
            vectors[cid],apply_details=apply_rotation(original_vectors[cid],transform,recipe['radius'])
            details={**fit_details,**apply_details,'scope':recipe['scope'],'target_conversation':cid,
                     'training_conversations':train_cids,'training_probe_ids':[r['id'] for k in train_cids for r in byconv[k]],
                     'uses_target_questions':cid in train_cids}
            path=out/'transforms'/name/f'{cid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(path,**transform)
''' +source[end:]
    change("'gradient_training_used_only_q0':True", "'gradient_training_used_only_q0':True,'fit_uses_only_q0':True,'no_iterative_gradient_training':True")
    change("print('INDEX_SOURCE '","print('TRANSFER_SOURCE '")
    change("print('INDEX_DEV '","print('TRANSFER_DEV '")
    output=here/'run_transfer_refinement.py'
    if output.exists():
        assert output.read_text(encoding='utf-8')==source
    else:
        output.write_text(source,encoding='utf-8',newline='\n')
    print(output)


if __name__=='__main__':main()
