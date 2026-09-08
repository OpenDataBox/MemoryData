"""Same reader/RRF/read cap, with optional constructor-defined retrieval keys.

Legacy memory goes through the original evaluator without any behavioral change.
New memory indexes index_text but supplies text as evidence to the reader.
Both logical text fields count toward storage cost when they differ.
"""
from pathlib import Path
import re

import refine as core


def evaluate(rt,memories,records,samples_by_id,budget,name,outdir):
    if not any('index_text' in u for units in memories.values() for u in units):
        return core.evaluate(rt,memories,records,samples_by_id,budget,name,outdir)
    import numpy as np
    from rank_bm25 import BM25Okapi
    prompts,metadata=[],[]
    for cid in sorted({r['conv_id'] for r in records}):
        selected_records=[r for r in records if r['conv_id']==cid]
        units=memories[cid]
        indexes=[u.get('index_text',u['text']) for u in units]
        dense=rt.encode(indexes)
        bm25=BM25Okapi([core.lexical(t) or ['_empty'] for t in indexes])
        questions=[samples_by_id[cid]['qa'][r['qa_index']]['question'] for r in selected_records]
        queries=rt.encode(questions,query=True)
        storage=sum(rt.ntok(u['text'])+(rt.ntok(index) if index!=u['text'] else 0) for u,index in zip(units,indexes))
        for record,question,query in zip(selected_records,questions,queries):
            def ranks(scores):
                order=np.argsort(-scores,kind='stable')
                positions=np.empty(len(order),dtype=np.int64)
                positions[order]=np.arange(len(order))
                return positions
            dense_scores=dense@query
            sparse_scores=np.asarray(bm25.get_scores(core.lexical(question)))
            fused=1/(60+ranks(dense_scores))+1/(60+ranks(sparse_scores))
            order=np.argsort(-fused,kind='stable')[:min(len(units),120)]
            context,hits,tokens=core.pack(units,order,rt.ntok,budget)
            prompts.append(f'Conversation memory:\n{context}\n\nQuestion: {question}\nAnswer:')
            metadata.append({**record,'question':question,'context':context,'read_tokens':tokens,
                             'source_ids':sorted({sid for i in hits for sid in units[i]['sources']}),
                             'memory_tokens':storage,'memory_units':len(units),
                             'index_units':sum('index_text' in u for u in units)})
    predictions=rt.generate(core.READER,prompts,max_tokens=96)
    rows=[]
    for row,prediction in zip(metadata,predictions):
        qa=samples_by_id[row['conv_id']]['qa'][row['qa_index']]
        gold=str(qa['answer'])
        evidence=[str(e) for e in qa.get('evidence',[]) if re.fullmatch(r'D\d+:\d+',str(e))]
        hit=len(set(evidence)&set(row['source_ids']))
        rows.append({**row,'candidate':name,'prediction':prediction,'gold':gold,
                     'official_f1':core.f1(prediction,gold,int(qa['category'])),
                     'generic_token_f1':core.generic_f1(prediction,gold),
                     'cited_source_recall':hit/len(set(evidence)) if evidence else None,'gold_evidence':evidence})
    splits={r['split'] for r in records}
    split=next(iter(splits)) if len(splits)==1 else 'all'
    core.save(Path(outdir)/f'{name}_{split}_items.json',rows)
    return rows
