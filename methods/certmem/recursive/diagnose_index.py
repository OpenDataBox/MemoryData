"""Aggregate source-objective versus downstream changes; never print QA text."""
import argparse
import json
from pathlib import Path
import refine as core


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    base={r['id']:r for r in read(args.run/'reference_replay_dev_items.json')}
    history=read(args.run/'history.json');results=[]
    for step in history['rounds']:
        name=step['name'];rows=read(args.run/f'{name}_dev_items.json')
        stats=[read(p) for p in (args.run/'optimizer'/name).glob('*.json')]
        n=sum(s['training_queries'] for s in stats)
        d={'name':name,'f1':step['dev']['official_f1'],
           'changed_contexts':sum(r['context']!=base[r['id']]['context'] for r in rows),
           'changed_predictions':sum(r['prediction']!=base[r['id']]['prediction'] for r in rows),
           'f1_improvements':sum(r['official_f1']>base[r['id']]['official_f1']+1e-12 for r in rows),
           'f1_regressions':sum(r['official_f1']<base[r['id']]['official_f1']-1e-12 for r in rows),
           'cited_source_recall':step['dev']['cited_source_recall'],
           'f1_delta_points':100*(step['dev']['official_f1']-core.summarize(list(base.values()))['official_f1'])}
        if 'initial_loss' in stats[0]:
            d['source_weighted_initial_loss']=sum(s['initial_loss']*s['training_queries'] for s in stats)/n
            d['source_weighted_final_loss']=sum(s['final_loss']*s['training_queries'] for s in stats)/n
        else:
            d['fit_mean_cosine_before']=sum(s['source_mean_cosine_before']*s['training_queries'] for s in stats)/n
            d['fit_mean_cosine_after']=sum(s['source_mean_cosine_after']*s['training_queries'] for s in stats)/n
            d['maximum_document_geometry_error']=max(s['maximum_pairwise_inner_product_error'] for s in stats)
        results.append(d)
    core.save(args.out,{'baseline':core.summarize(list(base.values())),'comparisons':results,
                        'limits':'Aggregate development diagnostics only. These are already exposed development questions, not new audit results.'})
    print(json.dumps(results))


if __name__=='__main__':main()
