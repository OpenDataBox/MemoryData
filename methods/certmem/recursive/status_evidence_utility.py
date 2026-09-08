"""Compact source-fit progress; audit outcomes are deliberately not displayed."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    ap.add_argument('--examples',type=int,default=0)
    ap.add_argument('--worst',action='store_true')
    args=ap.parse_args()
    status=json.loads((args.run/'status.json').read_text())
    output={'status':status}
    if (args.run/'summary.json').exists():
        output['source_fit_summary']=json.loads((args.run/'summary.json').read_text())['probe_fit']
    if args.examples:
        rows=[json.loads(p.read_text()) for p in (args.run/'items').glob('*.json')]
        rows=[r for r in rows if r['split']=='probe_fit' and 'analysis' in r]
        if args.worst:
            rows=[r for r in rows if r['generations']['full']['source_answer_f1']>=0.8]
            rows.sort(key=lambda r:r['generations']['minimum_bundle']['source_answer_f1']-r['generations']['full']['source_answer_f1'])
        else:
            rows.sort(key=lambda r:r['analysis']['best_pair_lift_over_best_single'],reverse=True)
        output['source_fit_examples']=[{'id':r['id'],'question':r['question'],'answer':r['answer'],
             'analysis':{k:v for k,v in r['analysis'].items() if k!='pair_interactions'},
             'generations':r['generations']} for r in rows[:args.examples]]
    print(json.dumps(output))


if __name__=='__main__':
    main()
