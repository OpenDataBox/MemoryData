"""Read committed source-only decisions and current worker progress."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    read=lambda p:json.loads(p.read_text())
    result={'status':read(args.run/'status.json')}
    for filename,key in [('reference_gate.json','reference_gate'),('source_history.json','source_history'),
                          ('source_selection_locked.json','selection'),('history.json','history')]:
        if (args.run/filename).exists():
            result[key]=read(args.run/filename)
    if (args.run/'source_baseline_contract.json').exists():
        values=read(args.run/'source_baseline_contract.json')
        result['source_baseline']={'total':len(values),'preserved':sum(v['preserved'] for v in values.values())}
    trials=[]
    for path in sorted((args.run/'source_trials').glob('*.json')):
        row=read(path)['result']
        trials.append({**{k:v for k,v in row.items() if k not in ('regressions','repairs')},
                       'regressions':len(row['regressions']),'repairs':len(row['repairs'])})
    result['source_trials']=trials
    print(json.dumps(result))


if __name__=='__main__':
    main()
