"""Compact progress only; never display benchmark questions or predictions."""
import json
import argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path,nargs='?',default=Path('runs/source_index_v1'))
    root=ap.parse_args().run
    names=('status.json','reference_gate.json','training_locked.json','source_selection_locked.json','history.json')
    result={}
    for name in names:
        path=root/name
        if not path.exists():
            continue
        row=json.loads(path.read_text())
        if name=='training_locked.json':
            row={'locked_at':row['locked_at'],'indexes':list(row['all_indexes_sha256']),
                 'gradient_training_used_only_q0':row['gradient_training_used_only_q0']}
        if name=='source_selection_locked.json':
            row={k:row[k] for k in ('locked_at','selected','result')}
        if name=='history.json':
            row={'winner':row['winner'],'best_f1':row['best_f1'],'rounds':[
                {'name':r['name'],'f1':r['dev']['official_f1'],'accepted':r['accepted']} for r in row['rounds']]}
        result[name]=row
    result['optimized_conversations']=len(list((root/'optimizer').glob('*/*.json')))
    result['source_trials']=[{k:v for k,v in json.loads(p.read_text())['result'].items() if k not in ('q0','fit_a')}
                             for p in sorted((root/'source_trials').glob('*.json'))]
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
