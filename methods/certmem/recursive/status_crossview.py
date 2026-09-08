"""Compact cross-view progress; audit memory outcomes only after completion."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    read=lambda p:json.loads(p.read_text())
    status=read(args.run/'status.json')
    result={'status':status}
    for name,key in [('reference_gate.json','reference_gate'),('source_history.json','source_history'),
                     ('query_view_lock.json','view_lock')]:
        if (args.run/name).exists():
            result[key]=read(args.run/name)
    quality=args.run/'query_views/quality.json'
    if quality.exists():
        q=read(quality)
        result['quality']={'input_probes':q['input_probes'],'paired_valid':q['paired_valid'],
                           'structure_rejections':len(q['rejected_structure'])}
    for name,key in [('source_baseline_contract.json','baseline_fit'),('previous_method_fit_contract.json','previous_method_fit')]:
        if (args.run/name).exists():
            values=read(args.run/name)
            result[key]={'total':len(values),'preserved':sum(v['preserved'] for v in values.values())}
    result['trials']=[]
    for path in sorted((args.run/'source_trials').glob('*.json')):
        row=read(path)['result']
        result['trials'].append({'name':row['name'],'repairs':len(row['repairs']),
                                 'regressions':len(row['regressions']),'feasible':row['feasible']})
    if status['phase']=='controlled_batch_complete':
        result['view_audit']=read(args.run/'view_audit_summary.json')
        result['dev_history']=read(args.run/'history.json')
    print(json.dumps(result))


if __name__=='__main__':
    main()
