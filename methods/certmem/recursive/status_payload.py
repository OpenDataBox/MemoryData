"""Compact state and aggregate compiler counts; no benchmark QA output."""
import json
import argparse
from pathlib import Path


def read(path):return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path,nargs='?',default=Path('runs/provenance_payload_v2'))
    root=ap.parse_args().run;result={}
    for name in ('status.json','reference_gate.json','source_payloads_locked.json','payload_memories_locked.json','source_selection_locked.json','history.json'):
        if not (root/name).exists():continue
        row=read(root/name)
        if name=='history.json':row={'winner':row['winner'],'best_f1':row['best_f1'],'rounds':[{'name':r['name'],'f1':r['dev']['official_f1'],'accepted':r['accepted']} for r in row['rounds']]}
        if name=='source_selection_locked.json':row={k:row[k] for k in ('locked_at','selected')}
        result[name]=row
    compiled=[]
    for path in sorted((root/'source_payloads').glob('*.json')):
        data=read(path);facts=[f for item in data.values() for f in item['facts']]
        compiled.append({'conv_id':path.stem,'groups':len(data),'literal':len(facts),'supported':sum(f['supported'] for f in facts),
                         'deletion_sensitive':sum(f['deletion_sensitive'] for f in facts),'rejected_records':sum(len(i['rejected']) for i in data.values()),
                         'final_complete_json':sum(isinstance(i['generation']['object'],dict) for i in data.values()),
                         'retried_groups':sum(len(i['generation'].get('attempts',[]))>1 for i in data.values())})
    result['completed_source_compilations']=compiled
    result['writer_cache_files']=len(list((root/'cache/provenance_payload_generations').glob('*.json')))
    generated=[read(p) for p in (root/'cache/provenance_payload_generations').glob('*.json')]
    result['generation_quality']={'parseable':sum(isinstance(v['object'],dict) for v in generated),
        'token_limit_stops':sum(v['finish_reason']=='length' for v in generated),
        'emitted_facts':sum(len(v['object'].get('facts',[])) for v in generated if isinstance(v['object'],dict))}
    result['source_trials']=[{k:v for k,v in read(p)['result'].items() if k not in ('q0','fit_a')} for p in sorted((root/'source_trials').glob('*.json'))]
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
