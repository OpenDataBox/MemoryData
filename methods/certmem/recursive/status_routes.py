"""Show route-construction progress and scores without QA text."""
import json
import sys
from pathlib import Path


def read(p):return json.loads(p.read_text())


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/coverage_routes_v1');result={}
    for name in ('status.json','reference_gate.json','routes_locked.json','route_memories_locked.json','source_selection_locked.json','history.json'):
        if not (root/name).exists():continue
        value=read(root/name)
        if name=='history.json':value={'winner':value['winner'],'best_f1':value['best_f1'],'rounds':[{'name':r['name'],'f1':r['dev']['official_f1'],'accepted':r['accepted']} for r in value['rounds']]}
        if name=='source_selection_locked.json':value={k:value[k] for k in ('locked_at','selected')}
        result[name]=value
    result['route_pools']=[{'conv_id':p.stem,'supported':len(read(p)),'dependent':sum(c['deletion_sensitive'] for c in read(p))} for p in sorted((root/'route_candidates').glob('*.json'))]
    result['constructed_memories']=len(list((root/'construction').glob('*/*.json')))
    result['source_trials']=[{k:v for k,v in read(p)['result'].items() if k not in ('q0','fit_a')} for p in sorted((root/'source_trials').glob('*.json'))]
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
