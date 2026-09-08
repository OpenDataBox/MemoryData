import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
def read(name):
    p=root/name
    return json.loads(p.read_text()) if p.exists() else None
data={name:read(name+'.json') for name in ('status','reference_gate','source_selection_locked','history')}
base=read('source_baseline_contract.json')
if base:
    data['source_baseline']={k:sum(r['preserved'] for r in v.values()) for k,v in base.items()}
data['trials']=[]
for p in sorted((root/'source_trials').glob('*.json')):
    r=json.loads(p.read_text())['result']
    data['trials'].append({'name':r['name'],'q0_repair':len(r['q0']['repairs']),
        'q0_regressions':len(r['q0']['regressions']),'a_repair':len(r['fit_a']['repairs']),
        'a_regressions':len(r['fit_a']['regressions']),'feasible':r['feasible']})
print(json.dumps(data))
