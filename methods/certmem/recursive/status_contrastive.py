import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
def read(name):
    p=root/name
    return json.loads(p.read_text()) if p.exists() else None
result={'status':read('status.json'),'reference_gate':read('reference_gate.json'),
        'selection':read('source_selection_locked.json')}
result['graphs']={}
for p in sorted((root/'event_graph').glob('*.json')):
    value=json.loads(p.read_text())
    result['graphs'][p.stem]={'proposed':len(value['proposal_edges']),'distinct':len(value['candidates']),
                             'supported':sum(c['accepted'] for c in value['candidates'])}
result['trials']=[]
for p in sorted((root/'source_trials').glob('*.json')):
    r=json.loads(p.read_text())['result']
    result['trials'].append({'name':r['name'],'net_q0':r['net_q0'],'net_a':r['net_a'],
                            'worst_view_feasible':r['worst_view_feasible'],'strict_feasible':r['strict_feasible']})
h=read('history.json')
if h:
    result['dev']={'winner':h['winner'],'best_f1':h['best_f1'],
                   'rounds':[{'name':r['name'],'f1':r['dev']['official_f1'],'storage':r['dev']['memory_tokens']} for r in h['rounds']]}
print(json.dumps(result))
