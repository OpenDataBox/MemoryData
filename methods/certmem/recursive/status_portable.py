"""Compact checkpoint state; print audit scores only after all methods finish."""
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
def read(name):
    p=root/name
    return json.loads(p.read_text()) if p.exists() else None
status=read('status.json')
data={'status':status,'reference_gate':read('reference_gate.json'),
      'memory_selection_locked':bool(read('memory_selection_locked.json')),
      'source_items_completed':len(list((root/'utility/items').glob('*.json')))}
if status and status['phase']=='audit_checkpoint_complete':
    data['completed_audit']=read('audit_results.json')
print(json.dumps(data))
