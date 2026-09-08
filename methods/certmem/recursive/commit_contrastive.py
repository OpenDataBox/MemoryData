"""Freeze the next development experiment before the independent audit finishes."""
import hashlib
from pathlib import Path
import time
import refine as core
from run_evidence_utility import read

names=('run_contrastive_refinement.py','contrastive_events.py','retrieval_contract.py',
       'crossview_probes.py','budgeted_evidence.py','evidence_fidelity.py','evidence_utility.py',
       'run_evidence_utility.py','evaluate_indexed.py','refine.py')
source={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
target=Path('contrastive_precommit.json')
if target.exists():
    assert read(target)['source_sha256']==source
else:
    assert not Path('runs/portable_parent_audit_v1/memory_selection_locked.json').exists()
    core.save(target,{'committed_at':time.time(),'source_sha256':source,
              'independent_audit_has_not_reached_memory_lock':True,
              'policy':'Four literal/contrastive event-pair controls at +2000/+4000 common selection budgets; original-source keys; Q0/A matched-75 worst-view and strict choices locked before dev. No independent audit100 result used.'})
print('CONTRASTIVE_PRECOMMIT_FROZEN '+str(read(target)['committed_at']))
