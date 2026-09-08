"""Hand off the one GPU after the frozen checkpoint, without reading its scores."""
import json
from pathlib import Path
import subprocess
import time


def call(*args,allow_status=False):
    r=subprocess.run(args,text=True,capture_output=True,timeout=30)
    if r.returncode and not allow_status:
        raise RuntimeError(r.stdout+r.stderr)
    return r.stdout.strip()


def main():
    root=Path('/workspace/locomo-refinement')
    while True:
        status=json.loads((root/'runs/portable_parent_audit_v1/status.json').read_text())
        live=call('supervisorctl','status','locomo-portable-audit',allow_status=True)
        if status['phase']=='audit_checkpoint_complete' and 'EXITED' in live:
            break
        if 'RUNNING' not in live and 'STARTING' not in live:
            raise RuntimeError('Checkpoint stopped unexpectedly: '+live)
        time.sleep(10)
    active=call('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader')
    if active:
        raise RuntimeError('GPU still occupied: '+active)
    print(call('supervisorctl','start','locomo-contrastive-refinement'),flush=True)
    (root/'contrastive_transition.json').write_text(json.dumps({'transitioned_at':time.time(),
       'read_audit_outcomes':False},indent=2))


if __name__=='__main__':
    main()
