"""Start the frozen audit checkpoint only after the identity GPU worker exits."""
import json
from pathlib import Path
import subprocess
import time


def call(*args,allow_status=False):
    result=subprocess.run(args,capture_output=True,text=True,timeout=30)
    if result.returncode and not allow_status:
        raise RuntimeError(result.stdout+result.stderr)
    return result.stdout.strip()


def main():
    root=Path('/workspace/locomo-refinement')
    while True:
        status=json.loads((root/'runs/identity_refinement_v1/status.json').read_text())
        live=call('supervisorctl','status','locomo-identity-refinement',allow_status=True)
        if status['phase']=='controlled_batch_complete' and 'EXITED' in live:
            break
        if 'RUNNING' not in live and 'STARTING' not in live:
            raise RuntimeError('Identity worker stopped unexpectedly: '+live)
        time.sleep(10)
    # Never load a second GPU engine; any residual user compute blocks handoff.
    active=call('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader')
    if active:
        raise RuntimeError('GPU still has compute processes; wait for confirmed exit: '+active)
    print(call('supervisorctl','start','locomo-portable-audit'),flush=True)
    (root/'portable_audit_transition.json').write_text(json.dumps({'transitioned_at':time.time(),
       'previous_run_completed':True,'policy_unchanged_by_identity_results':True},indent=2))


if __name__=='__main__':
    main()
