"""Wait for the full-fit source diagnostic, then reuse the one GPU."""
import json
from pathlib import Path
import subprocess
import time


def call(*args,allow_status=False):
    result=subprocess.run(args,text=True,capture_output=True,timeout=30)
    if result.returncode and not allow_status:
        raise RuntimeError(result.stdout+result.stderr)
    return result.stdout.strip()


def main():
    root=Path('/workspace/locomo-refinement')
    while True:
        status=json.loads((root/'runs/evidence_utility_fit_v1/status.json').read_text())
        live=call('supervisorctl','status','locomo-evidence-fit',allow_status=True)
        if status['phase']=='fit_complete':
            if 'RUNNING' in live or 'STARTING' in live:
                print(call('supervisorctl','stop','locomo-evidence-fit'),flush=True)
            elif 'EXITED' not in live:
                raise RuntimeError('Unexpected completed-fit supervisor state: '+live)
            break
        if 'RUNNING' not in live and 'STARTING' not in live:
            raise RuntimeError('Source-fit diagnostic stopped unexpectedly: '+live)
        print('WAIT_FOR_SOURCE_FIT '+str(status.get('completed',0))+'/'+str(status.get('total',278)),flush=True)
        time.sleep(10)
    print(call('supervisorctl','start','locomo-budgeted-refinement'),flush=True)
    print(call('supervisorctl','status','locomo-budgeted-refinement'),flush=True)
    (root/'budgeted_transition.json').write_text(json.dumps({'fit_completed':status['completed'],
                                                          'transitioned_at':time.time()},indent=2))


if __name__=='__main__':
    main()
