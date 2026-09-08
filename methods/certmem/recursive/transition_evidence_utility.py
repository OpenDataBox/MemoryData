"""Handoff the one GPU to the source-only experiment once all controls finish."""
import json
from pathlib import Path
import subprocess
import time


def command(*args):
    result=subprocess.run(args,text=True,capture_output=True,timeout=30)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return result.stdout.strip()


def main():
    root=Path('/workspace/locomo-refinement')
    source=root/'runs/continuous_v3'
    while True:
        live=command('supervisorctl','status','locomo-continuous-v3')
        if 'RUNNING' not in live:
            raise RuntimeError('Inspect parent before transition: '+live)
        status=json.loads((source/'status.json').read_text())
        history=json.loads((source/'history.json').read_text())
        queue=json.loads((root/'continuous_v3_queue.json').read_text())
        completed={r['name'] for r in history['rounds']}
        if status['phase']=='awaiting_candidates' and completed=={r['name'] for r in queue['recipes']}:
            break
        print('WAIT_FOR_CONTROLS '+str(len(completed))+'/'+str(len(queue['recipes'])),flush=True)
        time.sleep(10)
    assert not (source/'reference_gate_failed.json').exists()
    print(command('supervisorctl','stop','locomo-continuous-v3'),flush=True)
    print(command('supervisorctl','start','locomo-evidence-utility'),flush=True)
    print(command('supervisorctl','status','locomo-evidence-utility'),flush=True)
    (root/'evidence_utility_transition.json').write_text(json.dumps({'parent_completed':len(completed),
           'previous_best':history['winner'],'previous_best_f1':history['best_f1'],'transitioned_at':time.time()},indent=2))


if __name__=='__main__':
    main()
