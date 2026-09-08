"""Supervised handoff after the current immutable candidate batch has finished."""
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
    source=root/'runs/continuous_v2'
    while True:
        live=command('supervisorctl','status','locomo-continuous-v2')
        if 'RUNNING' not in live:
            raise RuntimeError('Parent is not running; inspect it before attempting transition: '+live)
        status=json.loads((source/'status.json').read_text())
        history=json.loads((source/'history.json').read_text())
        queue=json.loads((root/'continuous_v2_queue.json').read_text())
        completed={r['name'] for r in history['rounds']}
        if status['phase']=='awaiting_candidates' and completed=={r['name'] for r in queue['recipes']}:
            break
        print('WAIT_FOR_CURRENT_BATCH '+str(len(completed))+'/'+str(len(queue['recipes'])),flush=True)
        time.sleep(10)
    print(command('supervisorctl','stop','locomo-continuous-v2'),flush=True)
    print(command('supervisorctl','start','locomo-continuous-v3'),flush=True)
    print(command('supervisorctl','status','locomo-continuous-v3'),flush=True)
    (root/'continuous_v3_transition.json').write_text(json.dumps({'parent_completed':len(completed),
         'previous_best':history['winner'],'previous_best_f1':history['best_f1'],'transitioned_at':time.time()},indent=2))


if __name__=='__main__':
    main()
