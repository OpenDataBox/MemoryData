"""Stop only the named incomplete-output run, preserve its protocol and caches."""
import json
from pathlib import Path
import subprocess
import time


def main():
    name='locomo-payload-refinement';root=Path('runs/provenance_payload_v1')
    status=subprocess.run(['supervisorctl','status',name],text=True,capture_output=True).stdout
    if 'RUNNING' in status:
        subprocess.run(['supervisorctl','stop',name],check=True)
    final=subprocess.run(['supervisorctl','status',name],text=True,capture_output=True).stdout
    assert 'STOPPED' in final or 'EXITED' in final,final
    marker={'reason':'incomplete_structured_output_length','stopped_at':time.time(),'previous_worker_status':final.strip(),
            'next_run':'runs/provenance_payload_v2','action':'Keep complete initial outputs and all caches; retry only incomplete JSON at larger output caps under a new protocol.'}
    path=root/'protocol_superseded.json'
    if not path.exists():path.write_text(json.dumps(marker,indent=2))
    print(json.dumps(marker))


if __name__=='__main__':main()
