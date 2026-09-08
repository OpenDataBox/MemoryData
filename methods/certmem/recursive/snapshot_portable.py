"""Capture an immutable completed new-question checkpoint with source evidence."""
import argparse
from pathlib import Path
import json
import tarfile


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    ap.add_argument('archive',type=Path)
    args=ap.parse_args()
    protocol=json.loads((args.run/'protocol.json').read_text())
    assert json.loads((args.run/'status.json').read_text())['phase']=='audit_checkpoint_complete'
    with tarfile.open(args.archive.with_suffix('.tmp'),'w:gz') as archive:
        for name in ('protocol.json','status.json','source_sessions.json','source_generations.json','source_probes.json',
                     'utility_selection_locked.json','memory_selection_locked.json','reference_gate.json',
                     'reference_replay_dev_items.json','memories','construction','options','utility/items','audit_results.json'):
            archive.add(args.run/name,arcname='run/'+name)
        for p in sorted(args.run.glob('*_question_audit_items.json')):
            archive.add(p,arcname='run/'+p.name)
        for name in protocol['source_sha256']:
            archive.add(Path(name),arcname='source/'+name)
    args.archive.with_suffix('.tmp').replace(args.archive)
    print(json.dumps({'archive_bytes':args.archive.stat().st_size,'completed_audit_questions':100,
                      'methods':protocol['methods']}))


if __name__=='__main__':
    main()
