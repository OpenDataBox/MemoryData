"""Archive only committed results while a refinement worker continues running."""
import argparse
import io
import json
from pathlib import Path
import tarfile


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run_dir',type=Path)
    ap.add_argument('archive',type=Path)
    ap.add_argument('--since',type=int,default=0,help='Include only rounds after this history index; extract over an earlier snapshot')
    args=ap.parse_args()
    history=json.loads((args.run_dir/'history.json').read_text())
    protocol=json.loads((args.run_dir/'protocol.json').read_text())
    completed=[r['name'] for r in history['rounds'][args.since:] if 'dev' in r]
    temp=args.archive.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz') as archive:
        payload=json.dumps(history,indent=2).encode()
        info=tarfile.TarInfo('run/history.json')
        info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
        payload=json.dumps({'total_rounds':len(history['rounds']),'delta_since':args.since,'included':completed}).encode()
        info=tarfile.TarInfo('snapshot_metadata.json')
        info.size=len(payload)
        archive.addfile(info,io.BytesIO(payload))
        for name in ['protocol.json','status.json','seed_dev_items.json','seed_diagnostics.json']:
            archive.add(args.run_dir/name,arcname='run/'+name)
        for name in completed:
            for suffix in ['_dev_items.json','_diagnostics.json']:
                archive.add(args.run_dir/(name+suffix),arcname='run/'+name+suffix)
            archive.add(args.run_dir/'plans'/f'{name}.json',arcname=f'run/plans/{name}.json')
            archive.add(args.run_dir/'memories'/name,arcname=f'run/memories/{name}')
        names=set(protocol['source_sha256'])|{Path(protocol['args']['queue']).name,'environment.json',
                                            'run_continuous_4080s.sh','run_continuous_v2_4080s.sh','run_continuous_v3_4080s.sh'}
        for name in sorted(names):
            if not Path(name).is_file():
                continue
            archive.add(name,arcname='source/'+name)
    temp.replace(args.archive)
    print(json.dumps({'winner':history['winner'],'best_f1':history['best_f1'],
                      'completed':len(completed),'archive_bytes':args.archive.stat().st_size}))


if __name__=='__main__':
    main()
