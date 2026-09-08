"""Export only development conversations, stripping benchmark QA at the boundary."""
import argparse
import json
from pathlib import Path
import refine as core


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    samples=json.loads(args.data.read_text(encoding='utf-8'))
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    assert core.digest(samples)==manifest['dataset_sha256']
    allowed={r['conv_id'] for r in manifest['records'] if r['split']=='dev'}
    assert not allowed.intersection(manifest['holdout_convs'])
    value={'dataset_sha256':manifest['dataset_sha256'],
           'excluded_holdout_conversations':manifest['holdout_convs'],
           'sessions_by_id':{str(s['sample_id']):core.session_data(s) for s in samples if str(s['sample_id']) in allowed}}
    if args.out.exists() and json.loads(args.out.read_text(encoding='utf-8'))!=value:
        raise ValueError('Source corpus is frozen; use a new version')
    core.save(args.out,value)
    print(json.dumps({'conversations':len(allowed),'sessions':sum(map(len,value['sessions_by_id'].values())),
                      'qa_fields_exported':0,'digest':core.digest(value)}))


if __name__=='__main__':
    main()
