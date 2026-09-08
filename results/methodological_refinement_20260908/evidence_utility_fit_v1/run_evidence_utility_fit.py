"""Expand the unchanged joint-evidence policy to all source-fit probes only."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import refine as core
from evidence_utility import POLICY,select_probes,subset_jobs,subset_analysis,generation_controls
from run_evidence_utility import Scorer,read,summarize


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sources',type=Path,default=Path('research_data/source_sessions.json'))
    ap.add_argument('--probes',type=Path,default=Path('research_data/source_probes.json'))
    ap.add_argument('--environment',type=Path,default=Path('environment.json'))
    ap.add_argument('--parent',type=Path,default=Path('runs/evidence_utility_v1'))
    ap.add_argument('--out',type=Path,default=Path('runs/evidence_utility_fit_v1'))
    args=ap.parse_args()
    corpus,pool,environment=read(args.sources),read(args.probes),read(args.environment)
    assert set(corpus)=={'dataset_sha256','sessions_by_id','excluded_holdout_conversations'}
    assert not set(corpus['sessions_by_id']).intersection(corpus['excluded_holdout_conversations'])
    parent=read(args.parent/'protocol.json')
    assert read(args.parent/'status.json')['phase']=='pilot_complete'
    assert core.digest(pool)==parent['probe_pool_sha256']
    assert core.digest(corpus)==parent['source_corpus_sha256']
    assert POLICY==parent['policy'] and environment==parent['environment']
    for name,value in parent['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    selection_policy={**POLICY,'fit_per_conversation':len(pool['records']),'audit_per_conversation':0}
    selected=select_probes(pool,selection_policy)
    assert all(p['split']=='probe_fit' for p in selected)
    assert len(selected)==sum(p['split']=='probe_fit' for p in pool['records'])
    protocol={**parent,'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ('run_evidence_utility_fit.py','run_evidence_utility.py','evidence_utility.py','refine.py')},
              'parent_protocol_sha256':core.digest(parent),'selection_policy':selection_policy,
              'selected_ids':[p['id'] for p in selected],
              'purpose':'Expanded source-fit evidence diagnostic; identical utility policy and runtime; no audit/benchmark outcomes used.'}
    if (args.out/'protocol.json').exists():
        assert read(args.out/'protocol.json')==protocol
    core.save(args.out/'protocol.json',protocol)
    core.save(args.out/'selection.json',[{k:p[k] for k in ('id','conv_id','session','split')} for p in selected])
    for path in (args.parent/'cache').rglob('*.json'):
        target=args.out/'cache'/path.relative_to(args.parent/'cache')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            os.link(path,target)
    core.save(args.out/'status.json',{'phase':'loading','pid':os.getpid(),'heartbeat_at':time.time()})
    scorer=Scorer(environment,args.out)
    rows=[]
    for probe in selected:
        path=args.out/'items'/(probe['id'].replace(':','_')+'.json')
        if path.exists():
            rows.append(read(path))
            continue
        core.save(args.out/'status.json',{'phase':'scoring','pid':os.getpid(),'probe_id':probe['id'],
                                          'completed':len(rows),'total':len(selected),'heartbeat_at':time.time()})
        turns={t['id']:t for s in corpus['sessions_by_id'][probe['conv_id']] for t in s['turns']}
        jobs,skip=subset_jobs(probe,turns,scorer.ntok,POLICY['read_budget'])
        if skip:
            row={**probe,'skip':skip}
        else:
            scorer.score(jobs)
            analysis=subset_analysis(jobs)
            row={**probe,'analysis':analysis,'generations':scorer.generate(generation_controls(jobs,analysis)),
                 'subsets':[{k:v for k,v in job.items() if k not in ('user','answer')} for job in jobs]}
        core.save(path,row)
        rows.append(row)
        core.save(args.out/'summary.json',summarize(rows))
        print('FIT_PROBE '+str(len(rows))+'/'+str(len(selected))+' '+probe['id'],flush=True)
    core.save(args.out/'summary.json',summarize(rows))
    core.save(args.out/'status.json',{'phase':'fit_complete','pid':os.getpid(),'completed':len(rows),
                                    'total':len(selected),'heartbeat_at':time.time(),
                                    'next':'Use fit evidence to develop budgeted construction; user goal remains active.'})


if __name__=='__main__':
    main()
