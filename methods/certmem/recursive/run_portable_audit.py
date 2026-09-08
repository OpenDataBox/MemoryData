"""Three sequential GPU processes: source prep, unchanged NLL, locked audit.

The policy is the already selected parent_single_2000. Ongoing identity-control
scores never select or alter the policy for this separately reserved audit.
"""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import time

import refine as core
from evaluate_indexed import evaluate
from budgeted_evidence import storage_cost
from portable_parent import backend,augment,generate_source_probes
from evidence_utility import POLICY,subset_jobs,subset_analysis,generation_controls
from run_evidence_utility import Scorer,read

NAMES=('seed','r40_fused_four_turn','s_parent_single_2000')
SOURCES=('run_portable_audit.py','portable_parent.py','continuous_v2.py','memory_ops.py','continuous.py',
         'parent_evidence.py','budgeted_evidence.py','compile_research_data.py','evidence_fidelity.py',
         'evidence_utility.py','run_evidence_utility.py','evaluate_indexed.py','refine.py')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('prepare','score','evaluate'))
    ap.add_argument('--out',default='runs/portable_parent_audit_v1')
    ap.add_argument('--manifest',default='research_data/question_audit100_manifest.json')
    ap.add_argument('--model',default='Qwen/Qwen3-8B')
    ap.add_argument('--embed-model',default='Qwen/Qwen3-Embedding-0.6B')
    ap.add_argument('--embed-batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--budget',type=int,default=2048)
    args=ap.parse_args()
    out=Path(args.out)
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in SOURCES}
    old=Path('runs/pilot100_v3')
    old_manifest=read(old/'manifest.json')
    audit=read(args.manifest)
    assert audit['dataset_sha256']==old_manifest['dataset_sha256']
    ids={r['id'] for r in audit['records']}
    assert len(ids)==len(audit['records'])==100 and not ids.intersection(r['id'] for r in old_manifest['records'])
    cids=set(audit['conversations'])
    assert cids==set(old_manifest['holdout_convs'])=={r['conv_id'] for r in audit['records']}
    environment=read('environment.json')
    parent_protocol=read('runs/parent_evidence_v1/protocol.json')
    for name,value in parent_protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    plans={name:read(Path('runs/continuous_v3/plans')/f'{name}.json') for name in
           ('r06_calendar_month','r12_filter_current_best','r40_fused_four_turn','r24_source_qa_cards')}
    for plan in plans.values():
        for name,value in plan['source_sha256'].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==value
    config={k:v for k,v in vars(args).items() if k!='stage'}
    protocol={'args':config,'source_sha256':source,'environment':environment,'audit_manifest':audit,
              'parent_protocol_sha256':core.digest(parent_protocol),'plans':plans,'utility_policy':POLICY,
              'methods':list(NAMES),'frozen_method':'s_parent_single_2000','frozen_method_dev_f1':.5927437370002835,
              'policy':'Rebuild r06/r12/r40 from original seed, then identical source-QA prompt, literal span compiler, 2 probes/session with same seeded fit/session-audit split, all source-fit utility scores, and unverified parent_single +2000 budget. No source threshold adjustment.',
              'boundaries':'Audit question/answer text never enters preparation, utility scoring or construction. All three memory collections locked before new-question evaluation. Original holdout conversations were exposed in initial pilot; only these 100 question indices are new. Results must not be used for further tuning. Current identity-control outcomes do not change this frozen checkpoint.',
              'limits':'Same reader and three previously exposed conversations, not external generalization. This evaluates the already fixed parent policy, even if ongoing development later finds a different best. Stored source probe IDs depend on new source pool; sampling function and seed remain fixed.'}
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json')==protocol
    core.save(out/'protocol.json',protocol)
    status={'stage':args.stage,'pid':os.getpid(),'started_at':time.time()}
    def state(phase,**fields):
        core.save(out/'status.json',{**status,'phase':phase,'heartbeat_at':time.time(),**fields})
    if args.stage=='prepare':
        assert not (out/'memory_selection_locked.json').exists()
        # Strip benchmark QA at the boundary, before any construction function.
        samples=read('data/locomo10.json')
        assert core.digest(samples)==audit['dataset_sha256']
        sessions={str(s['sample_id']):core.session_data(s) for s in samples if str(s['sample_id']) in cids}
        dev_cids={r['conv_id'] for r in old_manifest['records'] if r['split']=='dev'}
        dev={str(s['sample_id']):s for s in samples if str(s['sample_id']) in dev_cids}
        del samples
        core.save(out/'source_sessions.json',{'dataset_sha256':audit['dataset_sha256'],'sessions_by_id':sessions})
        for p in Path('runs/identity_refinement_v1/cache').rglob('*'):
            if p.is_file() and p.suffix in ('.json','.npy'):
                target=out/'cache'/p.relative_to('runs/identity_refinement_v1/cache')
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    os.link(p,target)
        state('reference_model_load')
        rt=core.Runtime(args,environment)
        seed_name=read(old/'selection_locked.json')['winner']
        fit_by_conv=defaultdict(list)
        for p in Path('runs/evidence_utility_fit_v1/items').glob('*.json'):
            row=read(p)
            fit_by_conv[row['conv_id']].append(row)
        reproduced={}
        for cid in sorted(dev_cids):
            source_sessions=core.session_data(dev[cid])
            base=backend(rt,source_sessions,read(old/'memories'/seed_name/f'{cid}.json'),plans)
            assert base==read(Path('runs/continuous_v3/memories/r40_fused_four_turn')/f'{cid}.json')
            reproduced[cid],_,_=augment(rt,source_sessions,base,fit_by_conv[cid])
            assert reproduced[cid]==read(Path('runs/parent_evidence_v1/memories/s_parent_single_2000')/f'{cid}.json')
        replay=evaluate(rt,reproduced,[r for r in old_manifest['records'] if r['split']=='dev'],dev,2048,'reference_replay',out)
        assert abs(core.summarize(replay)['official_f1']-protocol['frozen_method_dev_f1'])<1e-12
        core.save(out/'reference_gate.json',{'passed':True,'expected_f1':protocol['frozen_method_dev_f1'],
                  'r40_and_parent_memory_reconstructed_exactly_for_dev_conversations':sorted(dev_cids)})
        del dev,reproduced
        for cid in sorted(cids):
            state('source_backend_construction',conversation=cid)
            seed=read(old/'memories'/seed_name/f'{cid}.json')
            core.save(out/'memories/seed'/f'{cid}.json',seed)
            core.save(out/'memories/r40_fused_four_turn'/f'{cid}.json',backend(rt,sessions[cid],seed,plans))
        state('source_question_generation')
        dump,pool=generate_source_probes(rt,sessions,plans['r24_source_qa_cards']['recipe']['operations'][0])
        core.save(out/'source_generations.json',dump)
        core.save(out/'source_probes.json',pool)
        state('source_preparation_complete',source_fit=sum(r['split']=='probe_fit' for r in pool['records']),
              source_session_audit_unused=sum(r['split']=='probe_audit' for r in pool['records']))
    elif args.stage=='score':
        assert read(out/'status.json')['phase']=='source_preparation_complete' or read(out/'status.json')['stage']=='score'
        corpus,pool=read(out/'source_sessions.json'),read(out/'source_probes.json')
        selected=sorted([r for r in pool['records'] if r['split']=='probe_fit'],key=lambda r:r['id'])
        assert {r['conv_id'] for r in selected}==cids
        core.save(out/'utility_selection_locked.json',{'locked_at':time.time(),'policy':POLICY,
                  'source_corpus_sha256':core.digest(corpus),'probe_pool_sha256':core.digest(pool),'selected_ids':[r['id'] for r in selected]})
        state('utility_model_load')
        scorer=Scorer(environment,out/'utility')
        for index,probe in enumerate(selected):
            path=out/'utility/items'/(probe['id'].replace(':','_')+'.json')
            if path.exists():
                continue
            state('source_utility_scoring',completed=index,total=len(selected))
            turns={t['id']:t for s in corpus['sessions_by_id'][probe['conv_id']] for t in s['turns']}
            jobs,skip=subset_jobs(probe,turns,scorer.ntok,POLICY['read_budget'])
            if skip:
                row={**probe,'skip':skip}
            else:
                scorer.score(jobs)
                analysis=subset_analysis(jobs)
                row={**probe,'analysis':analysis,'generations':scorer.generate(generation_controls(jobs,analysis)),
                     'subsets':[{k:v for k,v in j.items() if k not in ('user','answer')} for j in jobs]}
            core.save(path,row)
            print(f'PORTABLE_SOURCE_UTILITY {index+1}/{len(selected)}',flush=True)
        state('source_utility_complete',completed=len(selected))
    else:
        assert read(out/'status.json')['phase']=='source_utility_complete' or read(out/'status.json')['stage']=='evaluate'
        corpus=read(out/'source_sessions.json')
        rows=defaultdict(list)
        for p in sorted((out/'utility/items').glob('*.json')):
            row=read(p)
            rows[row['conv_id']].append(row)
        sessions=corpus['sessions_by_id']
        state('construction_model_load')
        rt=core.Runtime(args,environment)
        memories={name:{cid:read(out/'memories'/name/f'{cid}.json') for cid in cids} for name in NAMES[:2]}
        memories[NAMES[2]]={}
        for cid in sorted(cids):
            state('source_parent_construction',conversation=cid)
            memory,details,options=augment(rt,sessions[cid],memories[NAMES[1]][cid],rows[cid])
            memories[NAMES[2]][cid]=memory
            core.save(out/'memories'/NAMES[2]/f'{cid}.json',memory)
            core.save(out/'construction'/NAMES[2]/f'{cid}.json',details)
            core.save(out/'options'/f'{cid}.json',options)
        locked={'methods':list(NAMES),
                  'protocol_sha256':core.digest(protocol),'memories_sha256':core.digest(memories),
                  'source_corpus_sha256':core.digest(corpus),'source_fit_results_sha256':core.digest(dict(rows)),
                  'construction_used_benchmark_questions_or_answers':False}
        lock_path=out/'memory_selection_locked.json'
        if lock_path.exists():
            saved=read(lock_path)
            assert {k:v for k,v in saved.items() if k!='locked_at'}==locked
        else:
            core.save(lock_path,{'locked_at':time.time(),**locked})
        # First new-question QA access in this pipeline, after all memory locks.
        samples=read('data/locomo10.json')
        assert core.digest(samples)==audit['dataset_sha256']
        byid={str(s['sample_id']):s for s in samples if str(s['sample_id']) in cids}
        del samples
        results={}
        all_rows={}
        for name in NAMES:
            state('locked_new_question_audit',method=name)
            values=evaluate(rt,memories[name],audit['records'],byid,2048,name,out)
            results[name]=core.summarize(values)
            all_rows[name]=values
        core.save(out/'audit_results.json',{'summaries':results,'paired_delta_parent_vs_r40':core.paired_interval(all_rows[NAMES[1]],all_rows[NAMES[2]],20260908),
                  'frozen_method':NAMES[2],'no_audit_based_selection':True,'no_further_tuning_on_these_questions':True})
        state('audit_checkpoint_complete',n=100,frozen_method=NAMES[2],
              next='Report this frozen audit checkpoint without tuning from its outcomes; continue source/dev refinement separately. Goal remains active.')


if __name__=='__main__':
    main()
