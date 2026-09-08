"""Immutable, resumable GPU pilot for joint source evidence preservation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import refine as core
from evidence_utility import (POLICY,select_probes,subset_jobs,answer_token_request,
                             extract_answer_logprob,subset_analysis,generation_controls)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Scorer:
    def __init__(self, environment, out):
        from vllm import LLM
        self.model=environment['models']['Qwen/Qwen3-8B']
        self.cache=out/'cache'
        self.llm=LLM(model=self.model['path'],dtype='bfloat16',max_model_len=8192,
                     gpu_memory_utilization=0.78,max_num_seqs=8,max_num_batched_tokens=8192,
                     enable_prefix_caching=False,enforce_eager=True,seed=POLICY['seed'])
        self.tok=self.llm.get_tokenizer()
        self.ntok=lambda s:len(self.tok.encode(s,add_special_tokens=False))

    def score(self,jobs):
        from vllm import SamplingParams
        pending=[]
        for job in jobs:
            key=core.digest(['nll_token_concat_v1',self.model,POLICY['seed'],core.READER,job['user'],job['answer']])
            path=self.cache/'nll'/(key+'.json')
            if path.exists():
                job['score']=read(path)
            else:
                request,start,target=answer_token_request(self.tok,job['user'],job['answer'])
                if len(request['prompt_token_ids'])+1>8192:
                    raise ValueError('NLL prompt exceeds model length')
                pending.append((job,path,request,start,target))
        for offset in range(0,len(pending),8):
            batch=pending[offset:offset+8]
            outputs=self.llm.generate([x[2] for x in batch],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0),use_tqdm=False)
            if len(outputs)!=len(batch):
                raise ValueError('Scoring output count differs')
            for (job,path,request,start,target),output in zip(batch,outputs):
                job['score']=extract_answer_logprob(output,start,target)
                core.save(path,job['score'])

    def generate(self,controls):
        from vllm import SamplingParams
        result={}
        pending=[]
        for name,job in controls.items():
            key=core.digest(['utility_generation_v1',self.model,POLICY['seed'],core.READER,job['user'],POLICY['max_answer_tokens']])
            path=self.cache/'generation'/(key+'.json')
            if path.exists():
                result[name]=read(path)
            else:
                prompt=self.tok.apply_chat_template([{'role':'system','content':core.READER},
                     {'role':'user','content':job['user']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                pending.append((name,path,prompt))
        if pending:
            outputs=self.llm.generate([x[2] for x in pending],SamplingParams(temperature=0,max_tokens=POLICY['max_answer_tokens']),use_tqdm=False)
            for (name,path,prompt),output in zip(pending,outputs):
                value={'text':output.outputs[0].text.strip(),'finish_reason':output.outputs[0].finish_reason}
                core.save(path,value)
                result[name]=value
        for name,job in controls.items():
            result[name]={**result[name],'source_ids':job['source_ids'],'context_tokens':job['context_tokens'],
                          'source_answer_f1':core.generic_f1(result[name]['text'],job['answer'])}
        return result


def summarize(rows):
    result={}
    for split in ('probe_fit','probe_audit'):
        values=[r for r in rows if r['split']==split and 'analysis' in r]
        valid=[r for r in values if r['analysis']['source_dependent'] and
               r['generations']['full']['source_answer_f1']>=POLICY['minimum_full_answer_f1']]
        def avg(numbers):
            return sum(numbers)/len(numbers) if numbers else None
        result[split]={'scored':len(values),'source_dependent_and_full_consistent':len(valid),
             'full_answer_consistent':sum(r['generations']['full']['source_answer_f1']>=POLICY['minimum_full_answer_f1'] for r in values),
             'pair_needed_for_nll_preservation':sum(r['analysis']['has_preserving_pair_without_preserving_single'] for r in valid),
             'positive_pair_lift':sum(r['analysis']['best_pair_lift_over_best_single']>=POLICY['joint_lift_nats'] for r in valid),
             'valid_mean_full_tokens':avg([r['analysis']['full_tokens'] for r in valid]),
             'valid_mean_minimum_tokens':avg([r['analysis']['minimum_tokens'] for r in valid]),
             'valid_mean_single_policy_tokens':avg([r['analysis']['single_policy_tokens'] for r in valid]),
             'valid_control_f1':{name:avg([r['generations'][name]['source_answer_f1'] for r in valid])
                                for name in ('empty','full','answer_removed','best_single','best_pair','minimum_bundle','single_policy')}}
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sources',type=Path,required=True)
    ap.add_argument('--probes',type=Path,required=True)
    ap.add_argument('--environment',type=Path,default=Path('environment.json'))
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    corpus,pool,environment=read(args.sources),read(args.probes),read(args.environment)
    if set(corpus)!= {'dataset_sha256','sessions_by_id','excluded_holdout_conversations'}:
        raise ValueError('Source corpus schema must not contain benchmark QA')
    sessions=corpus['sessions_by_id']
    assert not set(sessions).intersection(corpus['excluded_holdout_conversations'])
    selected=select_probes(pool)
    assert {p['conv_id'] for p in selected}<=sessions.keys()
    source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ('run_evidence_utility.py','evidence_utility.py','refine.py')}
    protocol={'policy':POLICY,'reader':core.READER,'source_sha256':source_hashes,'source_corpus_sha256':core.digest(corpus),
              'probe_pool_sha256':core.digest(pool),'selected_ids':[p['id'] for p in selected],
              'environment':environment,'runtime':{'prefix_caching':False,'batch_size':8,'dtype':'bfloat16'},
              'purpose':'Source-only diagnostic; no benchmark F1 selection or generalization claim; session audit not used to tune policy'}
    if (args.out/'protocol.json').exists() and read(args.out/'protocol.json')!=protocol:
        raise ValueError('Frozen protocol changed; use a new output directory')
    core.save(args.out/'protocol.json',protocol)
    core.save(args.out/'selection.json',[{k:p[k] for k in ('id','conv_id','session','split')} for p in selected])
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
        turns={t['id']:t for s in sessions[probe['conv_id']] for t in s['turns']}
        jobs,skip=subset_jobs(probe,turns,scorer.ntok,POLICY['read_budget'])
        if skip:
            row={**probe,'skip':skip}
        else:
            scorer.score(jobs)
            analysis=subset_analysis(jobs)
            generations=scorer.generate(generation_controls(jobs,analysis))
            row={**probe,'analysis':analysis,'generations':generations,
                 'subsets':[{k:v for k,v in j.items() if k not in ('user','answer')} for j in jobs]}
        core.save(path,row)
        rows.append(row)
        core.save(args.out/'summary.json',summarize(rows))
        print('SOURCE_PROBE_RESULT '+json.dumps({'id':probe['id'],'completed':len(rows),'total':len(selected),
                    'subsets':len(jobs),'skip':skip,'analysis':{k:v for k,v in row.get('analysis',{}).items() if k!='pair_interactions'}}),flush=True)
    core.save(args.out/'summary.json',summarize(rows))
    core.save(args.out/'status.json',{'phase':'pilot_complete','pid':os.getpid(),'completed':len(rows),
                                    'total':len(selected),'heartbeat_at':time.time(),
                                    'next':'Inspect source-fit mechanism results and continue construction refinement; overall user goal remains active.'})


if __name__=='__main__':
    main()
