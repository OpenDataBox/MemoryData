"""Transfer the frozen parent-single policy to source-only conversations."""
from collections import defaultdict

import refine as core
from continuous_v2 import construct as apply_recipe
from parent_evidence import make_options,construct
from compile_research_data import cards_from_generations,split_probe_pool


def backend(rt,sessions,seed,plans):
    memory=seed
    for name in ('r06_calendar_month','r12_filter_current_best','r40_fused_four_turn'):
        memory=apply_recipe(rt,sessions,memory,plans[name]['recipe'])
    return memory


def augment(rt,sessions,parent,rows):
    if any(r['split']!='probe_fit' for r in rows):
        raise ValueError('Only source-fit rows enter the portable parent constructor')
    groups,mapped=make_options(rt,sessions,parent,rows)
    memory,details=construct(parent,groups,mapped,{}, {},rt.ntok,'parent_single',2000)
    return memory,details,{'groups':groups,'mapped':mapped}


def generate_source_probes(rt,sessions_by_id,qa_operation):
    records=[]
    for cid,sessions in sorted(sessions_by_id.items()):
        chunks=list(core.windows(sessions,size=qa_operation['window'],overlap=qa_operation['overlap']))
        users=[f"Recorded date: {s['date']}\nDialogue:\n"+'\n'.join(t['text'] for t in turns)+'\n\nMemory records:'
               for s,turns in chunks]
        outputs=rt.generate(qa_operation['prompt'],users,max_tokens=qa_operation['max_tokens'])
        for text,(session,turns) in zip(outputs,chunks):
            records.append({'conv_id':cid,'session':session['num'],'turn_ids':[t['id'] for t in turns],'text':text})
    dump={'records':records}
    pool=split_probe_pool(sessions_by_id,cards_from_generations(dump),seed=20260908,per_session=2)
    if not pool['records']:
        raise ValueError('No source-grounded probes; do not silently substitute benchmark questions')
    return dump,pool
