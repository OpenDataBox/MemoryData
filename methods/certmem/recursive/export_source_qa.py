"""Export cached source-only generation outputs for literal grounding recovery."""
import gzip
import json
from pathlib import Path

import refine as core


def main():
    root=Path('/workspace/locomo-refinement')
    run=root/'runs/continuous_v1'
    protocol=json.loads((run/'protocol.json').read_text())
    cids={r['conv_id'] for r in protocol['records']}
    recipe=json.loads((run/'plans/r24_source_qa_cards.json').read_text())['recipe']
    op=recipe['operations'][0]
    env=json.loads((root/'environment.json').read_text())
    model=env['models']['Qwen/Qwen3-8B']
    samples=json.loads((root/'data/locomo10.json').read_text())
    rows=[]
    for sample in samples:
        cid=str(sample['sample_id'])
        if cid not in cids:
            continue
        for session,turns in core.windows(core.session_data(sample),size=op['window'],overlap=op['overlap']):
            user=f"Recorded date: {session['date']}\nDialogue:\n"+'\n'.join(t['text'] for t in turns)+'\n\nMemory records:'
            key=core.digest([model,20260907,op['prompt'],user,op['max_tokens'],False])
            value=json.loads((run/'cache/generations'/f'{key}.json').read_text())
            rows.append({'conv_id':cid,'session':session['num'],'date':session['date'],
                         'turn_ids':[t['id'] for t in turns],'cache_key':key,**value})
    output=root/'source_qa_generations.json.gz'
    with gzip.open(output,'wt',encoding='utf-8') as stream:
        json.dump({'dataset_sha256':core.digest(samples),'model':model,'recipe':recipe,'records':rows},stream,ensure_ascii=False)
    print(json.dumps({'windows':len(rows),'archive_bytes':output.stat().st_size}))


if __name__=='__main__':
    main()
