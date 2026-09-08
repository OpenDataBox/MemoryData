import json
from pathlib import Path

import refine as core


def main():
    root=Path('/workspace/locomo-refinement')
    history=json.loads((root/'runs/continuous_v1/history.json').read_text())
    recipe=json.loads((root/'runs/continuous_v1/plans/r24_source_qa_cards.json').read_text())['recipe']
    op=recipe['operations'][0]
    env=json.loads((root/'environment.json').read_text())
    model=env['models']['Qwen/Qwen3-8B']
    sample=next(s for s in json.loads((root/'data/locomo10.json').read_text()) if s['sample_id']=='conv-26')
    session,turns=next(core.windows(core.session_data(sample),size=op['window'],overlap=op['overlap']))
    user=f"Recorded date: {session['date']}\nDialogue:\n"+'\n'.join(t['text'] for t in turns)+'\n\nMemory records:'
    key=core.digest([model,20260907,op['prompt'],user,op['max_tokens'],False])
    value=json.loads((root/'runs/continuous_v1/cache/generations'/f'{key}.json').read_text())
    print(json.dumps({k:v for k,v in value.items() if k!='text'}))
    print(value['text'][:5000])


if __name__=='__main__':
    main()
