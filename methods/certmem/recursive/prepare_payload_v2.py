"""Create immutable continuation with bounded retries of incomplete JSON only."""
from pathlib import Path


def main():
    here=Path(__file__).parent
    source=(here/'provenance_payload.py').read_text(encoding='utf-8')
    extra='''

_initial_generate=generate


def generate(rt,windows):
    """Reuse complete 384-token responses; retry incomplete JSON at 768/1536."""
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    values=_initial_generate(rt,windows)
    valid=lambda v:isinstance(v['object'],dict) and set(v['object'])=={'facts'} and isinstance(v['object']['facts'],list) and len(v['object']['facts'])<=4
    attempts={i:[] for i in range(len(windows))}
    for i,w in enumerate(windows):
        if not w['oversized']:
            key=core.digest(['provenance_payload_v1',rt.model_meta,rt.args.seed,WRITER,w['user'],SCHEMA,384])
            attempts[i].append({'max_tokens':384,'cache_key':key,'complete_json':valid(values[i])})
    for cap in (768,1536):
        pending=[]
        for i,w in enumerate(windows):
            if w['oversized'] or valid(values[i]):continue
            key=core.digest(['provenance_payload_v1',rt.model_meta,rt.args.seed,WRITER,w['user'],SCHEMA,cap])
            path=rt.cache/'provenance_payload_generations'/f'{key}.json'
            if path.exists():
                values[i]=json.loads(path.read_text())
                attempts[i].append({'max_tokens':cap,'cache_key':key,'complete_json':valid(values[i])})
            else:
                prompt=rt.tok.apply_chat_template([{'role':'system','content':WRITER},{'role':'user','content':w['user']}],
                    tokenize=False,add_generation_prompt=True,enable_thinking=False)
                assert rt.ntok(prompt)+cap<=8192
                pending.append((i,key,path,prompt))
        for offset in range(0,len(pending),12):
            batch=pending[offset:offset+12]
            generated=rt.llm.generate([x[3] for x in batch],SamplingParams(temperature=0,max_tokens=cap,
                guided_decoding=GuidedDecodingParams(json=SCHEMA)),use_tqdm=False)
            for (i,key,path,_),value in zip(batch,generated):
                answer=value.outputs[0]
                try:obj=json.loads(answer.text)
                except json.JSONDecodeError:obj=None
                values[i]={'object':obj,'raw':answer.text,'finish_reason':answer.finish_reason,'max_tokens':cap}
                core.save(path,values[i])
                attempts[i].append({'max_tokens':cap,'cache_key':key,'complete_json':valid(values[i])})
            print(f'PAYLOAD_RETRY cap={cap} {offset+len(batch)}/{len(pending)}',flush=True)
    return [{**value,'attempts':attempts[i],'complete_json':valid(value)} for i,value in enumerate(values)]
'''
    module=here/'provenance_payload_v2.py'
    if module.exists():assert module.read_text(encoding='utf-8')==source+extra
    else:module.write_text(source+extra,encoding='utf-8',newline='\n')
    runner=(here/'run_payload_refinement.py').read_text(encoding='utf-8')
    runner=runner.replace('from provenance_payload import','from provenance_payload_v2 import').replace("default='runs/provenance_payload_v1'","default='runs/provenance_payload_v2'")
    runner=runner.replace("source_names=('run_payload_refinement.py','provenance_payload.py'","source_names=('run_payload_refinement_v2.py','provenance_payload_v2.py','provenance_payload.py'")
    runner=runner.replace("'controls':'raw original focus;", "'generation_retry':'Reuse complete 384-token initial output; retry only incomplete JSON with unchanged source/prompt/schema at 768 then 1536 tokens. Preserve every attempt and reject any remaining incomplete output.',\n              'controls':'raw original focus;")
    start=runner.index("    for path in (previous/'cache').rglob('*'):")
    end=runner.index("    status={'pid'",start)
    runner=runner[:start]+'''    superseded=Path('runs/provenance_payload_v1')
    assert read(superseded/'protocol_superseded.json')['reason']=='incomplete_structured_output_length'
    for cache_parent in (previous,superseded):
        for path in (cache_parent/'cache').rglob('*'):
            if path.is_file() and path.suffix in ('.json','.npy'):
                dest=out/'cache'/path.relative_to(cache_parent/'cache');dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():os.link(path,dest)
''' +runner[end:]
    target=here/'run_payload_refinement_v2.py'
    if target.exists():assert target.read_text(encoding='utf-8')==runner
    else:target.write_text(runner,encoding='utf-8',newline='\n')
    print('Created payload v2 module and runner; old sources unchanged.')


if __name__=='__main__':main()
