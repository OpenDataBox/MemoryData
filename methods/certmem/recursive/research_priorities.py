"""Prioritize architectural controls after the user's methodological research steer."""
import json
from pathlib import Path

from refine import save


def main():
    root=Path('/workspace/locomo-refinement')
    if not root.exists():
        root=Path(__file__).parent
    v2=json.loads((root/'continuous_v2_queue.json').read_text(encoding='utf-8'))
    v3=json.loads((root/'continuous_v3_queue.json').read_text(encoding='utf-8'))
    history_path=root/'runs/continuous_v2/history.json'
    done={r['name'] for r in json.loads(history_path.read_text())['rounds']} if history_path.exists() else set()
    deferred={r['name'] for r in v2['recipes'] if r['name'].startswith(('r50a_','r50b_','r50c_')) and r['name'] not in done}
    old=v3['recipes']
    save(root/'pre_research_priority_queue.json',v3)
    v2['recipes']=[r for r in v2['recipes'] if r['name'] not in deferred]
    chosen={r['name']:r for r in old}
    new=[
        {'name':'m01_fused_without_rules','parent':'seed','operations':[{'op':'fuse_evidence','size':4,'overlap':2,'anchor':False}]},
        {'name':'m02_split_without_rules','parent':'m01_fused_without_rules','operations':[{'op':'split_fused_index'}]},
        {'name':'m03_key_only_intervention','parent':'m01_fused_without_rules','operations':[{'op':'split_fused_index','read_summary':True}]},
        {'name':'m04_index_without_rules','parent':'seed','operations':[{'op':'indexed_evidence','anchor':False}]}
    ]
    architecture=[chosen[n] for n in ['r51_replay_reference','r52_replay_seed']]+new
    architecture += [chosen[n] for n in ['r67_split_best_fused','r68_fused_key_both_payload','r53_fact_index_source_payload',
                                       'r57_indexed_summary_and_source','r58_indexed_source_control','r59_indexed_without_residual']]
    v3['recipes']=v2['recipes']+architecture
    v3['research_priority']='Architectural controls and no-rule variants; old dev F1 is exploratory. No novelty claim from key/evidence split alone.'
    selected={r['name'] for r in v3['recipes']}
    save(root/'deferred_parameter_recipes.json',[r for r in old if r['name'] not in selected])
    save(root/'continuous_v2_queue.json',v2)
    save(root/'continuous_v3_queue.json',v3)
    print(json.dumps({'v2_recipes':len(v2['recipes']),'v3_recipes':len(v3['recipes']),'deferred':sorted(deferred)}))


if __name__=='__main__':
    main()
