"""Append the next construction generation without editing prior recipes."""
import json
from pathlib import Path


def main():
    base=Path(__file__).parent
    queue=json.loads((base/'continuous_queue.json').read_text(encoding='utf-8'))
    queue['recipes'] += [
        {'name':'r36_selective_anchor_filtered','parent':'seed','operations':[{'op':'selective_anchor'},{'op':'filter_social_facts'}]},
        {'name':'r37_selective_anchor','parent':'seed','operations':[{'op':'selective_anchor'}]},
        {'name':'r38_fused_pair','parent':'best','operations':[{'op':'fuse_evidence','size':2,'overlap':1,'anchor':True}]},
        {'name':'r39_fused_three_turn','parent':'r12_filter_current_best','operations':[{'op':'fuse_evidence','size':3,'overlap':1,'anchor':True}]},
        {'name':'r40_fused_four_turn','parent':'r12_filter_current_best','operations':[{'op':'fuse_evidence','size':4,'overlap':2,'anchor':True}]},
        {'name':'r41_fused_with_fact_index','parent':'r12_filter_current_best','operations':[{'op':'fuse_evidence','size':3,'overlap':1,'anchor':True,'keep_fact_index':True}]},
        {'name':'r42_clean_source_cards','parent':'r13_extractive_cards','operations':[{'op':'clean_cards'}]},
        {'name':'r43_clean_anchored_cards','parent':'r13_extractive_cards','operations':[{'op':'clean_cards'},{'op':'anchor_time','style':'month'}]},
        {'name':'r44_clean_cards_with_context','parent':'r13_extractive_cards','operations':[{'op':'clean_cards'},{'op':'dialogue_blocks','size':2,'overlap':1},{'op':'anchor_time','style':'month'}]},
        {'name':'r45_entity_profiles','parent':'best','operations':[{'op':'entity_profiles','append':True}]},
        {'name':'r46_entity_profiles_only','parent':'best','operations':[{'op':'entity_profiles','append':False}]},
        {'name':'r47_profiles_with_context','parent':'r46_entity_profiles_only','operations':[{'op':'dialogue_blocks','size':3,'overlap':1},{'op':'anchor_time','style':'month'}]},
        {'name':'r48_salient_dialogue','parent':'r12_filter_current_best','operations':[{'op':'salient_dialogue'}]},
        {'name':'r49_salient_dialogue_only','parent':'r12_filter_current_best','operations':[{'op':'salient_dialogue','include_facts':False}]},
        {'name':'r50_salient_longer_context','parent':'r12_filter_current_best','operations':[{'op':'salient_dialogue','previous_turns':2}]},
        {'name':'r50a_fused_five_turn','parent':'r12_filter_current_best','operations':[{'op':'fuse_evidence','size':5,'overlap':2,'anchor':True}]},
        {'name':'r50b_fused_six_turn','parent':'r12_filter_current_best','operations':[{'op':'fuse_evidence','size':6,'overlap':3,'anchor':True}]},
        {'name':'r50c_fused_stride_one','parent':'r12_filter_current_best','operations':[{'op':'fuse_evidence','size':4,'overlap':3,'anchor':True}]}
    ]
    (base/'continuous_v2_queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared',len(queue['recipes']),'recipes')


if __name__=='__main__':
    main()
