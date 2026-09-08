import json
from pathlib import Path


def main():
    base=Path(__file__).parent
    queue=json.loads((base/'continuous_v2_queue.json').read_text(encoding='utf-8'))
    queue['recipes'] += [
        {'name':'r51_replay_reference','parent':'r12_filter_current_best','operations':[],'expected_f1':0.5711886887725268},
        {'name':'r52_replay_seed','parent':'seed','operations':[],'expected_f1':0.527942778998271},
        {'name':'r53_fact_index_source_payload','parent':'r12_filter_current_best','operations':[{'op':'indexed_evidence'}]},
        {'name':'r54_fact_index_single_source','parent':'r12_filter_current_best','operations':[{'op':'indexed_evidence','before':0}]},
        {'name':'r55_fact_index_surrounding','parent':'r12_filter_current_best','operations':[{'op':'indexed_evidence','before':1,'after':1}]},
        {'name':'r56_fact_index_prior_context','parent':'r12_filter_current_best','operations':[{'op':'indexed_evidence','before':2}]},
        {'name':'r57_indexed_summary_and_source','parent':'r12_filter_current_best','operations':[{'op':'indexed_evidence','include_summary':True}]},
        {'name':'r58_indexed_source_control','parent':'r12_filter_current_best','operations':[{'op':'indexed_evidence','index_source':True}]},
        {'name':'r59_indexed_without_residual','parent':'r12_filter_current_best','operations':[{'op':'indexed_evidence','residual':False}]},
        {'name':'r60_declarative_index','parent':'r16_declarative_facts','operations':[{'op':'indexed_evidence'}]},
        {'name':'r61_clean_card_index','parent':'r42_clean_source_cards','operations':[{'op':'indexed_evidence'}]},
        {'name':'r62_profile_index','parent':'r46_entity_profiles_only','operations':[{'op':'indexed_evidence','before':0}]},
        {'name':'r63_recursive_index_context','parent':'best','operations':[{'op':'indexed_evidence','before':1,'after':1}]},
        {'name':'r64_clean_index_best','parent':'best','operations':[{'op':'index_view'}]},
        {'name':'r65_clean_index_seed','parent':'seed','operations':[{'op':'index_view'}]},
        {'name':'r66_content_word_index','parent':'best','operations':[{'op':'index_view','remove_stopwords':True}]},
        {'name':'r67_split_best_fused','parent':'r40_fused_four_turn','operations':[{'op':'split_fused_index'}]},
        {'name':'r68_fused_key_both_payload','parent':'r40_fused_four_turn','operations':[{'op':'split_fused_index','read_summary':True}]},
        {'name':'r69_split_current_best','parent':'best','operations':[{'op':'split_fused_index'}]}
    ]
    (base/'continuous_v3_queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared',len(queue['recipes']),'recipes')


if __name__=='__main__':
    main()
