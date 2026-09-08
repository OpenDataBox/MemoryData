# 지속 refinement — 개발 점수 스냅샷

현재 최선: **r40_fused_four_turn**. 기존 최선 개발 F1 **52.79 → 58.62 (+5.83점)**.

기존 파일럿의 개발 70문항만 사용했다. 이전 검증 30문항은 이 반복에서 제외했으며 새 일반화 성능을 주장하지 않는다.

| 후보 | 출발 구성 | 개발 F1 | 저장 토큰/대화 | 읽기 토큰 | 채택 |
|---|---|---:|---:|---:|---|
| seed | r03_dialogue_residual | 52.79 | 69,080 | 2,037 | 기준 |
| r05_calendar_anchor | seed | 54.39 | 65,338 | 2,039 | 채택 |
| r06_calendar_month | seed | 54.86 | 64,874 | 2,038 | 채택 |
| r07_nonoverlap_pairs | seed | 47.95 | 40,709 | 2,038 | 기각 |
| r08_three_turn_blocks | seed | 54.04 | 50,298 | 2,037 | 기각 |
| r09_four_turn_blocks | seed | 54.44 | 59,214 | 2,038 | 기각 |
| r10_anchor_current_best | r06_calendar_month | 54.86 | 64,874 | 2,038 | 기각 |
| r11_social_filter | seed | 51.72 | 68,578 | 2,035 | 기각 |
| r12_filter_current_best | r06_calendar_month | 57.12 | 64,452 | 2,039 | 채택 |
| r13_extractive_cards | seed | 41.66 | 43,762 | 2,036 | 기각 |
| r14_anchored_cards | r13_extractive_cards | 42.32 | 37,712 | 2,040 | 기각 |
| r15_cards_with_pairs | r13_extractive_cards | 46.47 | 56,154 | 2,041 | 기각 |
| r16_declarative_facts | seed | 42.51 | 40,250 | 2,037 | 기각 |
| r17_anchored_declarative | r16_declarative_facts | 40.34 | 33,190 | 2,040 | 기각 |
| r18_declarative_with_pairs | r16_declarative_facts | 53.22 | 61,265 | 2,040 | 기각 |
| r19_anchored_four_turn | r09_four_turn_blocks | 53.93 | 56,694 | 2,038 | 기각 |
| r20_anchored_three_turn | r08_three_turn_blocks | 56.05 | 47,538 | 2,039 | 기각 |
| r21_anchored_nonoverlap | r07_nonoverlap_pairs | 52.02 | 37,764 | 2,040 | 기각 |
| r22_anchored_fact_raw_pair | seed | 55.06 | 65,450 | 2,038 | 기각 |
| r23_anchored_fact_raw_four | seed | 51.77 | 57,605 | 2,040 | 기각 |
| r24_source_qa_cards | seed | 42.26 | 29,484 | 2,033 | 기각 |
| r25_anchored_qa_cards | r24_source_qa_cards | 46.98 | 28,059 | 2,036 | 기각 |
| r26_qa_cards_with_context | r24_source_qa_cards | 49.98 | 64,884 | 2,033 | 기각 |
| r27_filtered_three_turn | r20_anchored_three_turn | 53.38 | 47,116 | 2,039 | 기각 |
| r28_filtered_fact_anchor | r22_anchored_fact_raw_pair | 54.86 | 65,028 | 2,039 | 기각 |
| r29_filtered_four_turn | r19_anchored_four_turn | 54.67 | 56,272 | 2,038 | 기각 |
| r30_filtered_partial_four | r23_anchored_fact_raw_four | 51.74 | 57,183 | 2,040 | 기각 |
| r31_filtered_declarative | r18_declarative_with_pairs | 51.95 | 59,901 | 2,040 | 기각 |
| r32_anchored_single_turn | seed | 54.86 | 41,710 | 2,039 | 기각 |
| r33_filtered_pair_blocks | seed | 54.83 | 62,465 | 2,040 | 기각 |
| r34_anchored_fact_only | seed | 52.15 | 9,329 | 2,039 | 기각 |
| r35_anchored_dialogue_only | seed | 50.68 | 53,136 | 2,028 | 기각 |
| r36_selective_anchor_filtered | seed | 56.74 | 69,341 | 2,037 | 기각 |
| r37_selective_anchor | seed | 53.97 | 69,847 | 2,037 | 기각 |
| r38_fused_pair | r12_filter_current_best | 55.95 | 63,980 | 2,027 | 기각 |
| r39_fused_three_turn | r12_filter_current_best | 53.24 | 48,125 | 2,028 | 기각 |
| r40_fused_four_turn | r12_filter_current_best | 58.62 | 57,177 | 2,017 | 현재 최선 |
| r41_fused_with_fact_index | r12_filter_current_best | 53.77 | 57,453 | 2,040 | 기각 |
| r42_clean_source_cards | r13_extractive_cards | 41.18 | 40,385 | 2,038 | 기각 |
| r43_clean_anchored_cards | r13_extractive_cards | 44.95 | 34,736 | 2,041 | 기각 |
| r44_clean_cards_with_context | r13_extractive_cards | 55.40 | 78,504 | 2,041 | 기각 |
| r45_entity_profiles | r40_fused_four_turn | 54.51 | 59,070 | 2,034 | 기각 |
| r46_entity_profiles_only | r40_fused_four_turn | 7.84 | 1,892 | 1,633 | 기각 |
| r47_profiles_with_context | r46_entity_profiles_only | 47.52 | 39,680 | 2,035 | 기각 |
| r48_salient_dialogue | r12_filter_current_best | 58.14 | 47,868 | 2,039 | 기각 |
| r49_salient_dialogue_only | r12_filter_current_best | 45.44 | 38,540 | 2,032 | 기각 |
| r50_salient_longer_context | r12_filter_current_best | 57.40 | 62,782 | 2,040 | 기각 |
| r50a_fused_five_turn | r12_filter_current_best | 56.32 | 48,693 | 2,027 | 기각 |
| r50b_fused_six_turn | r12_filter_current_best | 52.96 | 53,881 | 2,021 | 기각 |
| r50c_fused_stride_one | r12_filter_current_best | 54.31 | 98,811 | 2,016 | 기각 |
| r51_replay_reference | r12_filter_current_best | 57.12 | 64,452 | 2,039 | 기각 |
| r52_replay_seed | seed | 52.79 | 69,080 | 2,037 | 기각 |
| m01_fused_without_rules | seed | 46.92 | 60,137 | 2,023 | 기각 |
| m02_split_without_rules | m01_fused_without_rules | 46.36 | 59,214 | 2,017 | 기각 |
| m03_key_only_intervention | m01_fused_without_rules | 45.86 | 71,478 | 2,017 | 기각 |
| m04_index_without_rules | seed | 49.19 | 69,080 | 2,032 | 기각 |
| r67_split_best_fused | r40_fused_four_turn | 46.95 | 56,272 | 2,023 | 기각 |
| r68_fused_key_both_payload | r40_fused_four_turn | 48.20 | 66,501 | 2,019 | 기각 |
| r53_fact_index_source_payload | r12_filter_current_best | 48.13 | 64,452 | 2,029 | 기각 |
| r57_indexed_summary_and_source | r12_filter_current_best | 52.88 | 75,401 | 2,031 | 기각 |
| r58_indexed_source_control | r12_filter_current_best | 54.39 | 92,760 | 2,031 | 기각 |
| r59_indexed_without_residual | r12_filter_current_best | 46.31 | 37,637 | 2,033 | 기각 |

| 질문 유형 | 문항 수 | 기존 최선 | 현재 최선 | 차이 |
|---|---:|---:|---:|---:|
| 다단계 | 12 | 55.13 | 50.53 | -4.61 |
| 시간 | 14 | 44.06 | 66.94 | +22.87 |
| 개방형 지식 | 5 | 2.61 | 0.00 | -2.61 |
| 단일 사실 | 39 | 61.64 | 65.64 | +4.00 |

예측 4340개를 로컬에서 재채점했다. 문항 ID·개발 대화 분리·2,048토큰 상한·전체 요약 점수·최선 선택을 검증했다.
메모리 생성기는 대화만 받는다. reader·채점·읽기 예산과 BM25+dense RRF 검색 알고리즘은 고정했다. v3에서는 구성기가 만든 검색용 표현과 읽기용 증거의 분리도 비교했으며, 서로 다르면 둘 다 저장 비용에 포함한다.
저장 예산을 일치시킨 비교는 아니며 개발 70문항에 대한 반복 선택 편향이 있다. 최고 점수는 새 일반화 성능이 아니다.
r24_source_qa_cards와 그 파생 후보는 생성 출력 형식 실패로 원문 fallback이 대부분이었다. 따라서 올바르게 구성한 Q&A 메모리 방식의 성능으로 해석하지 않는다.
후보 개수를 완료 조건으로 삼지 않는다. 실행 중인 worker에 새 후보를 추가하고, 결과를 분석해 다음 구성을 정한다.
결과와 문항별 근거: `D:\MemoryData\results\recursive_locomo_20260907\continuous_v3\run`.
