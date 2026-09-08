# Agent memory related work: 조사 메모

검토일: **2026-09-03**. 앞선 대화와 MemoryData 원고에 맞춰 **LLM agent의 외부·장기 메모리**를 중심으로 선정했다. 원고는 데이터 관리 관점의 시스템 비교 연구이므로, 일반적인 attention/KV-cache 최적화와 고전 신경망 메모리 전체를 포괄하는 survey로 확장하지 않았다. 수정본은 약 **300단어, 4개 문단, 인용 논문 38편**이다. 첫 초안의 약 730단어·24편에서 분량을 줄이고 문헌 14편을 추가했다. 최근 선정 논문은 2026년 8월까지 포함한다. 전체 문헌을 빠짐없이 수집한 systematic review는 아니다.

## 산출물

- `related_work_20260903.tex`: 논문용 영문 LaTeX 초안.
- `related_work_20260903.bib`: 본문에서 실제 인용한 38편의 BibTeX.
- `related_work_20260903.md`: 출처 링크를 붙인 읽기용 영문 초안.
- `../output/memory_related_work_20260903/`: 원문 메타데이터, 주요 최신 논문의 HTML 본문, 출판사 BibTeX, 조회 기록 및 검증 결과.

## 기반 연구를 선정한 이유

피인용 수만으로 순위를 매기지 않고, 후속 연구에서 반복되는 설계의 출발점인지 함께 보았다. RAG는 외부 지식 접근의 기반, Generative Agents는 experience stream과 reflection, Reflexion은 실패 경험의 언어적 재사용, MemGPT는 명시적 계층 관리, MemoryBank는 지속 대화 메모리와 forgetting의 기반으로 선정했다. HippoRAG는 구조화 검색으로 이어지는 연결 고리다.

| 논문 | 발표 형태 | 이 원고에서 맡는 역할 | 1차 출처 |
|---|---|---|---|
| RAG | NeurIPS 2020 | parametric knowledge와 external corpus의 결합 | [출판본](https://proceedings.neurips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) |
| Generative Agents | UIST 2023 | 경험 스트림, 중요도·최근성·관련성 기반 검색, reflection | [출판본 DOI](https://doi.org/10.1145/3586183.3606763) |
| Reflexion | NeurIPS 2023 | verbal feedback를 episodic memory로 활용 | [출판본](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) |
| MemGPT | arXiv, 2023부터 | in-context와 external memory 사이의 명시적 이동 | [현재 arXiv 레코드](https://arxiv.org/abs/2310.08560) |
| MemoryBank | AAAI 2024 | 장기 대화 저장·검색·요약·망각 | [출판본](https://ojs.aaai.org/index.php/AAAI/article/view/29946) |
| HippoRAG | NeurIPS 2024 | graph와 personalized PageRank를 통한 associative retrieval | [출판본](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html) |

피인용 수 확인을 위해 OpenAlex를 조회했다. 2026-09-03 조회에서 개별 출판본 레코드는 Generative Agents 1,774회, Reflexion 444회, MemoryBank 178회를 반환했다. 이는 **OpenAlex의 특정 레코드에 연결된 인용 수**이며 Google Scholar의 전체 인용 수가 아니다. preprint와 출판본이 분리되어 있어 합산하지 않았으며, 이 값을 분야 전체 순위나 논문 품질 척도로 쓰지 않았다. RAG와 MemGPT를 포함한 원시 검색 결과도 그대로 보존했다.

출처: [Generative Agents 레코드](https://openalex.org/W4387835442), [Reflexion 레코드](https://openalex.org/W7133222210), [MemoryBank 레코드](https://openalex.org/W4393147158). 원시 조회: `../output/memory_related_work_20260903/citation_snapshot.json`.

## 최근 방법론: 무엇이 달라지는가

| 논문 / cite key | 발표 형태 | 본문에 반영한 핵심 | 1차 출처 |
|---|---|---|---|
| A-MEM / `mrw_amem` | NeurIPS 2025 | 설명·키워드·태그가 있는 노트와 동적 연결·갱신 | [출판본](https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html) |
| Mem0 / `mrw_mem0` | arXiv 2025 | 사실 추출, 기존 기억과의 조정, graph 확장 | [논문](https://arxiv.org/abs/2504.19413) |
| Zep / `mrw_zep` | arXiv 2025 | 시간에 따라 변하는 지식의 graph 표현 | [논문](https://arxiv.org/abs/2501.13956) |
| MemoryOS / `mrw_memoryos` | EMNLP 2025 | 계층별 저장·업데이트·검색·생성 관리 | [출판본](https://aclanthology.org/2025.emnlp-main.1318/) |
| Hindsight / `mrw_hindsight` | arXiv 2025부터 | 사실·경험·entity summary·belief 구분과 retain/recall/reflect | [논문](https://arxiv.org/abs/2512.12818) |
| LightMem / `mrw_lightmem` | ICLR 2026 | 필터링, topic-aware consolidation, offline update | [출판본](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a05b72653ec5b473732129829ae04195-Abstract-Conference.html) |
| SimpleMem / `mrw_simplemem` | ICML 2026 공식 목록 확인 | compact structured memory, semantic synthesis, intent-aware retrieval | [논문](https://arxiv.org/abs/2601.02553), [학회 목록](https://icml.cc/Downloads/2026) |
| MEM1 / `mrw_mem1` | arXiv 2025부터 | RL로 compact recurrent memory와 reasoning을 공동 학습 | [논문](https://arxiv.org/abs/2506.15841) |
| AdaMEM / `mrw_adamem` | ICML 2026 공식 목록 확인 | 실행 중 trajectory retrieval과 strategy synthesis; 별도 STEP-MFT | [논문](https://arxiv.org/abs/2606.05684), [저자 페이지](https://yunx-z.github.io/AdaMEM/), [학회 목록](https://icml.cc/Downloads/2026) |
| MAGE / `mrw_execution_state` | arXiv 2026-06 | 실행 상태 tree의 grow/compress/maintain/revise | [논문](https://arxiv.org/abs/2606.06090) |
| LeanMem / `mrw_leanmem` | arXiv 2026-08 | profile/event/source-grounded record의 선택적 저장·유지·검색 | [논문](https://arxiv.org/abs/2608.03463) |
| MESA / `mrw_mesa` | arXiv 2026-08 | 질문별로 여러 memory structure의 일부를 선택·결합 | [논문](https://arxiv.org/abs/2608.10108) |

최근 preprint는 최신 설계 방향을 소개하기 위해 포함했다. 피인용이 많이 누적된 연구라고 표현하거나 peer review를 통과했다고 일괄 취급하지 않았다. `arXiv` 표기는 검토에 사용한 서지 형태를 뜻하며 다른 발표 경로가 전혀 없다는 단정은 아니다.

## 평가·분석 연구

| 논문 / cite key | 발표 형태 | 관련성 | 1차 출처 |
|---|---|---|---|
| LoCoMo / `mrw_locomo` | ACL 2024 | 긴 다중 세션 대화 이해 | [출판본](https://aclanthology.org/2024.acl-long.747/) |
| LongMemEval / `mrw_longmemeval` | ICLR 2025 | 추출·다중 세션 추론·갱신·시간·abstention | [논문](https://arxiv.org/abs/2410.10813), [저자 저장소](https://github.com/xiaowu0162/LongMemEval) |
| MemoryAgentBench / `mrw_memoryagentbench` | ICLR 2026 | incremental interaction에서 서로 다른 memory capability 평가 | [논문](https://arxiv.org/abs/2507.05257), [저자 저장소](https://github.com/HUST-AI-HYZ/MemoryAgentBench) |
| LoCoMo-Plus / `mrw_locomo_plus` | arXiv 2026-02 | 나중에 명시적으로 질문하지 않는 latent constraint의 적용 | [논문](https://arxiv.org/abs/2602.10715) |
| Memory in the Age of AI Agents / `mrw_survey` | arXiv 2025부터 | forms/functions/dynamics의 종합 taxonomy | [논문](https://arxiv.org/abs/2512.13564) |
| Memory in the LLM Era / `mrw_memoryeab` | arXiv 2026-04부터 | 모듈별 통합 framework와 시스템 비교; 가장 가까운 연구 중 하나 | [논문](https://arxiv.org/abs/2604.01707) |

## 우리 원고의 포지셔닝에 중요한 점

**통합 평가, 4개 모듈 분해, accuracy–cost 비교 자체를 최초 기여로 쓰기는 어렵다.** Wu et al.의 현재 공개 본문은 extraction/management/storage/retrieval 분해와 LoCoMo, LongMemEval, MemoryArena 비교를 이미 포함한다. 기존 로컬 초안의 '두 벤치마크' 설명만 그대로 이어받으면 현재 버전을 충분히 반영하지 못한다. 이번 초안은 이 연구를 명시적으로 인용하고, MemoryData 원고가 주장하는 evidence retention, retrieval fidelity, update behavior, operational cost의 연결을 분석 초점으로 서술했다.

이 마지막 문단은 로컬 원고의 연구 범위를 바탕으로 작성했다. 해당 분석이 다른 연구에 전혀 없다는 주장이나 실험 결과·시스템 수·데이터셋 수를 새로 만들지 않았다. 실제 제출 전에는 이 차별점에 대응하는 실험과 분석이 본문에 존재하는지 확인해야 한다.

논문 간 점수·배수 개선을 직접 비교하지 않았다. 서로 다른 backbone, embedding, retrieval budget, 데이터 subset, judge와 offline 비용 포함 여부가 섞이기 때문이다. 또한 연구 논문의 방법과 현재 제품 SDK의 구현을 구분했다. 예를 들어 Mem0 문단은 인용한 2025년 논문을 설명하며, 최신 SDK의 모든 세부 동작을 대표한다는 뜻이 아니다.

## 이름·버전·서지 정보 확인

- 여기서 LightMem은 Fang et al.의 **2510.18866**이다. 다른 저자들이 발표한 **2604.07798**도 LightMem이라는 이름을 사용하므로 혼동하지 않았다.
- A-MEM(**2502.12110**)과 AdaMEM(**2606.05684**)은 별도 연구다. AdaMEM은 test-time trajectory/strategy adaptation 연구를 인용했다.
- MemoryOS(**2506.06326**)와 MemOS(**2507.03724**)는 별도 연구다. 수정본에는 둘 다 포함하고, 각각 계층 관리와 heterogeneous memory 관리의 근거로 인용했다.
- LangMem은 이 범위에서 사용할 단일 공식 연구 논문을 확인하지 못했으므로 논문 목록에 끼워 넣지 않았다. 구현 설명이 필요하면 [공식 문서](https://langchain-ai.github.io/langmem/)를 software/web reference로 별도 인용하면 된다.
- SimpleMem의 현재 공개 설명은 online semantic synthesis와 intent-aware retrieval planning을 사용한다. 과거 버전의 명칭을 그대로 인용하지 않았다.
- 저자 정보는 출판사 BibTeX를 우선했다. 예를 들어 Reflexion의 출판본 저자 목록과 최신 arXiv 저자 목록은 다르므로 혼합하지 않았다. 출판본을 사용하지 않은 MemGPT와 Zep은 현재 arXiv 메타데이터를 사용했다.
- SimpleMem과 AdaMEM은 ICML 공식 목록에서 제목을 확인했다. 개별 poster 페이지는 접근하지 못했으므로 미확인 PMLR volume/pages를 만들지 않았다.

## BibTeX 사용 및 검증

이 초안의 인용 키는 모두 `mrw_`로 시작한다. 원고에서 다음과 같이 사용할 수 있다.

```latex
\input{related_work_20260903}
% For a document using only this draft's references:
\bibliography{related_work_20260903}
```

기존 `sample.bib`도 함께 쓰는 전체 원고에 통합할 때는 같은 논문이 두 citation key로 중복 출력되지 않도록 기존 키와 합쳐야 한다. 이번 파일은 독립적으로 검토할 수 있는 본문과 참고문헌 세트이며, 기존 원고의 인용 키를 일괄 변경하지 않았다.

검증 결과: BibTeX parser로 38개 entry를 읽었고 중복 키, 본문에서 참조한 미정의 키, 사용하지 않는 entry가 없었다. 각 entry의 title/author/year/URL 존재도 확인했다. TeX compiler가 확인되지 않아 PDF 컴파일은 수행하지 않았다. 원시 검증 결과는 `../output/memory_related_work_20260903/validation.json`에 있다.

## 축약본에서 추가한 14편

논문별 설명을 길게 늘리는 대신, 각 문장이 설명하는 설계 계열에 맞춰 인용을 묶었다. 기존 24편은 모두 유지했다. 아래 논문의 저자·제목은 원문 메타데이터와 대조했으며, 출판사 레코드가 확보된 항목은 출판본 BibTeX를 우선했다.

| 추가 논문 / cite key | 서지 형태 | 해당 문장을 뒷받침하는 내용 | 출처 |
|---|---|---|---|
| Voyager / `mrw_voyager` | TMLR 2024 | executable skill library를 통한 경험 재사용 | [저자 연구실](https://rpl.cs.utexas.edu/publications/2024/03/14/wang-tmlr24-voyager/), [논문](https://arxiv.org/abs/2305.16291) |
| ReadAgent / `mrw_readagent` | ICML 2024 | gist compression 후 필요한 원문을 선택적으로 재조회 | [출판본](https://proceedings.mlr.press/v235/lee24c.html) |
| MemTree / `mrw_memtree` | ICLR 2025 | 추상화 수준이 다른 기억을 동적 tree로 조직 | [논문](https://arxiv.org/abs/2410.14052), [저자 페이지](https://www.yujia.io/) |
| HippoRAG 2 / `mrw_hipporag2` | ICML 2025 | graph·passage 통합과 associative retrieval | [출판본](https://proceedings.mlr.press/v267/gutierrez25a.html) |
| MemOS / `mrw_memos` | arXiv 2025부터 | plaintext·activation·parameter memory의 통합 관리 | [논문](https://arxiv.org/abs/2507.03724) |
| Memory-R1 / `mrw_memory_r1` | arXiv 2025부터 | RL 기반 memory operation 및 활용 policy | [논문](https://arxiv.org/abs/2508.19828) |
| MemRL / `mrw_memrl` | arXiv 2026 | 가중치 변경 없이 episodic memory의 utility를 학습 | [논문](https://arxiv.org/abs/2601.03192) |
| MemGen / `mrw_memgen` | arXiv 2025부터 | reasoning state에 따라 latent memory 생성 | [논문](https://arxiv.org/abs/2509.24704) |
| MAGMA / `mrw_magma` | ACL 2026 | semantic·temporal·causal·entity의 다중 graph 표현 | [출판본](https://aclanthology.org/2026.acl-long.1709/) |
| MemAgent / `mrw_memagent` | arXiv 2025부터 | 긴 입력을 순차 처리하며 memory overwrite를 RL로 최적화 | [논문](https://arxiv.org/abs/2507.02259) |
| MemBench / `mrw_membench` | Findings of ACL 2025 | factual/reflective memory와 participation/observation 평가 | [출판본](https://aclanthology.org/2025.findings-acl.989/) |
| MemoryArena / `mrw_memoryarena` | arXiv 2026 | 이전 세션의 경험이 이후 행동에 필요한 agentic task 평가 | [논문](https://arxiv.org/abs/2602.16313) |
| AMA-Bench / `mrw_amabench` | ICML 2026 저자 저장소 확인 | agent-environment trajectory 기반 long-horizon QA 평가 | [논문](https://arxiv.org/abs/2602.22769), [저자 저장소](https://github.com/AMA-Bench/AMA-Bench/) |
| A Survey on the Memory Mechanism of Large Language Model-based Agents / `mrw_survey2024` | ACM TOIS 2025; preprint 2024 | memory design·evaluation·application의 체계적 분류 | [출판본 DOI](https://doi.org/10.1145/3748302) |

MemGen은 외부 텍스트 저장소와 대비되는 latent-memory 설계로 포함했다. MemRL과 Memory-R1은 모두 RL을 사용하지만, 모델 가중치를 학습하는지 memory utility를 학습하는지가 달라 별도 문장으로 구분했다. 또한 MemoryArena의 task execution과 AMA-Bench의 trajectory QA를 같은 평가로 표현하지 않았다.
