<div align="center">
  <h1><img src="./Memory.png" alt="MemoryData logo" width="96" align="middle" />&nbsp;MemoryData</h1>
  <p><b>A Unified Memory Benchmark Suite for Memory-Augmented Agents</b></p>
  <p><i>"One pipeline. Four benchmark families. Twenty-two method presets. One consistent execution interface."</i></p>
</div>

<div align="center">

  [![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![Methods](https://img.shields.io/badge/Methods-22%20Presets-7C3AED?style=flat-square)]()
  [![Benchmarks](https://img.shields.io/badge/Benchmarks-4%20Families-F97316?style=flat-square)]()
  [![Platform](https://img.shields.io/badge/Platform-Linux·macOS·Windows-00D4FF?style=flat-square)]()
  [![Taxonomy](https://img.shields.io/badge/Taxonomy-Reference·Sequential·Topological·Hybrid-16A34A?style=flat-square)]()

</div>

<div align="center">
  <a href="#-introduction">Introduction</a> &nbsp;•&nbsp;
  <a href="#-features">Features</a> &nbsp;•&nbsp;
  <a href="#-quick-start">Quick Start</a> &nbsp;•&nbsp;
  <a href="#-repository-layout">Layout</a> &nbsp;•&nbsp;
  <a href="#-method-overview">Methods</a> &nbsp;•&nbsp;
  <a href="#-benchmark-overview">Benchmarks</a> &nbsp;•&nbsp;
  <a href="#-configuration-conventions">Config</a> &nbsp;•&nbsp;
  <a href="#-output-artifacts">Artifacts</a> &nbsp;•&nbsp;
  <a href="#-faq">FAQ</a> &nbsp;•&nbsp;
  <a href="#-citation">Citation</a>
</div>

<br/>

<div align="center">
  <img src="./MemoryData_overview.png" alt="MemoryData main results" width="920" />
  <br/>
  <sub>Main results from the accompanying paper: memory-augmented agent methods compared across the <b>LongMemEval</b>, <b>LoCoMo</b>, and <b>DB-Bench</b> benchmarks under exact-match, ROUGE-L, and LLM-judge metric families. Bars are grouped by paradigm — Reference Baselines, Sequential Context, Structural Topological, and Multi-Paradigm Hybrid.</sub>
</div>


## ✨ Introduction

Memory-augmented agents, structured memory architectures, and retrieval-based baselines are usually evaluated in isolation — each paper ships its own loader, its own runtime adapter, and its own metric harness. Results are hard to compare, and reproducing a single number across two methods often means re-implementing both.

**MemoryData closes that gap.** It is a research-oriented benchmark suite that unifies four benchmark families (MemoryAgentBench, LoCoMo, LongBench, MemBench), twenty-two method presets, and a shared runtime under a single `main.py` launcher, so that heterogeneous memory formulations can be compared under one consistent execution interface and one stable artifact layout.


## 📚 Features

<table>
<tr>

<td width="50%">

**🚀 Unified Launcher**
`main.py` is the single entry point for benchmark execution, artifact writing, and optional post-run evaluation hooks. Select any method and any benchmark by pointing at two YAML files.

</td>
<td width="50%">

**🧩 22 Method Presets**
Flattened YAML presets span reference baselines, sequential context, structural topological, and multi-paradigm hybrid architectures — each wired to its vendored runtime.

</td>
</tr>
<tr>
<td>

**📊 4 Benchmark Families**
MemoryAgentBench, LoCoMo, LongBench, and MemBench, each with full and category-specific or slice-specific configs ready to run.

</td>
<td>

**🗂 Consistent Taxonomy**
Methods are grouped following the paper's RQ1 effectiveness taxonomy, so presets are discoverable by paradigm instead of by filename.

</td>
</tr>
<tr>
<td>

**📦 Structured Artifacts**
Every run emits a result JSON, persisted agent state, and optional logs under a stable, override-able `results/` root for reproducible post-processing.

</td>
<td>

**🖥 Cross-Platform**
Separate dependency manifests for Linux/macOS and Windows, with BM25 and long-context reference paths retained under `utils/`.

</td>
</tr>
</table>


## 🕹 Quick Start

**Prerequisites:** Python 3.11, an OpenAI-compatible model endpoint, and the benchmark datasets placed under `datasets/`.

### Step 1: Create the environment

```bash
conda create -n memory-bench python=3.11
conda activate memory-bench
```

| Platform | Command |
| --- | --- |
| Linux / macOS | `pip install -r requirements.txt` |
| Windows | `pip install -r requirements-windows.txt` |

### Step 2: Configure model endpoints and keys

Most presets assume OpenAI-compatible serving endpoints. Update the YAML files in `config/` so that `model`, `base_url`, `embedding_base_url`, and related provider fields match the model servers available in your environment.

| Variable | Used by | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | Most presets | Default key variable for chat and embedding calls |
| `OPENAI_API_BASE` | MemOS example environment | Refer to `methods/MemOS/config/.env.example` when using MemOS-specific setup |

### Step 3: Prepare datasets

Datasets are **not** bundled with this repository. Place them under `datasets/` according to the loader expectations.

| Benchmark | Default path | Format | Notes |
| --- | --- | --- | --- |
| MemoryAgentBench | `datasets/MemoryAgentBench/eval_dataset_collection/` | HuggingFace `save_to_disk` directory | Falls back to `ai-hyz/MemoryAgentBench` if the local copy is absent |
| LoCoMo | `datasets/LoCoMo/rq1_4cat_600_dist/locomo_4cat_600_dist.json` | JSON file | Used by the full and category-specific LoCoMo presets |
| LongBench | `datasets/longBench_rep150_proportional/datasets` | HuggingFace `save_to_disk` directory | Targets the proportional subset |
| MemBench | `datasets/MemBench/MemData/FirstAgent/*.json` | JSON files | `simple`, `noisy`, `knowledge_update`, `highlevel`, `RecMultiSession` |

Reference layout:

```text
datasets/
├── MemoryAgentBench/
│   └── eval_dataset_collection/          # HuggingFace save_to_disk directory
├── LoCoMo/
│   └── rq1_4cat_600_dist/
│       └── locomo_4cat_600_dist.json
├── longBench_rep150_proportional/
│   └── datasets/                         # HuggingFace save_to_disk directory
└── MemBench/
    └── MemData/FirstAgent/               # simple / noisy / knowledge_update / highlevel / RecMultiSession
```

### Step 4: Run experiments

Command template:

```bash
python main.py --agent_config <agent_yaml> --dataset_config <dataset_yaml>
```

Representative runs:

| Scenario | Agent config | Dataset config | Extra flags |
| --- | --- | --- | --- |
| Default MemoryAgentBench run | `config/reference_long_context_agent.yaml` | `benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml` | - |
| Small smoke run | `config/reference_long_context_agent.yaml` | `benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml` | `--max_test_queries_ablation 1` |
| LoCoMo evaluation | `config/hybrid_simplemem.yaml` | `benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml` | - |
| LongBench evaluation | `config/reference_embedding_rag.yaml` | `benchmark/longbench/config/LongBench_rep150_proportional.yaml` | - |
| MemBench evaluation | `config/sequential_mem0.yaml` | `benchmark/membench/config/MemBench_simple.yaml` | - |

Example:

```bash
python main.py \
  --agent_config config/reference_long_context_agent.yaml \
  --dataset_config benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml
```


## 🗂 Repository Layout

```text
project-root/
├── main.py                        # unified experiment entry point
├── config/                        # flattened presets: reference, sequential, topological, hybrid
├── benchmark/
│   ├── memoryagentbench/          # MemoryAgentBench loaders and benchmark configs
│   ├── locomo/                    # LoCoMo configs and JSON loader
│   ├── longbench/                 # LongBench proportional-subset support
│   └── membench/                  # MemBench slice configs and loader
├── evaluation/
│   └── longmemeval/               # retained LongMemEval sidecar evaluation helpers
├── methods/                        # method runtimes grouped by the paper taxonomy
│   ├── embedding_rag/              # reference dense-retrieval baseline
│   ├── memagent/  mem0/  memochat/ # sequential context architectures
│   ├── cognee/  graph_rag/  hipporag/  memtree/  raptor/  zep/  zep_local/ # structural topological architectures
│   └── a_mem/  everos/  letta/  lightmem/  memorag/  memoryos/  self_rag/  simplemem/  MemOS/ # multi-paradigm hybrid architectures
├── utils/                          # shared runtime utilities, including long-context and BM25 reference paths
├── requirements.txt               # dependency manifest for Linux/macOS
└── requirements-windows.txt       # dependency manifest for Windows
```


## 🧠 Method Overview

The taxonomy below follows the grouping used in the main RQ1 effectiveness table of the accompanying paper. Methods retained in the released codebase but not displayed in that specific summary table are assigned to the corresponding taxonomy group for completeness.

| Group | Method | Representative preset | Runtime entry | Notes |
| --- | --- | --- | --- | --- |
| Reference Baselines | Long Context | `reference_long_context_agent.yaml` | `utils/agent.py` | Direct long-context answering baseline without an external memory store |
| Reference Baselines | Embedding RAG | `reference_embedding_rag.yaml` | `methods/embedding_rag/embedding_retriever.py` | Reference dense-retrieval baseline |
| Reference Baselines | BM25 RAG | `reference_simple_rag_bm25.yaml` | `utils/agent.py` | Sparse lexical retrieval baseline retained for comparison and smoke runs |
| Sequential Context Architectures | MemAgent | `sequential_memagent.yaml` | `methods/memagent/` | Recurrent sequential-memory baseline |
| Sequential Context Architectures | Mem0 | `sequential_mem0.yaml` | `methods/mem0/source/mem0/` | Sequential memory storage with persistent structured state |
| Sequential Context Architectures | MemoChat | `sequential_memochat.yaml` | `methods/memochat/memochat_adapter.py` | Sequential dialogue memory with rolling summaries |
| Structural Topological Architectures | Cognee | `topological_cognee.yaml` | `methods/cognee/source/cognee/` | Graph-structured memory runtime |
| Structural Topological Architectures | Zep Local | `topological_zep_local.yaml` | `methods/zep_local/main.py` | Local graph-memory service path |
| Structural Topological Architectures | MemTree | `topological_memtree.yaml` | `methods/memtree/memtree_adapter.py` | Tree-structured memory organization with provenance |
| Structural Topological Architectures | GraphRAG | `topological_graph_rag.yaml` | `methods/graph_rag/graph_rag.py` | Structured graph-based retrieval baseline |
| Structural Topological Architectures | HippoRAG | `topological_hippo_rag_v2_openai.yaml` | `methods/hipporag/` | Retrieval over graph-style document organization |
| Structural Topological Architectures | RAPTOR | `topological_raptor.yaml` | `methods/raptor/raptor.py` | Hierarchical cluster-and-summarize retrieval baseline |
| Structural Topological Architectures | Zep | `topological_zep.yaml` | `methods/zep/zep.py` | Cloud-backed graph-memory integration |
| Multi-Paradigm Hybrid Architectures | Letta | `hybrid_letta.yaml` | `utils/agent.py` | Integrated through vendored Letta source and local runtime management |
| Multi-Paradigm Hybrid Architectures | LightMem | `hybrid_lightmem.yaml` | `methods/lightmem/lightmem_adapter.py` | Layered memory construction and retrieval |
| Multi-Paradigm Hybrid Architectures | SimpleMem | `hybrid_simplemem.yaml` | `methods/simplemem/simplemem_adapter.py` | Hybrid semantic, keyword, and structured retrieval |
| Multi-Paradigm Hybrid Architectures | MemOS | `hybrid_memos.yaml` | `methods/MemOS/source/src/` | Vendored memory operating system runtime |
| Multi-Paradigm Hybrid Architectures | MemoryOS | `hybrid_memoryos.yaml` | `methods/memoryos/memoryos_adapter.py` | Local runtime wrapper for the preserved MemoryOS implementation |
| Multi-Paradigm Hybrid Architectures | A-MEM | `hybrid_a_mem.yaml` | `methods/a_mem/a_mem_adapter.py` | Hybrid memory writing and retrieval with provenance tracking |
| Multi-Paradigm Hybrid Architectures | EverOS | `hybrid_everos.yaml` | `methods/everos/everos_adapter.py` | Search-oriented external memory runtime |
| Multi-Paradigm Hybrid Architectures | Self-RAG | `hybrid_self_rag.yaml` | `methods/self_rag/self_rag.py` | Retrieval-augmented generation baseline retained in the current code release |
| Multi-Paradigm Hybrid Architectures | MemoRAG | `hybrid_memo_rag.yaml` | `methods/memorag/` | Cache-heavy retrieval pipeline for long contexts |


## 📊 Benchmark Overview

| Benchmark family | Config files | Task focus | Expected input format |
| --- | --- | --- | --- |
| MemoryAgentBench / Accurate Retrieval | `benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml`<br>`benchmark/memoryagentbench/Accurate_Retrieval/config/LongMemEval/Longmemeval_s.yaml` | Question answering and long-memory retrieval under curated MemoryAgentBench splits | HuggingFace `save_to_disk` copy under `datasets/MemoryAgentBench/eval_dataset_collection/`, or fallback to `ai-hyz/MemoryAgentBench` |
| MemoryAgentBench / Conflict Resolution | `benchmark/memoryagentbench/Conflict_Resolution/config/Factconsolidation_mh_6k.yaml` | Resolving conflicting facts across long interaction histories | Same MemoryAgentBench loading path as above |
| MemoryAgentBench / Test-Time Learning | `benchmark/memoryagentbench/Test_Time_Learning/config/ICL/ICL_banking77.yaml` | In-context adaptation and label-space memorization | Same MemoryAgentBench loading path as above |
| LoCoMo | `benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml`<br>`benchmark/locomo/config/Locomo_qa_4cat_600_dist_cat1_multi_hop.yaml`<br>`benchmark/locomo/config/Locomo_qa_4cat_600_dist_cat2_temporal.yaml`<br>`benchmark/locomo/config/Locomo_qa_4cat_600_dist_cat3_open_domain.yaml`<br>`benchmark/locomo/config/Locomo_qa_4cat_600_dist_cat4_single_hop.yaml` | Conversational QA over long dialogues, with full and category-specific subsets | JSON file, typically `datasets/LoCoMo/rq1_4cat_600_dist/locomo_4cat_600_dist.json` |
| LongBench | `benchmark/longbench/config/LongBench_rep150_proportional.yaml` | Long-context multiple-choice reasoning on the proportional subset used by the current preset | HuggingFace `save_to_disk` directory, typically `datasets/longBench_rep150_proportional/datasets` |
| MemBench | `benchmark/membench/config/MemBench_simple.yaml`<br>`benchmark/membench/config/MemBench_noisy.yaml`<br>`benchmark/membench/config/MemBench_knowledge_update.yaml`<br>`benchmark/membench/config/MemBench_highlevel.yaml`<br>`benchmark/membench/config/MemBench_RecMultiSession.yaml` | Memory stress tests covering simple recall, noise, knowledge updates, high-level reasoning, and multi-session recommendation | Slice-specific JSON files under `datasets/MemBench/MemData/FirstAgent/` |


## ⚙️ Configuration Conventions

| Field | Meaning |
| --- | --- |
| `provider` | Chat-model backend type, typically `openai_compatible` in the default presets |
| `base_url` | Endpoint for the chat model server |
| `embedding_provider` | Backend type for embedding generation when the method uses vector retrieval |
| `embedding_base_url` | Endpoint for the embedding model server |
| `*_api_key_env` | Environment variable name used to resolve API keys at runtime |
| `retrieve_num` | Retrieval depth or top-`k` used by retrieval-enabled methods |


## 📦 Output Artifacts

| Artifact type | Default location | Description |
| --- | --- | --- |
| Result JSON | `results/outputs/<model>/<dataset>/<name_tag>_results.json` | Main evaluation output with metrics, query-level records, and summary fields |
| Agent states | `results/agents/` | Persisted agent memory, retrieval caches, and method-specific state |
| Artifact root override | `--artifact_root /path/to/artifacts` | Rebases the outer artifact root while keeping the internal layout unchanged |

Artifact layout:

```text
results/
├── outputs/                                     # evaluation outputs grouped by model and dataset
│   └── <model>/                                 # model or preset-specific output namespace
│       └── <dataset>/                           # benchmark-specific output namespace
│           └── <name_tag>_results.json          # primary result file with metrics and records
├── agents/                                      # persisted agent state and method-side caches
│   └── <model_or_method>/                       # runtime-specific storage namespace
└── logs/                                        # optional execution logs when enabled by the run
```

Example:

```bash
python main.py \
  --agent_config config/reference_long_context_agent.yaml \
  --dataset_config benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml \
  --artifact_root /path/to/artifacts
```

When `--artifact_root` is specified, the pipeline preserves the same internal `results/outputs`, `results/agents`, and `results/logs` organization under the new root, which makes it straightforward to isolate repeated experiment batches while keeping downstream parsing and post-processing logic unchanged.


## 🤔 FAQ

<details>
<summary><b>Are the datasets bundled with the repository?</b></summary>
<br/>
No. Datasets are not distributed here. Place them under <code>datasets/</code> following the paths in the <a href="#-quick-start">Quick Start</a> section. MemoryAgentBench additionally falls back to the <code>ai-hyz/MemoryAgentBench</code> HuggingFace mirror when no local copy is present.
</details>

<details>
<summary><b>Do I need to rebuild or reinstall anything between runs?</b></summary>
<br/>
No. MemoryData is a plain Python pipeline launched via <code>python main.py</code>. Switching methods or benchmarks is just a matter of pointing <code>--agent_config</code> and <code>--dataset_config</code> at different YAML files.
</details>

<details>
<summary><b>Which model providers are supported?</b></summary>
<br/>
The default presets target OpenAI-compatible chat and embedding endpoints, so any provider that exposes that interface works. Update <code>base_url</code>, <code>embedding_base_url</code>, and the relevant <code>*_api_key_env</code> fields in the chosen preset to match your server.
</details>

<details>
<summary><b>How do I force a clean re-run?</b></summary>
<br/>
Pass <code>--force</code> to delete saved results, rebuild local agent state, and reset supported external persistence before the run. Use <code>--retry_failed_queries</code> to retry previously failed queries instead of skipping them when resuming.
</details>


## 📒 Citation

If you find this benchmark suite useful in your research, please cite:

```bibtex
@article{zhoumemorydata2026,
    title={Are We Ready For An Agent-Native Memory System?},
    author={Wei Zhou and Xuanhe Zhou and Shaokun Han and Hongming Xu and Guoliang Li and Zhiyu Li and Feiyu Xiong and Fan Wu},
    year={2026},
    journal={arXiv preprint arXiv:2606.24775},
    url={https://arxiv.org/abs/2606.24775}
}
```
