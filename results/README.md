# Published experiment snapshot — 2026-09-08

This snapshot collects the completed local baseline results, recursive memory
experiments, research notes, and the code needed to inspect them. The exact
capture time and SHA-256 hashes are recorded in
[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json).

## Completed baseline

| Method | Dataset / scope | Backbone | Official token F1 | Status |
| --- | --- | --- | --- | --- |
| Mem0 | LoCoMo, 10 conversations, all 1,540 non-adversarial questions | Qwen3.5-9B FP16 | 36.49836% | Complete; no missing, duplicate, or empty predictions |

- [Final report, category scores, usage and recovery accounting](locomo_full_20260908/mem0/final_r5/locomo-finish-mem0-composite-r5/report.json)
- [1,540 per-question predictions](locomo_full_20260908/mem0/final_r5/locomo-finish-mem0-composite-r5/predictions.jsonl)
- [Normalized detailed results](locomo_full_20260908/mem0/final_r5/locomo-finish-mem0-composite-r5/normalized_results.json)
- [Independent verification](locomo_full_20260908/mem0/independent-verification.json)
- [Execution plans, pinned scorer, source archives and environment](locomo_full_20260908/provenance/)
- [Baseline progress at the recorded observation time](locomo_full_20260908/STATUS.md)

E-MEM and A-MEM do not have completed full-run results in this local snapshot.
Their progress observations are not final scores. The older
[comparison queue handoff](../COMPARISON_QUEUE_STATUS.md) is historical context;
use the dated result reports for completion and score claims.

[Publication validation](PUBLICATION_VALIDATION.json) records the 1,540-row
re-score and the offline unit-test checks, including local dependency limits.

The selected-run and all-attempt usage journals are included as lossless
`selected_usage.jsonl.gz` and `all_attempt_usage.jsonl.gz` alongside the final
report. They preserve the accounting for the failed attempt and recovery.
The publication manifest records both compressed and original hashes. To use
tools expecting plain JSONL, decompress them into a separate local directory.

## Recursive memory research

These experiments have different question subsets and protocols from the full
Mem0 baseline; their scores are not directly comparable to the baseline row.

| Collection | Contents | Entry point |
| --- | --- | --- |
| Recursive LoCoMo pilot | Original 100-question pilot, raw RAG controls, development histories and successive memory constructions | [Pilot report](recursive_locomo_20260907/REPORT_KO.md), [progress](recursive_locomo_20260907/PROGRESS_KO.md) |
| Methodological refinement | Evidence, parent-memory, retrieval, identity, audit, index, transfer, payload and route experiments; predictions and validation records | [Research progress](methodological_refinement_20260908/CURRENT_PROGRESS_KO.md), [protocol](methodological_refinement_20260908/PROTOCOL_KO.md) |
| Implementation | Experiment runners, analysis, verification and unit tests | [CertMem](../methods/certmem/README.md), [recursive protocol](../methods/certmem/recursive/README_KO.md) |

Development-set improvements are reported as development results. The reports
describe the question/conversation boundaries and limitations of the separate
100-question checkpoint. A snapshot may include an in-progress experiment's
protocol without a completed result; completion must be checked in its report.

## Included and excluded files

Included: authored code and documents, predictions, metrics, source provenance,
protocols, learned vectors, memory snapshots, and generated-answer/checker
records used by the verification scripts. Some evidence directories end in
`_cache`; those contain experimental observations needed for verification.

Excluded: virtual environments, installed parser dependencies, model weights,
raw benchmark downloads, temporary PDF renders and TeX build files, operational
logs, database/runtime state, redundant result-download archives, and repeated
raw Mem0 execution directories. Small immutable source archives are retained
because they identify the exact evaluated implementation. The pinned tokenizer
reference is retained for offline token-count verification.

Original local files were preserved. The older download-verification records
also describe local archives intentionally omitted from Git; use the publication
manifest to verify this published file set. Absolute paths in historical research
records refer to the machine where each experiment ran.

## Verify the published snapshot

From the repository root, using Python 3.10 or newer (standard library only):

```bash
python scripts/verify_published_results.py
```

This checks every published result's size and SHA-256 digest, and verifies that
compressed usage journals reproduce their original bytes. Re-scoring Mem0 also
requires the original LoCoMo dataset (its expected hash is in the provenance).
Use `scripts/score_locomo_comparison.py` with that dataset, the published
predictions, the pinned scorer, and the decompressed selected usage journal.
