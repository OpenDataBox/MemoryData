# Manuscript verification - 2026-09-03

This audit concerns the written artifact, not the scientific performance of MTM.

| Requirement | Verified evidence |
|---|---|
| Independent framework-based manuscript | New title, abstract, seven main sections, typed-memory algorithm, and implementation contracts; no excluded prior-method name or dependency in the manuscript |
| Both supplied illustrations in the main text | Figure 1 is on PDF page 3 in Section 3; Figure 2 is on page 7, before Section 5. Both original PNGs and their copies match by SHA-256. No illustration appendix remains |
| Existing Related Work incorporated | All 38 original citation keys retained; original bibliography copied byte-for-byte |
| References resolve | 44 bibliography entries and 44 cited keys; zero missing keys and zero duplicate keys; final PDF contains no unresolved `??` markers |
| Prior-work limitations are explicit | Introduction, closest-overlap paragraph, and limitations distinguish relational binding, stale derived knowledge, and ungrounded feedback; existing systems' overlapping capabilities are acknowledged |
| Concise two-benchmark design | LoCoMo categories 1–4 and LongMemEval cleaned S only. Accuracy/cost, three individual component removals, and native-category/evidence analysis; no synthetic correction benchmark. Three descriptive subsections, without R1/R2/RQ experiment labels |
| Dataset scope | Local LoCoMo JSON has 10 conversations and 1,540 category 1–4 questions (282/321/96/841); 446 category 5 questions are explicitly excluded. Official LongMemEval documentation specifies 500 cleaned-S questions. No LongMemEval benchmark run or local full-dataset validation is claimed |
| LoCoMo example grounded | Local official `conv-26`: D7:11 on July 12, 2023; D17:10 on October 13, 2023; Becoming Nicole answer; both evidence IDs. Official evaluation code identifies category 1 as multi-hop; the earlier misleading category description was corrected |
| No evaluation-label feedback | Explicit evaluator-only warning in the example section and original-figure caption; frozen snapshot protocol for official QA |
| No fabricated findings | Abstract, experimental section, limitations, appendix, and page header identify the empirical status as pending; no measured superiority is claimed |
| Official formatting | ICLR 2027 style and bibliography style match downloaded official files; working-copy title overrides do not modify the style files |
| Main-text length | Conclusion ends on page 9; full PDF has 13 portrait pages including references and the implementation appendix. Both figures are within the main-text page budget |
| Compilation | Tectonic completed with exit code 0; no unresolved citations, no overfull boxes, and no replacement characters in extracted PDF text |
| Visual QA | All 13 pages of the revised PDF were rendered to PNG and visually inspected, including both main-text figures, the benchmark table, experimental subsections, references, and appendix. No clipping, overlap, or missing glyphs found |

The TeX toolchain emitted non-fatal underfull-box warnings, a package-comment UTF-8 warning, and duplicate destination warnings for float anchors. These did not produce missing glyphs or visible layout defects. Final review covered the rendered artifact, not just the compiler's exit status.

Protocol checks used the [LoCoMo repository](https://github.com/snap-research/locomo), [LoCoMo official evaluator](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py), [LongMemEval repository](https://github.com/xiaowu0162/LongMemEval), and [LongMemEval official evaluator](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py). LongMemEval abstention is selected by its question-ID suffix; gold answer-session and answer-bearing-turn annotations remain evaluator-only.

Not completed or claimed: integrated MTM implementation, learned-procedure evaluation, official-baseline execution, benchmark measurements, or an acceptance prediction. These are future research work. The completed deliverables are the revised manuscript with both supplied figures in the main text and a concise experimental design restricted to the two requested benchmarks.
