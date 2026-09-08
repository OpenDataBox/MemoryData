# Brain-inspired multi-timescale memory manuscript

An independent method draft based on the supplied `framework.png` and `framework_locomo_example.png`. MTM is a working name. The algorithm is proposed; no benchmark superiority is claimed.

## Files

- `main.tex`: complete English manuscript, algorithm, two main-text figures, a two-benchmark experimental design, limitations, disclosure, and implementation appendix.
- `related_work.tex`: adaptation of the existing Related Work, preserving all 38 original citation keys.
- `related_work_20260903.bib`: unchanged copy of the existing bibliography.
- `references_extra.bib`: cognitive motivation and closest revision/consolidation studies.
- `paper_direction_ko.md`: Korean explanation of the argument, prior-work problems, figure caveats, and concise experiments.
- `figures/`: unchanged copies of both supplied illustrations.

Official ICLR 2027 style files are included unchanged. The working-copy title says this is not yet a submission. Remove the working-copy presentation overrides only for an actual submission; do not enable the accepted-paper flag.

## Build

From this directory:

```powershell
& 'D:\MemoryData\.runtime-cache\paper-tools\tectonic\tectonic.exe' --keep-logs --keep-intermediates --outdir 'D:\MemoryData\tmp\pdfs\brain_memory' main.tex
```

After compilation and visual QA, the deliverable is `D:\MemoryData\output\pdf\brain_inspired_multitimescale_memory.pdf`. Build files are not experimental results.

## Scope and evidence

The LoCoMo example's dates, speakers, turn IDs, question, answer, and official category were checked against the local official dataset. Category 1 is multi-hop according to the official evaluator. Its illustrated memory operations are not execution logs. Both supplied PNGs appear in the main text: the framework in Section 3 and the example in Section 4.

Evaluation uses only LoCoMo categories 1–4 (1,540 questions across ten conversations) and LongMemEval cleaned S (500 questions). Accuracy/cost, three core-module removals, and native-category analysis share these two benchmarks; no synthetic correction benchmark is added. Official QA must not feed gold labels, scoring signals, or previous evaluation questions into memory. Reconsolidation may occur while ingesting the native history chronologically; each question starts from an isolated history-only snapshot.

An integrated implementation, validated procedure induction, official-baseline runs, and the three matched experiments are still needed for empirical claims. No existing method's tests or scores are represented as evidence for this new framework.
