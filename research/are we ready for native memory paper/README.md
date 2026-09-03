# Related Work integration

`related_work.tex` is written for the ICLR LaTeX style (`natbib` commands
`\citep` and `\citet`). Insert it after the introduction:

```tex
\input{related_work}
```

The citation keys intentionally match the current arXiv source where possible.
`related_work_citations.bib` is a standalone, verified export. If the paper's
existing `sample.bib` is retained, merge these records by key instead of loading
both files: in arXiv v1, the LoCoMo record lists the wrong authors, the
MemoryAgentBench record uses a placeholder group author, the MemTree URL is
stale, and some newer system/survey records contain outdated metadata.

If the paper uses plain `\cite` rather than `natbib` author--year commands,
replace `\citep` and `\citet` mechanically after integration.
