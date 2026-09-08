"""Export the same manuscript as Markdown, a PDF copy, and an Overleaf ZIP."""
from pathlib import Path
import json
import re
import shutil
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
CITATIONS = {
    "rmm": ("Tan et al., 2025", "https://aclanthology.org/2025.acl-long.413/"),
    "higmem": ("Cao et al., 2026", "https://aclanthology.org/2026.findings-acl.1690/"),
    "hmem": ("Sun et al., 2026", "https://aclanthology.org/2026.eacl-long.15/"),
    "seem": ("Lu et al., 2026", "https://aclanthology.org/2026.acl-long.277/"),
    "tiermem": ("Zhu et al., 2026", "https://arxiv.org/abs/2602.17913v1"),
    "locomo": ("Maharana et al., 2024", "https://aclanthology.org/2024.acl-long.747/"),
    "longmemeval": ("Wu et al., 2025", "https://arxiv.org/abs/2410.10813"),
    "trustmem": ("Yang et al., 2026", "https://arxiv.org/abs/2606.25161"),
    "demem": ("Zou et al., 2026", "https://arxiv.org/abs/2605.10870"),
}

def convert(text):
    text = re.sub(r"\\citep\{([^}]+)\}", lambda m: "(" + "; ".join(f"[{CITATIONS[k][0]}]({CITATIONS[k][1]})" for k in m[1].split(",")) + ")", text)
    text = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\textbf\{([^}]+)\}", r"**\1**", text)
    text = text.replace("Equation~\\ref{eq:objective}", "Equation (1)")
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = text.replace("\\begin{equation}", "$$").replace("\\end{equation}", "$$")
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return "\n\n".join(re.sub(r"\s*\n\s*", " ", p).strip() if "$$" not in p else p for p in paragraphs)

def main():
    source = (ROOT / "main.tex").read_text(encoding="utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S)[1]
    introduction = source.split("\\section{Introduction}", 1)[1].split("\\bibliography", 1)[0]
    title = "CertMem: Evidence Residuals for Token-Efficient Agent Memory"
    markdown = f"# {title}\n\n> Working draft: proposed framework; empirical evaluation pending.\n\n## Abstract\n\n{convert(abstract)}\n\n## 1 Introduction\n\n{convert(introduction)}\n"
    (ROOT / "draft_en.md").write_text(markdown, encoding="utf-8")
    target = WORKSPACE / "output" / "pdf" / "certmem_abstract_introduction_iclr.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "main.pdf", target)
    archive_path = ROOT / "certmem_iclr_source.zip"
    filenames = ["main.tex", "references.bib", "iclr2027_conference.sty", "iclr2027_conference.bst", "draft_en.md", "reading_notes_ko.md", "source_manifest.json", "build.ps1"]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in filenames:
            archive.write(ROOT / filename, arcname=filename)
    pdf = PdfReader(target)
    extracted = "\n".join(p.extract_text() or "" for p in pdf.pages)
    log = (ROOT / "main.log").read_text(encoding="utf-8", errors="replace")
    issues = [line for line in log.splitlines() if any(s in line for s in ["Overfull", "undefined", "LaTeX Warning", "Package natbib Warning"])]
    if issues:
        raise ValueError(issues)
    normalized = re.sub(r"\s+", "", extracted).lower()
    assert all(term in normalized for term in ["abstract", "introduction", "references"])
    assert all(name in extracted for name in ["RMM", "HiGMem", "H-MEM", "SEEM", "TierMem"])
    result = {"pages": len(pdf.pages), "abstract_words": len(abstract.split()), "pdf": str(target), "source_zip": str(archive_path), "tex_warnings": issues, "page_text_lengths": [len(p.extract_text() or "") for p in pdf.pages], "visual_review": "pending"}
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
