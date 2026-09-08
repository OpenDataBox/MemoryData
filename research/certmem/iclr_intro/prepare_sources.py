"""Fetch primary-source bibliographic metadata and the official ICLR style."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import re
import zipfile
from urllib.request import Request, urlopen
from types import SimpleNamespace
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parent
ANTHOLOGY = {
    "rmm": "2025.acl-long.413",
    "higmem": "2026.findings-acl.1690",
    "hmem": "2026.eacl-long.15",
    "seem": "2026.acl-long.277",
    "locomo": "2024.acl-long.747",
}
ARXIV = {
    "tiermem": "2602.17913",
    "longmemeval": "2410.10813",
    "trustmem": "2606.25161",
    "demem": "2605.10870",
}

def get(url):
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=45) as response:
        data = response.read()
    return SimpleNamespace(content=data, text=data.decode("utf-8", errors="replace"))

class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and "name" in attrs and "content" in attrs:
            self.values.setdefault(attrs["name"], []).append(attrs["content"])

def anthology(item):
    key, identifier = item
    url = f"https://aclanthology.org/{identifier}.bib"
    bib = get(url).text
    if not bib.lstrip().startswith("@"):
        raise ValueError(f"Not BibTeX: {url}")
    bib = re.sub(r"(@\w+\{)[^,]+,", lambda m: m[1] + key + ",", bib, count=1)
    # Abstracts are not required by the bibliography and greatly increase file size.
    bib = re.sub(r'\n\s*abstract\s*=\s*".*?"\s*(?=\n\})', '', bib, flags=re.S)
    return key, bib, {"url": url, "sha256": hashlib.sha256(bib.encode()).hexdigest()}

def arxiv(item):
    key, identifier = item
    url = f"https://arxiv.org/abs/{identifier}"
    html = get(url).text
    parser = Metadata()
    parser.feed(html)
    def metas(name):
        return parser.values.get(name, [])
    title = metas("citation_title")[0]
    authors = metas("citation_author")
    year = metas("citation_date")[0][:4]
    if key == "longmemeval":
        bib = (f"@inproceedings{{{key},\n  title = {{{{{title}}}}},\n"
               f"  author = {{{' and '.join(authors)}}},\n  year = {{2025}},\n"
               "  booktitle = {The Thirteenth International Conference on Learning Representations},\n"
               f"  url = {{{url}}}\n}}\n")
    else:
        bib = (f"@misc{{{key},\n  title = {{{{{title}}}}},\n"
               f"  author = {{{' and '.join(authors)}}},\n  year = {{{year}}},\n"
               f"  howpublished = {{arXiv:{identifier}}},\n  url = {{{url}}}\n}}\n")
    return key, bib, {"url": url, "title": title, "authors": authors, "sha256": hashlib.sha256(html.encode()).hexdigest()}

def main():
    manifest = {"checked_at": "2026-09-08", "references": {}}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(anthology, i) for i in ANTHOLOGY.items()]
        futures += [executor.submit(arxiv, i) for i in ARXIV.items()]
        records = [f.result() for f in futures]
    (ROOT / "references.bib").write_text("\n".join(x[1] for x in records), encoding="utf-8")
    for key, _, metadata in records:
        manifest["references"][key] = metadata
    url = "https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip"
    data = get(url).content
    manifest["style_url"] = url
    manifest["style_zip_sha256"] = hashlib.sha256(data).hexdigest()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for filename in ["iclr2027_conference.sty", "iclr2027_conference.bst"]:
            candidates = [n for n in archive.namelist() if n.rsplit("/", 1)[-1] == filename and not n.startswith("__MACOSX")]
            if len(candidates) != 1:
                raise ValueError(candidates)
            content = archive.read(candidates[0])
            (ROOT / filename).write_bytes(content)
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, _, metadata in records:
        print(key, metadata.get("title", metadata["url"]))
    print("Official ICLR style retrieved.")

if __name__ == "__main__":
    main()
