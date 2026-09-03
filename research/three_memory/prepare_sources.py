"""Fetch public citation metadata and the official ICLR style, without modifying other papers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
STYLE_URL = 'https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip'
PAPERS = {
    'locomo': '2402.17753',
    'longmemeval': '2410.10813',
    'mem0': '2504.19413',
    'amem': '2502.12110',
    'simplemem': '2601.02553',
    'lightmem': '2510.18866',
    'reflexion': '2303.11366',
    'adamem': '2606.05684',
    'reme': '2512.10696',
    'recuris': '2608.24876',
}


class Metadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'meta' and values.get('name', '').startswith('citation_'):
            self.tags.setdefault(values['name'], []).append(values.get('content', ''))


def fetch(url):
    request = Request(url, headers={'User-Agent': 'Academic-manuscript-source-check/1.0'})
    with urlopen(request, timeout=60) as response:
        return response.read()


def latex(value):
    return value.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def paper(item):
    key, identifier = item
    url = f'https://arxiv.org/abs/{identifier}'
    page = fetch(url)
    parser = Metadata()
    parser.feed(page.decode('utf-8'))
    tags = parser.tags
    title = tags['citation_title'][0]
    authors = tags['citation_author']
    date = tags.get('citation_date', tags.get('citation_online_date', ['']))[0]
    year_match = re.search(r'20\d{2}', date)
    if not authors or not year_match:
        raise RuntimeError(f'Incomplete metadata for {url}: {tags}')
    year = year_match.group()
    bib = '\n'.join([
        f'@misc{{{key},',
        '  title = {{' + latex(title) + '}},',
        '  author = {' + ' and '.join(latex(name) for name in authors) + '},',
        f'  year = {{{year}}},',
        '  howpublished = {arXiv:' + identifier + '},',
        '  url = {' + url + '}',
        '}',
    ])
    record = {'key': key, 'title': title, 'authors': authors, 'date': date,
              'url': url, 'html_sha256': sha256(page).hexdigest()}
    return bib, record


def main():
    style_archive = fetch(STYLE_URL)
    style_hashes = {}
    with ZipFile(BytesIO(style_archive)) as archive:
        for filename in ['iclr2027_conference.sty', 'iclr2027_conference.bst']:
            matches = [name for name in archive.namelist() if name.endswith('/' + filename) or name == filename]
            if len(matches) != 1:
                raise RuntimeError(f'Unexpected style archive contents for {filename}: {matches}')
            content = archive.read(matches[0])
            (ROOT / filename).write_bytes(content)
            style_hashes[filename] = sha256(content).hexdigest()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(paper, PAPERS.items()))
    entries, records = zip(*results)
    langmem = '\n'.join([
        '@misc{langmem,',
        '  author = {{LangChain}},',
        '  title = {{LangMem: Core Concepts}},',
        '  year = {2026},',
        '  note = {Online documentation, accessed September 3, 2026},',
        '  url = {https://langchain-ai.github.io/langmem/concepts/conceptual_guide/}',
        '}',
    ])
    (ROOT / 'references.bib').write_text(
        '% Metadata fetched from primary public sources.\n\n' + '\n\n'.join(entries) + '\n\n' + langmem + '\n',
        encoding='utf-8',
    )
    manifest = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'style_url': STYLE_URL,
        'style_zip_sha256': sha256(style_archive).hexdigest(),
        'style_files_sha256': style_hashes,
        'papers': list(records),
        'langmem': 'https://langchain-ai.github.io/langmem/concepts/conceptual_guide/',
        'manuscript_status': 'Original proposal and prospective experiments; no measured results.',
    }
    (ROOT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'style_files': style_hashes, 'papers': [{k: r[k] for k in ['key', 'title', 'date']} for r in records]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
