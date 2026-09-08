"""Collect primary paper metadata and a dated citation-count snapshot."""
import concurrent.futures
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)

PAPERS = {
    'rag': '2005.11401',
    'generative_agents': '2304.03442',
    'reflexion': '2303.11366',
    'memgpt': '2310.08560',
    'memorybank': '2305.10250',
    'hipporag': '2405.14831',
    'mem0': '2504.19413',
    'amem': '2502.12110',
    'zep': '2501.13956',
    'memoryos': '2506.06326',
    'lightmem': '2510.18866',
    'simplemem': '2601.02553',
    'adamem': '2606.05684',
    'hindsight': '2512.12818',
    'mem1': '2506.15841',
    'leanmem': '2608.03463',
    'mesa': '2608.10108',
    'execution_state': '2606.06090',
    'survey': '2512.13564',
    'memoryeab': '2604.01707',
    'locomo': '2402.17753',
    'longmemeval': '2410.10813',
    'locomo_plus': '2602.10715',
    'memoryagentbench': '2507.05257',
    'voyager': '2305.16291',
    'readagent': '2402.09727',
    'memtree': '2410.14052',
    'hipporag2': '2502.14802',
    'memos': '2507.03724',
    'memory_r1': '2508.19828',
    'memrl': '2601.03192',
    'memgen': '2509.24704',
    'magma': '2601.03236',
    'memagent': '2507.02259',
    'membench': '2506.21605',
    'memoryarena': '2602.16313',
    'amabench': '2602.22769',
    'survey2024': '2404.13501',
}

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'MemoryData-Literature-Review/1.0'})
    with urllib.request.urlopen(req, timeout=35) as response:
        return response.read().decode('utf-8')

def clean(s):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s))).strip()

class MetadataParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'meta':
            return
        attrs = dict(attrs)
        name = attrs.get('name', attrs.get('property', ''))
        if name.startswith('citation_'):
            self.values.setdefault(name, []).append(attrs.get('content', ''))

def paper(item):
    key, aid = item
    url = 'https://arxiv.org/abs/' + aid
    try:
        cache = ROOT / (aid + '.html')
        if cache.exists():
            raw = cache.read_text(encoding='utf-8')
        else:
            raw = fetch(url)
            cache.write_text(raw, encoding='utf-8')
        parser = MetadataParser()
        parser.feed(raw)
        metas = parser.values
        abstract = re.search(r'<blockquote[^>]*class="abstract[^"]*"[^>]*>(.*?)</blockquote>', raw, re.S)
        comments = re.search(r'<td[^>]*class="tablecell comments"[^>]*>(.*?)</td>', raw, re.S)
        return key, {'arxiv': aid, 'url': url, 'metadata': metas,
                     'abstract': clean(abstract.group(1)) if abstract else '',
                     'comments': clean(comments.group(1)) if comments else '',
                     'accessed': '2026-09-03'}
    except Exception as exc:
        return key, {'arxiv': aid, 'url': url, 'error': str(exc)}

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        data = dict(pool.map(paper, PAPERS.items()))
    (ROOT / 'primary_metadata.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    for key, p in data.items():
        m = p.get('metadata', {})
        print(json.dumps({'key': key, 'title': m.get('citation_title'),
                          'date': m.get('citation_date'), 'authors': m.get('citation_author'),
                          'comments': p.get('comments'), 'abstract': p.get('abstract'),
                          'error': p.get('error')}, ensure_ascii=False))

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
