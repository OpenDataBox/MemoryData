import concurrent.futures
import json
import urllib.parse
from pathlib import Path
from fetch_sources import fetch

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'primary_metadata.json').read_text(encoding='utf-8'))

def query(key):
    title = data[key]['metadata']['citation_title'][0]
    url = 'https://api.openalex.org/works?search=' + urllib.parse.quote(title) + '&per-page=3&select=id,title,doi,publication_year,cited_by_count,ids'
    try:
        result = json.loads(fetch(url))
        return key, {'accessed': '2026-09-03', 'request_url': url, 'results': result.get('results', [])}
    except Exception as exc:
        return key, {'accessed': '2026-09-03', 'request_url': url, 'error': str(exc)}

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    result = dict(pool.map(query, ['rag', 'generative_agents', 'reflexion', 'memgpt', 'memorybank', 'hipporag']))
(ROOT / 'citation_snapshot.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=True))
