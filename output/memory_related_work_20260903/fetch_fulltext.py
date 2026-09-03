import concurrent.futures
import json
import re
import sys
from pathlib import Path
from fetch_sources import fetch, clean

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
KEYS = ['memoryeab', 'lightmem', 'simplemem', 'adamem', 'leanmem', 'mesa', 'execution_state', 'hindsight']
data = json.loads((ROOT / 'primary_metadata.json').read_text(encoding='utf-8'))

def run(key):
    aid = data[key]['arxiv']
    url = 'https://arxiv.org/html/' + aid
    path = ROOT / (aid + '_fulltext.html')
    try:
        raw = path.read_text(encoding='utf-8') if path.exists() else fetch(url)
        path.write_text(raw, encoding='utf-8')
        paragraphs = [clean(p) for p in re.findall(r'<p\b[^>]*>(.*?)</p>',raw,re.S)]
        paragraphs = [p for p in paragraphs if len(p) > 120]
        (ROOT / (aid + '_paragraphs.json')).write_text(json.dumps(paragraphs, ensure_ascii=False, indent=2), encoding='utf-8')
        return {'key':key,'url':url,'paragraph_count':len(paragraphs), 'selected':paragraphs[3:7]}
    except Exception as e:
        return {'key':key,'url':url,'error':str(e)}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run, KEYS):
        print(json.dumps(result, ensure_ascii=False))
