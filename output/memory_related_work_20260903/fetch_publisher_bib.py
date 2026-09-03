import concurrent.futures
import html
import json
import re
import sys
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from fetch_sources import fetch

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
PAGES = {
 'rag': 'https://proceedings.neurips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html',
 'reflexion': 'https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html',
 'hipporag': 'https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html',
 'amem': 'https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html',
 'lightmem': 'https://proceedings.iclr.cc/paper_files/paper/2026/hash/a05b72653ec5b473732129829ae04195-Abstract-Conference.html',
 'locomo': 'https://aclanthology.org/2024.acl-long.747/',
 'memoryos': 'https://aclanthology.org/2025.emnlp-main.1318/',
 'magma': 'https://aclanthology.org/2026.acl-long.1709/',
 'membench': 'https://aclanthology.org/2025.findings-acl.989/',
 'readagent': 'https://proceedings.mlr.press/v235/lee24c.html',
 'hipporag2': 'https://proceedings.mlr.press/v267/gutierrez25a.html',
}

class Links(HTMLParser):
 def __init__(self): super().__init__(); self.urls=[]
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if tag=='a' and any(x in attrs.get('href','').lower() for x in ['.bib','bibtex']): self.urls.append(attrs['href'])

def run(item):
 key,url=item
 try:
  if 'proceedings.mlr.press' in url:
   raw=fetch(url)
   match=re.search(r'@InProceedings\{.*?^\s*\}',raw,re.I|re.S|re.M)
   if not match: return key, {'source':url,'error':'No inline publisher BibTeX'}
   bib=html.unescape(match.group(0))
   (ROOT/(key+'_publisher.bib')).write_text(bib,encoding='utf-8')
   return key, {'source':url,'bib_url':url+'#bibtex','bib':bib}
  raw=fetch(url); parser=Links(); parser.feed(raw)
  if 'aclanthology.org' in url: parser.urls=[url.rstrip('/')+'.bib']
  if not parser.urls and 'aclanthology.org' in url: parser.urls.append(url.rstrip('/')+'.bib')
  if not parser.urls: return key, {'source':url,'error':'No BibTeX link'}
  biburl=urllib.parse.urljoin(url,parser.urls[0])
  bib=fetch(biburl)
  (ROOT/(key+'_publisher.bib')).write_text(bib,encoding='utf-8')
  return key, {'source':url,'bib_url':biburl,'bib':bib}
 except Exception as exc: return key, {'source':url,'error':str(exc)}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=dict(pool.map(run,PAGES.items()))
for key,doi in [('generative_agents','10.1145/3586183.3606763'),('memorybank','10.1609/aaai.v38i17.29946'),('survey2024','10.1145/3748302')]:
 url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')+'/transform/application/x-bibtex'
 try:
  bib=fetch(url); (ROOT/(key+'_publisher.bib')).write_text(bib,encoding='utf-8')
  results[key]={'source':'https://doi.org/'+doi,'bib_url':url,'bib':bib}
 except Exception as exc: results[key]={'source':url,'error':str(exc)}
(ROOT/'publisher_sources.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
for key,result in results.items(): print(key, result.get('error') or result['bib'])
