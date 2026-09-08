"""Build an independently keyed bibliography and a readable manuscript draft."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'parser_deps'))
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

sys.stdout.reconfigure(encoding='utf-8')
PAPER = ROOT.parent.parent / 'paper'
metadata = json.loads((ROOT / 'primary_metadata.json').read_text(encoding='utf-8'))
publisher = json.loads((ROOT / 'publisher_sources.json').read_text(encoding='utf-8'))

conference = {
 'simplemem': ('2026', 'Proceedings of the 43rd International Conference on Machine Learning', 'https://icml.cc/Downloads/2026'),
 'adamem': ('2026', 'Proceedings of the 43rd International Conference on Machine Learning', 'https://icml.cc/Downloads/2026'),
 'longmemeval': ('2025', 'International Conference on Learning Representations', 'https://github.com/xiaowu0162/LongMemEval'),
 'memoryagentbench': ('2026', 'International Conference on Learning Representations', 'https://github.com/HUST-AI-HYZ/MemoryAgentBench'),
 'memtree': ('2025', 'International Conference on Learning Representations', 'https://www.yujia.io/'),
 'amabench': ('2026', 'Proceedings of the 43rd International Conference on Machine Learning', 'https://github.com/AMA-Bench/AMA-Bench/'),
}

replacements = {
 '’': "'", '‘': "'", '–': '--', '—': '---',
 'ü': r'\"{u}', 'ä': r'\"{a}', 'ö': r'\"{o}',
 'é': r"\'{e}", 'ó': r"\'{o}", 'í': r"\'{i}",
}

def latex(s):
 for old,new in replacements.items(): s=s.replace(old,new)
 return s

entries=[]
audit={}
for key,record in metadata.items():
 if record.get('error'): raise ValueError((key,record['error']))
 m=record['metadata']
 pub=publisher.get(key,{})
 if pub.get('bib'):
  raw_bib=re.sub(r'\bmonth\s*=\s*sept\b', 'month = {September}', pub['bib'], flags=re.I)
  parsed=bibtexparser.loads(raw_bib).entries
  if len(parsed)!=1: raise ValueError(f'Invalid publisher bibliography: {key}')
  entry=parsed[0]
  entry={k:v for k,v in entry.items() if k in {'ENTRYTYPE','ID','author','title','booktitle','journal','volume','number','pages','year','doi','url','publisher'}}
  if not entry.get('author') or not entry.get('title'): raise ValueError(key)
  source=pub['source']
 else:
  entry={
   'ENTRYTYPE':'misc','ID':'',
   'title':m['citation_title'][0],
   'author':' and '.join(m['citation_author']),
   'year':m['citation_date'][0][:4],
   'eprint':record['arxiv'], 'archiveprefix':'arXiv',
   'doi':'10.48550/arXiv.'+record['arxiv'], 'url':record['url'],
  }
  source=record['url']
 if key in conference:
  year,booktitle,source=conference[key]
  entry.update(ENTRYTYPE='inproceedings', year=year, booktitle=booktitle)
  entry.pop('doi',None)
 if key == 'voyager':
  entry.update(ENTRYTYPE='article', year='2024', journal='Transactions on Machine Learning Research')
  entry.pop('doi',None)
  source='https://rpl.cs.utexas.edu/publications/2024/03/14/wang-tmlr24-voyager/'
 entry['ID']='mrw_'+key
 entry['title']='{'+entry['title']+'}'
 entry={k:latex(v) for k,v in entry.items()}
 entries.append(entry)
 audit[key]={
  'citekey':entry['ID'], 'title':m['citation_title'][0],
  'year':entry['year'], 'venue':entry.get('booktitle',entry.get('journal','arXiv preprint')),
  'bibliographic_source':source, 'paper_url':record['url'],
  'accessed':'2026-09-03',
 }

db=BibDatabase(); db.entries=entries
writer=BibTexWriter(); writer.order_entries_by=None
writer.indent='  '
bib=('% Verified on 2026-09-03. Independent mrw_ keys avoid key collisions with sample.bib.\n'
     '% Publisher records take precedence over arXiv metadata for published papers.\n'
     '% Conference listings are documented in the accompanying research notes.\n\n'+writer.write(db))
bibpath=PAPER/'related_work_20260903.bib'
bibpath.write_text(bib,encoding='utf-8')
(ROOT/'bibliography_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')

tex=(PAPER/'related_work_20260903.tex').read_text(encoding='utf-8')
plain=re.sub(r'^%.*\n','',tex,flags=re.M)
plain=re.sub(r'\\section\{[^}]+\}', '# Related Work', plain)
plain=re.sub(r'\\label\{[^}]+\}\n','',plain)
plain=re.sub(r'\\paragraph\{([^}]+)\}',r'**\1**',plain)
def citation(match):
 links=[]
 for citekey in match.group(1).split(','):
  key=citekey.strip().removeprefix('mrw_')
  author=metadata[key]['metadata']['citation_author'][0].split(',')[0]
  links.append(f'[{author} et al., {audit[key]["year"]}]({audit[key]["paper_url"]})')
 return ' ('+'; '.join(links)+')'
plain=re.sub(r'~?\\citep\{([^}]+)\}',citation,plain)
plain=plain.replace('~',' ')
plain=('검토 기준: 2026-09-03. LLM agent의 외부·장기 메모리에 초점을 둔 영문 초안. '
       '최근 선정 논문은 2026년 8월까지 포함한다.\n\n'+plain)
(PAPER/'related_work_20260903.md').write_text(plain,encoding='utf-8')

parsed=bibtexparser.loads(bib).entries
keys=[e['ID'] for e in parsed]
cited={k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in group.split(',')}
assert len(keys)==len(set(keys))==len(metadata)
assert cited==set(keys),(cited-set(keys),set(keys)-cited)
for entry in parsed:
 assert all(entry.get(f) for f in ['author','title','year','url']),entry['ID']
 assert int(entry['year'])<=2026
assert '\\cite' not in plain
words=len(re.findall(r"\b[\w'-]+\b",re.sub(r'\\\w+\{[^}]*\}','',re.sub(r'^%.*$','',tex,flags=re.M))))
validation={'bibtex_entries':len(keys),'unique_citations':len(cited),
            'missing_keys':[],'unused_keys':[],'approximate_body_words':words,
            'bibtex_parse':'passed','latex_compilation':'not run; no TeX compiler found',
            'checked_on':'2026-09-03'}
(ROOT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps(validation,indent=2))
