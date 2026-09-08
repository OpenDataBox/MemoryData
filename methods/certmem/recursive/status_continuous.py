"""Compact read-only status, including only completed candidate summaries."""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run_dir',type=Path)
    ap.add_argument('--since',type=int,default=0)
    args=ap.parse_args()
    def read(name):
        return json.loads((args.run_dir/name).read_text())
    status=read('status.json')
    history=read('history.json')
    results=[]
    for row in history['rounds'][args.since:]:
        record={k:row[k] for k in ('name','parent','accepted')}
        if 'dev' in row:
            record.update({k:row['dev'][k] for k in ('official_f1','memory_tokens','by_category','cited_source_recall')})
        else:
            record['error']=row['error']
        results.append(record)
    print(json.dumps({'status':status,'completed':len(history['rounds']),
                      'winner':history['winner'],'best_f1':history['best_f1'],'new_results':results}))


if __name__=='__main__':
    main()
