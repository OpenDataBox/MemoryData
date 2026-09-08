"""Compact status for the controlled construction batch."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    args=ap.parse_args()
    read=lambda path:json.loads(path.read_text())
    result={'status':read(args.run/'status.json')}
    if (args.run/'reference_gate.json').exists():
        result['reference_gate']=read(args.run/'reference_gate.json')
    if (args.run/'source_verification.json').exists():
        verification=read(args.run/'source_verification.json')
        result['verification']=dict(Counter(v['verdict'] for group in verification.values() for v in group.values()))
    if (args.run/'history.json').exists():
        history=read(args.run/'history.json')
        result.update({'best_f1':history['best_f1'],'winner':history['winner'],
            'rounds':[{'name':r['name'],'f1':r['dev']['official_f1'],'memory_tokens':r['dev']['memory_tokens'],
                       'read_tokens':r['dev']['read_tokens'],'accepted':r['accepted']} for r in history['rounds']]})
    print(json.dumps(result))


if __name__=='__main__':
    main()
