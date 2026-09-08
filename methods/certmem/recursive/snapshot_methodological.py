"""Snapshot committed construction comparisons without mutable caches."""
import argparse
import io
import json
from pathlib import Path
import tarfile
import refine as core


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    ap.add_argument('archive',type=Path)
    ap.add_argument('--since',type=int,default=0)
    args=ap.parse_args()
    history=json.loads((args.run/'history.json').read_text())
    protocol=json.loads((args.run/'protocol.json').read_text())
    names=[r['name'] for r in history['rounds'][args.since:]]
    with tarfile.open(args.archive.with_suffix('.tmp'),'w:gz') as archive:
        payload=json.dumps(history,indent=2).encode()
        item=tarfile.TarInfo('run/history.json')
        item.size=len(payload)
        archive.addfile(item,io.BytesIO(payload))
        for name in ('protocol.json','status.json','reference_gate.json','reference_replay_dev_items.json',
                     'source_verification.json','verification','options','source_history.json',
                     'source_baseline_contract.json','source_selection_locked.json','source_trials',
                     'query_views','query_view_lock.json','previous_method_fit_contract.json',
                     'view_audit','view_audit_summary.json','event_graph','event_graph_locked.json',
                     'indexes','training','optimizer','transforms','base_memories','training_locked.json',
                     'original_reference_dev_items.json','source_payloads','source_payloads_locked.json','payload_memories_locked.json',
                     'route_candidates','route_geometry','routes_locked.json','route_memories_locked.json'):
            path=args.run/name
            if path.exists():
                archive.add(path,arcname='run/'+name)
        for path in sorted(args.run.glob('*_source_fit_items.json')):
            archive.add(path,arcname='run/'+path.name)
        for path in sorted(args.run.glob('*_source_view_*_items.json')):
            archive.add(path,arcname='run/'+path.name)
        generated=args.run/'cache/structured_generations'
        if generated.exists():
            archive.add(generated,arcname='run/query_view_generation_cache')
        event_generated=args.run/'cache/contrastive_event_generations'
        if event_generated.exists():
            archive.add(event_generated,arcname='run/event_generation_cache')
        payload_generated=args.run/'cache/provenance_payload_generations'
        if payload_generated.exists():
            archive.add(payload_generated,arcname='run/payload_generation_cache')
            from provenance_payload import CHECKER
            from contrastive_events import original_context
            sessions=json.loads(Path('research_data/source_sessions.json').read_text())['sessions_by_id']
            seen=set()
            for path in sorted((args.run/'source_payloads').glob('*.json')):
                for item in json.loads(path.read_text()).values():
                    for fact in item['facts']:
                        cited={e['source_id'] for e in fact['evidence']}
                        remaining=set(item['window']['window_ids'])-cited
                        contexts=[original_context(sessions[path.stem],cited),original_context(sessions[path.stem],remaining) if remaining else '(no conversation evidence)']
                        for context in contexts:
                            user=f'Original conversation:\n{context}\n\nStatement: {fact["text"]}\n\nVerdict:'
                            key=core.digest([protocol['environment']['models'][protocol['args']['model']],protocol['args']['seed'],CHECKER,user,16,False])
                            if key not in seen:
                                archive.add(args.run/'cache/generations'/f'{key}.json',arcname=f'run/payload_checker_cache/{key}.json')
                                seen.add(key)
        if any(r.get('family')=='source_event_distinction' for r in protocol['recipes']):
            parents={}
            for cid in sorted({r['conv_id'] for r in protocol['records']}):
                path=Path('runs/continuous_v3/memories/r40_fused_four_turn')/f'{cid}.json'
                parents[cid]=json.loads(path.read_text())
                archive.add(path,arcname=f'run/graph_parents/{cid}.json')
                key=core.digest([protocol['environment']['models'][protocol['args']['embed_model']],
                                 [u['text'] for u in parents[cid]],False,protocol['args']['embed_batch_size'],'oom_backoff_v1'])
                archive.add(args.run/'cache/embeddings'/f'{key}.npy',arcname=f'run/graph_parent_embeddings/{cid}.npy')
            assert core.digest(parents)==protocol['parent_memory_sha256']
        if any(r.get('family') in ('source_supervised_memory_index','source_grounded_payload','source_factual_routes','source_route_identity') for r in protocol['recipes']):
            for split,filename in (('dev','reference_replay_dev_items.json'),
                                   ('q0','source_baseline_source_fit_items.json'),
                                   ('fit_a','fit_baseline_source_view_fit_items.json')):
                rows=json.loads((args.run/filename).read_text())
                for cid in sorted({r['conv_id'] for r in rows}):
                    selected=[r for r in rows if r['conv_id']==cid]
                    questions=[r['question'] for r in selected]
                    key=core.digest([protocol['environment']['models'][protocol['args']['embed_model']],
                                     questions,True,protocol['args']['embed_batch_size'],'oom_backoff_v1'])
                    archive.add(args.run/'cache/embeddings'/f'{key}.npy',arcname=f'run/evaluation_queries/{split}/{cid}.npy')
                    payload=json.dumps({'ids':[r['id'] for r in selected],'cache_key':key}).encode()
                    item=tarfile.TarInfo(f'run/evaluation_queries/{split}/{cid}.json')
                    item.size=len(payload)
                    archive.addfile(item,io.BytesIO(payload))
        for name in names:
            archive.add(args.run/f'{name}_dev_items.json',arcname=f'run/{name}_dev_items.json')
            archive.add(args.run/'plans'/f'{name}.json',arcname=f'run/plans/{name}.json')
            for directory in ('memories','construction'):
                archive.add(args.run/directory/name,arcname=f'run/{directory}/{name}')
        for name in protocol['source_sha256']:
            archive.add(Path(name),arcname='source/'+name)
    args.archive.with_suffix('.tmp').replace(args.archive)
    print(json.dumps({'included':len(names),'completed_total':len(history['rounds']),'delta_since':args.since,
                      'best_f1':history['best_f1'],'winner':history['winner'],
                      'archive_bytes':args.archive.stat().st_size}))


if __name__=='__main__':
    main()
